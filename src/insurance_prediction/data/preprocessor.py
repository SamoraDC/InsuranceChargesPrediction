"""
Módulo de pré-processamento otimizado para Gradient Boosting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import joblib
import warnings

from ..config.settings import Config
from ..utils.logging import get_logger

# Configurar logger
logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DataPreprocessor:
    """
    Classe de pré-processamento otimizada para Gradient Boosting.
    
    O Gradient Boosting não requer normalização e se beneficia de:
    - Features categóricas bem encodadas
    - Features de interação específicas do domínio
    - Tratamento cuidadoso de outliers
    - Seleção de features relevantes
    """
    
    def __init__(self, random_state: int = None):
        """
        Inicializa o preprocessor.
        
        Args:
            random_state: Semente para reprodutibilidade.
        """
        self.random_state = random_state or Config.RANDOM_STATE
        self.label_encoders = {}
        self.feature_selector = None
        self.scaler = None  # Para consistência, mas GB não precisa
        self.selected_features = None
        self.preprocessing_stats = {}
        
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas duplicadas.
        
        Args:
            data: DataFrame de entrada.
            
        Returns:
            DataFrame sem duplicatas.
        """
        initial_rows = len(data)
        data_clean = data.drop_duplicates()
        removed_rows = initial_rows - len(data_clean)
        
        if removed_rows > 0:
            logger.info(f"Removidas {removed_rows} linhas duplicadas ({removed_rows/initial_rows*100:.2f}%)")
        
        self.preprocessing_stats['removed_duplicates'] = removed_rows
        return data_clean
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes.
        
        Args:
            data: DataFrame de entrada.
            
        Returns:
            DataFrame com valores ausentes tratados.
        """
        data_filled = data.copy()
        missing_info = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            
            if missing_count > 0:
                missing_pct = (missing_count / len(data)) * 100
                missing_info[col] = {"count": missing_count, "percentage": missing_pct}
                
                if col in Config.NUMERICAL_COLUMNS:
                    # Para GB, usar mediana é melhor que média (mais robusto a outliers)
                    fill_value = data[col].median()
                    data_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"Coluna {col}: {missing_count} valores preenchidos com mediana ({fill_value:.2f})")
                
                elif col in Config.CATEGORICAL_COLUMNS:
                    # Usar moda para categóricas
                    fill_value = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
                    data_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"Coluna {col}: {missing_count} valores preenchidos com moda ({fill_value})")
        
        self.preprocessing_stats['missing_values'] = missing_info
        return data_filled
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detecta outliers usando método IQR (mais conservador para GB).
        
        Args:
            data: DataFrame de entrada.
            method: Método de detecção ('iqr' recomendado para GB).
            
        Returns:
            Informações sobre outliers.
        """
        outliers_info = {}
        
        for col in Config.NUMERICAL_COLUMNS:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Usar multiplier mais conservador para GB (1.5 -> 2.0)
                multiplier = 2.0  # Mais conservador
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                outliers_info[col] = {
                    'count': outliers_count,
                    'percentage': (outliers_count / len(data)) * 100,
                    'bounds': (lower_bound, upper_bound),
                    'indices': data[outliers_mask].index.tolist()
                }
                
                logger.info(f"Coluna {col}: {outliers_count} outliers ({outliers_count/len(data)*100:.2f}%)")
        
        return outliers_info
    
    def handle_outliers(self, data: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Trata outliers de forma conservadora (GB é robusto a outliers).
        
        Args:
            data: DataFrame de entrada.
            method: Método de tratamento ('cap' recomendado).
            
        Returns:
            DataFrame com outliers tratados.
        """
        data_processed = data.copy()
        outliers_info = self.detect_outliers(data)
        total_capped = 0
        
        for col, info in outliers_info.items():
            if info['count'] > 0 and method == 'cap':
                lower_bound, upper_bound = info['bounds']
                
                # Aplicar capping
                data_processed[col] = data_processed[col].clip(lower=lower_bound, upper=upper_bound)
                total_capped += info['count']
                
                logger.info(f"Outliers em {col} limitados entre {lower_bound:.2f} e {upper_bound:.2f}")
        
        self.preprocessing_stats['outliers_capped'] = total_capped
        return data_processed
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica variáveis categóricas usando Label Encoding (melhor para GB).
        
        Args:
            X: Features a serem codificadas.
            fit: Se deve ajustar os encoders.
            
        Returns:
            DataFrame com variáveis codificadas.
        """
        X_encoded = X.copy()
        
        for col in Config.CATEGORICAL_COLUMNS:
            if col in X_encoded.columns:
                if fit:
                    # Criar e ajustar encoder
                    encoder = LabelEncoder()
                    X_encoded[col] = encoder.fit_transform(X_encoded[col].astype(str))
                    self.label_encoders[col] = encoder
                    
                    logger.info(f"Label encoding aplicado em {col}: {len(encoder.classes_)} categorias")
                else:
                    # Usar encoder já ajustado
                    if col in self.label_encoders:
                        encoder = self.label_encoders[col]
                        # Tratar categorias não vistas
                        X_encoded[col] = X_encoded[col].astype(str)
                        
                        # Mapear categorias conhecidas
                        known_categories = set(encoder.classes_)
                        X_encoded[col] = X_encoded[col].apply(
                            lambda x: x if x in known_categories else encoder.classes_[0]
                        )
                        X_encoded[col] = encoder.transform(X_encoded[col])
                    else:
                        logger.warning(f"Encoder para {col} não encontrado")
        
        return X_encoded
    
    def create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features específicas do domínio de seguros.
        
        Args:
            data: DataFrame de entrada.
            
        Returns:
            DataFrame com novas features.
        """
        data_enhanced = data.copy()
        new_features_count = 0
        
        # Features de interação críticas para seguros
        if all(col in data.columns for col in ['age', 'smoker']):
            # Idade * Fumante (risco composto)
            smoker_binary = (data['smoker'] == 'yes').astype(int)
            data_enhanced['age_smoker_risk'] = data['age'] * smoker_binary
            new_features_count += 1
        
        if all(col in data.columns for col in ['bmi', 'smoker']):
            # BMI * Fumante (obesidade + fumo = risco altíssimo)
            data_enhanced['bmi_smoker_risk'] = data['bmi'] * smoker_binary
            new_features_count += 1
        
        if all(col in data.columns for col in ['age', 'bmi']):
            # Idade * BMI
            data_enhanced['age_bmi_interaction'] = data['age'] * data['bmi']
            new_features_count += 1
        
        # Features categóricas derivadas
        if 'age' in data.columns:
            # Faixas etárias
            data_enhanced['age_group'] = pd.cut(
                data['age'], 
                bins=[0, 25, 35, 50, 65, 100], 
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            new_features_count += 1
        
        if 'bmi' in data.columns:
            # Categorias de BMI
            data_enhanced['bmi_category'] = pd.cut(
                data['bmi'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=[0, 1, 2, 3, 4]  # Underweight, Normal, Overweight, Obese, Severely Obese
            ).astype(int)
            new_features_count += 1
        
        # Feature de risco composto
        if all(col in data.columns for col in ['age', 'bmi', 'smoker']):
            risk_score = 0
            risk_score += (data['age'] > 50).astype(int) * 2  # Idade alta
            risk_score += (data['bmi'] > 30).astype(int) * 3  # Obesidade
            risk_score += (data['smoker'] == 'yes').astype(int) * 4  # Fumo (maior peso)
            
            data_enhanced['composite_risk_score'] = risk_score
            new_features_count += 1
        
        # Features de densidade populacional (aproximação pela região)
        if 'region' in data.columns:
            # Mapear regiões para densidade aproximada (encoding manual mais informativo)
            region_density = {
                'northeast': 3,  # Alta densidade
                'southeast': 2,  # Média-alta densidade
                'northwest': 1,  # Baixa densidade
                'southwest': 1   # Baixa densidade
            }
            data_enhanced['region_density'] = data['region'].map(region_density).fillna(1)
            new_features_count += 1
        
        logger.info(f"Criadas {new_features_count} features específicas do domínio")
        self.preprocessing_stats['new_features_created'] = new_features_count
        
        return data_enhanced
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'selectkbest', n_features: int = None) -> pd.DataFrame:
        """
        Seleciona features mais relevantes para GB.
        
        Args:
            X: Features de entrada.
            y: Variável target.
            method: Método de seleção.
            n_features: Número de features a selecionar.
            
        Returns:
            DataFrame com features selecionadas.
        """
        if n_features is None:
            n_features = min(Config.PREPROCESSING_CONFIG['n_features_to_select'], X.shape[1])
        
        # Para GB, usar SelectKBest com f_regression funciona bem
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Obter nomes das features selecionadas
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Criar DataFrame com features selecionadas
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        # Log das features selecionadas com seus scores
        feature_scores = selector.scores_[selector.get_support()]
        feature_ranking = list(zip(selected_features, feature_scores))
        feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Selecionadas {len(selected_features)} features de {X.shape[1]} disponíveis")
        logger.info("Top 10 features por score:")
        for i, (feature, score) in enumerate(feature_ranking[:10]):
            logger.info(f"  {i+1:2d}. {feature}: {score:.2f}")
        
        self.preprocessing_stats['feature_selection'] = {
            'total_features': X.shape[1],
            'selected_features': len(selected_features),
            'top_features': feature_ranking[:10]
        }
        
        return X_selected_df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Ajusta o preprocessor e transforma os dados.
        
        Args:
            X: Features de entrada.
            y: Variável target.
            
        Returns:
            Tupla com (X transformado, y).
        """
        logger.info("Iniciando pré-processamento (fit_transform)...")
        
        # 1. Tratar valores ausentes
        X_processed = self.handle_missing_values(X)
        
        # 2. Tratar outliers de forma conservadora
        X_processed = self.handle_outliers(X_processed, method='cap')
        
        # 3. Criar features específicas do domínio
        X_processed = self.create_domain_features(X_processed)
        
        # 4. Codificar variáveis categóricas
        X_processed = self.encode_categorical_features(X_processed, fit=True)
        
        # 5. Seleção de features
        if Config.PREPROCESSING_CONFIG['feature_selection']:
            X_processed = self.select_features(X_processed, y)
        
        logger.info(f"Pré-processamento concluído: {X_processed.shape}")
        
        return X_processed, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados usando preprocessor já ajustado.
        
        Args:
            X: Features de entrada.
            
        Returns:
            Dados transformados.
        """
        logger.info("Aplicando transformações...")
        
        # 1. Tratar valores ausentes
        X_processed = self.handle_missing_values(X)
        
        # 2. Tratar outliers
        X_processed = self.handle_outliers(X_processed, method='cap')
        
        # 3. Criar features do domínio
        X_processed = self.create_domain_features(X_processed)
        
        # 4. Codificar categóricas (sem fit)
        X_processed = self.encode_categorical_features(X_processed, fit=False)
        
        # 5. Selecionar features (se aplicável)
        if self.selected_features and self.feature_selector:
            # Garantir que todas as features necessárias estão presentes
            missing_features = set(self.selected_features) - set(X_processed.columns)
            if missing_features:
                logger.warning(f"Features ausentes: {missing_features}")
                # Adicionar features ausentes com valor 0
                for feature in missing_features:
                    X_processed[feature] = 0
            
            # Selecionar apenas as features treinadas
            X_processed = X_processed[self.selected_features]
        
        logger.info(f"Transformação concluída: {X_processed.shape}")
        
        return X_processed
    
    def save_preprocessor(self, filepath: Path) -> None:
        """
        Salva o preprocessor treinado.
        
        Args:
            filepath: Caminho para salvar.
        """
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'preprocessing_stats': self.preprocessing_stats,
            'config': {
                'random_state': self.random_state,
                'preprocessing_config': Config.PREPROCESSING_CONFIG
            }
        }
        
        # Criar diretório se necessário
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor salvo em: {filepath}")
    
    def load_preprocessor(self, filepath: Path) -> None:
        """
        Carrega preprocessor salvo.
        
        Args:
            filepath: Caminho do preprocessor.
        """
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_selector = preprocessor_data['feature_selector']
        self.selected_features = preprocessor_data['selected_features']
        self.preprocessing_stats = preprocessor_data['preprocessing_stats']
        
        logger.info(f"Preprocessor carregado de: {filepath}")
        logger.info(f"Features selecionadas: {len(self.selected_features) if self.selected_features else 0}")


def preprocess_insurance_data(data: pd.DataFrame, 
                            target_column: str = None,
                            test_size: float = None,
                            random_state: int = None) -> Dict[str, Any]:
    """
    Função principal para pré-processamento de dados de seguro para Gradient Boosting.
    
    Args:
        data: DataFrame com os dados.
        target_column: Nome da coluna target.
        test_size: Proporção para conjunto de teste.
        random_state: Semente para reprodutibilidade.
        
    Returns:
        Dicionário com dados processados e metadados.
    """
    target_column = target_column or Config.TARGET_COLUMN
    test_size = test_size or Config.TEST_SIZE
    random_state = random_state or Config.RANDOM_STATE
    
    logger.info("Iniciando pré-processamento para Gradient Boosting...")
    
    # Inicializar preprocessor
    preprocessor = DataPreprocessor(random_state=random_state)
    
    # 1. Remover duplicatas
    data_clean = preprocessor.remove_duplicates(data)
    
    # 2. Separar features e target
    X = data_clean[Config.FEATURE_COLUMNS].copy()
    y = data_clean[target_column].copy()
    
    # 3. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Divisão dos dados - Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # 4. Aplicar pré-processamento
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Compilar resultado
    result = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_processed,
        'y_test': y_test,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.selected_features or X_train_processed.columns.tolist(),
        'preprocessing_stats': preprocessor.preprocessing_stats,
        'original_shape': data.shape,
        'processed_shape': X_train_processed.shape
    }
    
    logger.info(f"Pré-processamento concluído:")
    logger.info(f"  - Shape original: {data.shape}")
    logger.info(f"  - Shape processado: {X_train_processed.shape}")
    logger.info(f"  - Features finais: {len(result['feature_names'])}")
    logger.info(f"  - Duplicatas removidas: {preprocessor.preprocessing_stats.get('removed_duplicates', 0)}")
    logger.info(f"  - Outliers tratados: {preprocessor.preprocessing_stats.get('outliers_capped', 0)}")
    
    return result


if __name__ == "__main__":
    # Teste do preprocessor
    from ..utils.logging import setup_logging
    from .loader import load_insurance_data
    
    setup_logging("INFO")
    
    try:
        # Carregar dados
        data, _ = load_insurance_data()
        
        # Preprocessar dados
        processed_data = preprocess_insurance_data(data)
        
        logger.info("✅ Pré-processamento de teste concluído!")
        logger.info(f"Treino: {processed_data['X_train'].shape}")
        logger.info(f"Teste: {processed_data['X_test'].shape}")
        logger.info(f"Features: {processed_data['feature_names'][:10]}...")  # Primeiras 10
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}") 