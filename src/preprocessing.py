"""
Módulo de pré-processamento de dados para o projeto de predição de seguros.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import joblib

from .config import (
    RANDOM_STATE,
    TEST_SIZE,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSFORMATION_METHODS,
    FEATURE_SELECTION_METHODS,
    OUTLIER_METHODS,
    PROCESSED_DATA_DIR
)

# Configurar logging
logger = logging.getLogger(__name__)

# Suprimir warnings específicos
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DataPreprocessor:
    """
    Classe responsável pelo pré-processamento completo dos dados.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Inicializa o preprocessor.
        
        Args:
            random_state: Semente para reprodutibilidade.
        """
        self.random_state = random_state
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.polynomial_features = None
        self.preprocessor_pipeline = None
        self.feature_names_ = None
        self.outlier_detector = None
        
        # Estatísticas para transformações
        self.transformation_stats = {}
        
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas duplicadas do dataset.
        
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
        else:
            logger.info("Nenhuma linha duplicada encontrada")
            
        return data_clean
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                       columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detecta outliers nos dados.
        
        Args:
            data: DataFrame de entrada.
            method: Método para detecção ('iqr', 'zscore', 'isolation_forest').
            columns: Colunas para análise. Se None, usa todas as numéricas.
            
        Returns:
            Dicionário com informações dos outliers.
        """
        if columns is None:
            columns = NUMERICAL_COLUMNS
            
        outliers_info = {}
        
        for col in columns:
            if col in data.columns:
                outliers_info[col] = {}
                
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outliers_info[col]['method'] = 'IQR'
                    outliers_info[col]['bounds'] = (lower_bound, upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[col]))
                    outliers_mask = z_scores > 3
                    outliers_info[col]['method'] = 'Z-Score'
                    outliers_info[col]['threshold'] = 3
                
                outliers_info[col]['count'] = outliers_mask.sum()
                outliers_info[col]['percentage'] = (outliers_mask.sum() / len(data)) * 100
                outliers_info[col]['indices'] = data[outliers_mask].index.tolist()
                
                logger.info(f"Coluna {col}: {outliers_mask.sum()} outliers ({outliers_mask.sum()/len(data)*100:.2f}%)")
        
        return outliers_info
    
    def handle_outliers(self, data: pd.DataFrame, method: str = 'cap', 
                       outliers_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Trata outliers nos dados.
        
        Args:
            data: DataFrame de entrada.
            method: Método de tratamento ('remove', 'cap', 'log_transform').
            outliers_info: Informações dos outliers da detecção.
            
        Returns:
            DataFrame com outliers tratados.
        """
        data_processed = data.copy()
        
        if outliers_info is None:
            outliers_info = self.detect_outliers(data)
        
        for col, info in outliers_info.items():
            if info['count'] > 0:
                if method == 'cap':
                    if 'bounds' in info:  # IQR method
                        lower_bound, upper_bound = info['bounds']
                        data_processed[col] = data_processed[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"Outliers em {col} limitados entre {lower_bound:.2f} e {upper_bound:.2f}")
                
                elif method == 'remove':
                    outlier_indices = info['indices']
                    data_processed = data_processed.drop(outlier_indices)
                    logger.info(f"Removidas {len(outlier_indices)} linhas com outliers em {col}")
        
        return data_processed
    
    def apply_transformations(self, data: pd.DataFrame, 
                            target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Aplica transformações de normalização aos dados.
        
        Args:
            data: DataFrame de entrada.
            target_col: Nome da coluna target (para transformações específicas).
            
        Returns:
            DataFrame com transformações aplicadas.
        """
        data_transformed = data.copy()
        
        # Transformação logarítmica para variáveis com distribuição assimétrica
        skewed_cols = []
        for col in NUMERICAL_COLUMNS:
            if col in data_transformed.columns:
                skewness = data_transformed[col].skew()
                if abs(skewness) > 1:  # Consideramos assimétrico se |skewness| > 1
                    skewed_cols.append(col)
                    logger.info(f"Coluna {col} tem skewness = {skewness:.3f}")
        
        # Aplicar transformação Box-Cox para variáveis assimétricas (valores positivos)
        for col in skewed_cols:
            if col in data_transformed.columns and (data_transformed[col] > 0).all():
                try:
                    transformed_data, lambda_param = boxcox(data_transformed[col])
                    data_transformed[f'{col}_boxcox'] = transformed_data
                    self.transformation_stats[f'{col}_boxcox'] = {'lambda': lambda_param, 'method': 'boxcox'}
                    logger.info(f"Aplicada transformação Box-Cox em {col} (λ={lambda_param:.3f})")
                except:
                    logger.warning(f"Não foi possível aplicar Box-Cox em {col}")
        
        # Transformação Yeo-Johnson (funciona com valores negativos também)
        for col in skewed_cols:
            if col in data_transformed.columns:
                try:
                    transformed_data, lambda_param = yeojohnson(data_transformed[col])
                    data_transformed[f'{col}_yeojohnson'] = transformed_data
                    self.transformation_stats[f'{col}_yeojohnson'] = {'lambda': lambda_param, 'method': 'yeojohnson'}
                    logger.info(f"Aplicada transformação Yeo-Johnson em {col} (λ={lambda_param:.3f})")
                except:
                    logger.warning(f"Não foi possível aplicar Yeo-Johnson em {col}")
        
        return data_transformed
    
    def create_polynomial_features(self, data: pd.DataFrame, degree: int = 2,
                                 columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Cria features polinomiais.
        
        Args:
            data: DataFrame de entrada.
            degree: Grau dos polinômios.
            columns: Colunas para criar features. Se None, usa todas as numéricas.
            
        Returns:
            DataFrame com features polinomiais.
        """
        if columns is None:
            columns = NUMERICAL_COLUMNS
        
        # Selecionar apenas colunas que existem nos dados
        available_columns = [col for col in columns if col in data.columns]
        
        if not available_columns:
            logger.warning("Nenhuma coluna numérica encontrada para criar features polinomiais")
            return data
        
        # Criar features polinomiais
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        
        # Aplicar apenas nas colunas numéricas originais
        X_numeric = data[available_columns]
        X_poly = poly.fit_transform(X_numeric)
        
        # Criar nomes para as novas features
        feature_names = poly.get_feature_names_out(available_columns)
        
        # Criar DataFrame com features polinomiais
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=data.index)
        
        # Combinar com dados originais (excluindo colunas numéricas originais para evitar duplicação)
        data_with_poly = data.copy()
        
        # Adicionar apenas as novas features (grau > 1)
        new_features = [col for col in feature_names if col not in available_columns]
        for feature in new_features:
            data_with_poly[feature] = poly_df[feature]
        
        self.polynomial_features = poly
        logger.info(f"Criadas {len(new_features)} features polinomiais de grau {degree}")
        
        return data_with_poly
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação específicas para o domínio de seguros.
        
        Args:
            data: DataFrame de entrada.
            
        Returns:
            DataFrame com features de interação.
        """
        data_interactions = data.copy()
        
        # Interações específicas para predição de seguros
        if 'age' in data.columns and 'smoker' in data.columns:
            # Idade * Fumante (fumantes mais velhos têm maior risco)
            smoker_numeric = (data['smoker'] == 'yes').astype(int)
            data_interactions['age_smoker_interaction'] = data['age'] * smoker_numeric
            
        if 'bmi' in data.columns and 'smoker' in data.columns:
            # BMI * Fumante (BMI alto + fumante = risco muito alto)
            data_interactions['bmi_smoker_interaction'] = data['bmi'] * smoker_numeric
            
        if 'age' in data.columns and 'bmi' in data.columns:
            # Idade * BMI (pessoas mais velhas com BMI alto)
            data_interactions['age_bmi_interaction'] = data['age'] * data['bmi']
            
        if 'children' in data.columns and 'age' in data.columns:
            # Filhos * Idade (familias maiores com responsável mais velho)
            data_interactions['children_age_interaction'] = data['children'] * data['age']
        
        # Feature categórica: risco combinado
        if all(col in data.columns for col in ['age', 'bmi', 'smoker']):
            # Categorizar risco baseado em idade, BMI e hábito de fumar
            risk_score = (
                (data['age'] > 50).astype(int) +  # Idade > 50
                (data['bmi'] > 30).astype(int) +  # BMI > 30 (obesidade)
                (data['smoker'] == 'yes').astype(int)  # Fumante
            )
            data_interactions['risk_category'] = risk_score
            
        logger.info(f"Criadas {len(data_interactions.columns) - len(data.columns)} features de interação")
        
        return data_interactions
    
    def encode_categorical_features(self, data: pd.DataFrame, 
                                  method: str = 'onehot') -> pd.DataFrame:
        """
        Codifica variáveis categóricas.
        
        Args:
            data: DataFrame de entrada.
            method: Método de encoding ('onehot', 'label', 'target').
            
        Returns:
            DataFrame com variáveis categóricas codificadas.
        """
        data_encoded = data.copy()
        
        categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in data.columns]
        
        if not categorical_cols:
            logger.info("Nenhuma variável categórica encontrada")
            return data_encoded
        
        if method == 'onehot':
            # One-Hot Encoding
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            
            # Aplicar encoding
            encoded_data = encoder.fit_transform(data[categorical_cols])
            
            # Criar nomes das colunas
            feature_names = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i]
                # Pular a primeira categoria (dropped)
                for cat in categories[1:]:
                    feature_names.append(f"{col}_{cat}")
            
            # Criar DataFrame com dados codificados
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
            
            # Remover colunas categóricas originais e adicionar codificadas
            data_encoded = data_encoded.drop(columns=categorical_cols)
            data_encoded = pd.concat([data_encoded, encoded_df], axis=1)
            
            self.encoder = encoder
            logger.info(f"Aplicado One-Hot Encoding em {len(categorical_cols)} colunas categóricas")
            
        elif method == 'label':
            # Label Encoding
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                data_encoded[col] = le.fit_transform(data[col])
                label_encoders[col] = le
            
            self.encoder = label_encoders
            logger.info(f"Aplicado Label Encoding em {len(categorical_cols)} colunas categóricas")
        
        return data_encoded
    
    def scale_features(self, data: pd.DataFrame, method: str = 'standard',
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aplica escalonamento às features numéricas.
        
        Args:
            data: DataFrame de entrada.
            method: Método de escalonamento ('standard', 'minmax', 'robust').
            exclude_columns: Colunas para não escalonar.
            
        Returns:
            DataFrame com features escalonadas.
        """
        data_scaled = data.copy()
        
        # Identificar colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir colunas específicas se fornecidas
        if exclude_columns:
            numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        if not numeric_cols:
            logger.warning("Nenhuma coluna numérica encontrada para escalonamento")
            return data_scaled
        
        # Escolher scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalonamento não suportado: {method}")
        
        # Aplicar escalonamento
        data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        self.scaler = scaler
        logger.info(f"Aplicado escalonamento {method} em {len(numeric_cols)} colunas numéricas")
        
        return data_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'selectkbest', n_features: int = 10) -> pd.DataFrame:
        """
        Seleciona as melhores features.
        
        Args:
            X: Features de entrada.
            y: Variável target.
            method: Método de seleção ('selectkbest', 'selectfrommodel', 'rfe').
            n_features: Número de features a selecionar.
            
        Returns:
            DataFrame com features selecionadas.
        """
        if method == 'selectkbest':
            # SelectKBest com f_regression
            selector = SelectKBest(score_func=f_regression, k=n_features)
            
        elif method == 'selectfrommodel':
            # SelectFromModel com ExtraTreesRegressor
            estimator = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state)
            selector = SelectFromModel(estimator, max_features=n_features)
            
        elif method == 'rfe':
            # RFE com LinearRegression
            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=n_features)
            
        else:
            raise ValueError(f"Método de seleção não suportado: {method}")
        
        # Aplicar seleção
        X_selected = selector.fit_transform(X, y)
        
        # Obter nomes das features selecionadas
        if hasattr(selector, 'get_support'):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f"feature_{i}" for i in range(X_selected.shape[1])]
        
        # Criar DataFrame com features selecionadas
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_selector = selector
        self.feature_names_ = selected_features
        
        logger.info(f"Selecionadas {len(selected_features)} features usando {method}")
        logger.info(f"Features selecionadas: {selected_features}")
        
        return X_selected_df
    
    def create_preprocessing_pipeline(self, include_feature_selection: bool = True,
                                    feature_selection_method: str = 'selectkbest',
                                    n_features: int = 10) -> Pipeline:
        """
        Cria pipeline completo de pré-processamento.
        
        Args:
            include_feature_selection: Se deve incluir seleção de features.
            feature_selection_method: Método de seleção de features.
            n_features: Número de features a selecionar.
            
        Returns:
            Pipeline de pré-processamento.
        """
        # Definir transformadores para colunas categóricas e numéricas
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        
        # Criar ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_COLUMNS),
                ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
            ]
        )
        
        # Criar pipeline
        if include_feature_selection:
            if feature_selection_method == 'selectkbest':
                feature_selector = SelectKBest(score_func=f_regression, k=n_features)
            elif feature_selection_method == 'selectfrommodel':
                estimator = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state)
                feature_selector = SelectFromModel(estimator, max_features=n_features)
            else:
                estimator = LinearRegression()
                feature_selector = RFE(estimator, n_features_to_select=n_features)
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', feature_selector)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor)
            ])
        
        self.preprocessor_pipeline = pipeline
        logger.info("Pipeline de pré-processamento criado")
        
        return pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta o pipeline e transforma os dados.
        
        Args:
            X: Features de entrada.
            y: Variável target (opcional).
            
        Returns:
            Dados transformados.
        """
        if self.preprocessor_pipeline is None:
            self.create_preprocessing_pipeline()
        
        if y is not None:
            X_transformed = self.preprocessor_pipeline.fit_transform(X, y)
        else:
            X_transformed = self.preprocessor_pipeline.fit_transform(X)
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados usando pipeline já ajustado.
        
        Args:
            X: Features de entrada.
            
        Returns:
            Dados transformados.
        """
        if self.preprocessor_pipeline is None:
            raise ValueError("Pipeline não foi ajustado. Execute fit_transform primeiro.")
        
        X_transformed = self.preprocessor_pipeline.transform(X)
        return X_transformed
    
    def save_preprocessor(self, filepath: Path) -> None:
        """
        Salva o preprocessor treinado.
        
        Args:
            filepath: Caminho para salvar o preprocessor.
        """
        preprocessor_data = {
            'pipeline': self.preprocessor_pipeline,
            'scaler': self.scaler,
            'encoder': self.encoder,
            'feature_selector': self.feature_selector,
            'polynomial_features': self.polynomial_features,
            'feature_names': self.feature_names_,
            'transformation_stats': self.transformation_stats
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor salvo em: {filepath}")
    
    def load_preprocessor(self, filepath: Path) -> None:
        """
        Carrega preprocessor salvo.
        
        Args:
            filepath: Caminho do preprocessor salvo.
        """
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor_pipeline = preprocessor_data['pipeline']
        self.scaler = preprocessor_data['scaler']
        self.encoder = preprocessor_data['encoder']
        self.feature_selector = preprocessor_data['feature_selector']
        self.polynomial_features = preprocessor_data['polynomial_features']
        self.feature_names_ = preprocessor_data['feature_names']
        self.transformation_stats = preprocessor_data['transformation_stats']
        
        logger.info(f"Preprocessor carregado de: {filepath}")


def preprocess_insurance_data(data: pd.DataFrame, 
                            target_column: str = TARGET_COLUMN,
                            test_size: float = TEST_SIZE,
                            remove_outliers: bool = True,
                            apply_transformations: bool = True,
                            create_polynomial: bool = True,
                            create_interactions: bool = True,
                            feature_selection: bool = True,
                            random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Função principal para pré-processamento completo dos dados de seguro.
    
    Args:
        data: DataFrame com os dados.
        target_column: Nome da coluna target.
        test_size: Proporção para conjunto de teste.
        remove_outliers: Se deve remover outliers.
        apply_transformations: Se deve aplicar transformações.
        create_polynomial: Se deve criar features polinomiais.
        create_interactions: Se deve criar features de interação.
        feature_selection: Se deve fazer seleção de features.
        random_state: Semente para reprodutibilidade.
        
    Returns:
        Dicionário com dados processados e metadados.
    """
    logger.info("Iniciando pré-processamento completo dos dados")
    
    # Inicializar preprocessor
    preprocessor = DataPreprocessor(random_state=random_state)
    
    # 1. Remover duplicatas
    data_clean = preprocessor.remove_duplicates(data)
    
    # 2. Separar features e target
    X = data_clean[FEATURE_COLUMNS].copy()
    y = data_clean[target_column].copy()
    
    # 3. Detectar e tratar outliers
    if remove_outliers:
        outliers_info = preprocessor.detect_outliers(X)
        X = preprocessor.handle_outliers(X, method='cap', outliers_info=outliers_info)
    
    # 4. Aplicar transformações
    if apply_transformations:
        X = preprocessor.apply_transformations(X)
    
    # 5. Criar features polinomiais
    if create_polynomial:
        X = preprocessor.create_polynomial_features(X, degree=2)
    
    # 6. Criar features de interação
    if create_interactions:
        X = preprocessor.create_interaction_features(X)
    
    # 7. Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # 8. Aplicar encoding e escalonamento
    X_train_processed = preprocessor.encode_categorical_features(X_train, method='onehot')
    X_test_processed = preprocessor.encode_categorical_features(X_test, method='onehot')
    
    # Garantir que test tenha as mesmas colunas que train
    missing_cols = set(X_train_processed.columns) - set(X_test_processed.columns)
    for col in missing_cols:
        X_test_processed[col] = 0
    
    X_test_processed = X_test_processed[X_train_processed.columns]
    
    # 9. Escalonamento
    exclude_target = [target_column] if target_column in X_train_processed.columns else []
    X_train_scaled = preprocessor.scale_features(X_train_processed, method='standard', 
                                               exclude_columns=exclude_target)
    X_test_scaled = X_test_processed.copy()
    X_test_scaled[X_train_scaled.columns] = preprocessor.scaler.transform(X_test_processed[X_train_scaled.columns])
    
    # 10. Seleção de features
    if feature_selection:
        n_features = min(15, X_train_scaled.shape[1])  # Máximo 15 features
        X_train_selected = preprocessor.select_features(X_train_scaled, y_train, 
                                                       method='selectkbest', n_features=n_features)
        X_test_selected = X_test_scaled[preprocessor.feature_names_]
    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
    
    # Criar resultado
    result = {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names_ if feature_selection else X_train_selected.columns.tolist(),
        'original_shape': data.shape,
        'processed_shape': X_train_selected.shape,
        'removed_duplicates': len(data) - len(data_clean)
    }
    
    logger.info(f"Pré-processamento concluído:")
    logger.info(f"  - Shape original: {data.shape}")
    logger.info(f"  - Shape final: {X_train_selected.shape}")
    logger.info(f"  - Features selecionadas: {len(result['feature_names'])}")
    
    return result


if __name__ == "__main__":
    # Exemplo de uso
    from .data_loader import load_insurance_data
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Carregar dados
        data, _ = load_insurance_data()
        
        # Preprocessar dados
        processed_data = preprocess_insurance_data(data)
        
        print(f"✅ Pré-processamento concluído!")
        print(f"Treino: {processed_data['X_train'].shape}")
        print(f"Teste: {processed_data['X_test'].shape}")
        print(f"Features: {processed_data['feature_names']}")
        
    except Exception as e:
        print(f"❌ Erro: {e}") 