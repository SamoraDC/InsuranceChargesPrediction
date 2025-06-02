"""
Módulo de predição para modelos de prêmios de seguro.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import joblib
from datetime import datetime

from .config import (
    MODELS_DIR,
    MODEL_ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    CATEGORICAL_VALUES,
    NUMERICAL_RANGES
)

# Configurar logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class InsurancePredictor:
    """
    Classe para fazer predições de prêmios de seguro usando modelos treinados.
    """
    
    def __init__(self, model_path: Optional[Path] = None, 
                 preprocessor_path: Optional[Path] = None):
        """
        Inicializa o preditor.
        
        Args:
            model_path: Caminho para o modelo treinado.
            preprocessor_path: Caminho para o preprocessor.
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_info = None
        
        # Carregar modelo e preprocessor se fornecidos
        if model_path:
            self.load_model(model_path)
        
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: Path) -> None:
        """
        Carrega modelo treinado.
        
        Args:
            model_path: Caminho para o modelo.
        """
        try:
            if model_path.suffix == '.pkl':
                self.model = joblib.load(model_path)
                logger.info(f"Modelo carregado de: {model_path}")
                
                # Tentar carregar informações do modelo
                info_path = model_path.parent / "model_info.pkl"
                if info_path.exists():
                    self.model_info = joblib.load(info_path)
                    logger.info(f"Informações do modelo carregadas de: {info_path}")
            else:
                raise ValueError(f"Formato de arquivo não suportado: {model_path.suffix}")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: Path) -> None:
        """
        Carrega preprocessor treinado.
        
        Args:
            preprocessor_path: Caminho para o preprocessor.
        """
        try:
            preprocessor_data = joblib.load(preprocessor_path)
            
            # Extrair componentes do preprocessor
            self.preprocessor = preprocessor_data.get('pipeline')
            self.feature_names = preprocessor_data.get('feature_names')
            
            logger.info(f"Preprocessor carregado de: {preprocessor_path}")
            logger.info(f"Features esperadas: {len(self.feature_names) if self.feature_names else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar preprocessor: {e}")
            raise
    
    def validate_input(self, data: Union[Dict, pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Valida e normaliza dados de entrada.
        
        Args:
            data: Dados de entrada (dict, DataFrame ou Series).
            
        Returns:
            DataFrame validado.
        """
        # Converter para DataFrame se necessário
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Tipo de dados não suportado: {type(data)}")
        
        # Verificar se todas as colunas necessárias estão presentes
        missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltando: {missing_cols}")
        
        # Selecionar apenas colunas de features
        df = df[FEATURE_COLUMNS].copy()
        
        # Validar tipos de dados e valores
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                # Verificar valores categóricos válidos
                if col in CATEGORICAL_VALUES:
                    invalid_values = set(df[col]) - set(CATEGORICAL_VALUES[col])
                    if invalid_values:
                        raise ValueError(f"Valores inválidos em {col}: {invalid_values}")
                
                # Converter para string
                df[col] = df[col].astype(str)
        
        for col in NUMERICAL_COLUMNS:
            if col in df.columns:
                # Converter para numérico
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar valores numéricos válidos
                if col in NUMERICAL_RANGES:
                    min_val = NUMERICAL_RANGES[col]['min']
                    max_val = NUMERICAL_RANGES[col]['max']
                    
                    out_of_range = (df[col] < min_val) | (df[col] > max_val)
                    if out_of_range.any():
                        logger.warning(f"Valores fora do range em {col}: {df[col][out_of_range].tolist()}")
        
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
            # Preencher valores nulos com mediana/moda
            for col in df.columns:
                if df[col].isnull().any():
                    if col in NUMERICAL_COLUMNS:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', 
                                      inplace=True)
        
        logger.info(f"Dados validados: {df.shape}")
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Aplica pré-processamento aos dados.
        
        Args:
            data: DataFrame com dados validados.
            
        Returns:
            Array numpy com dados preprocessados.
        """
        if self.preprocessor is None:
            logger.warning("Preprocessor não carregado. Usando dados brutos.")
            # Encoding básico manual se não houver preprocessor
            processed_data = data.copy()
            
            # One-hot encoding manual
            for col in CATEGORICAL_COLUMNS:
                if col in processed_data.columns:
                    dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
                    processed_data.drop(col, axis=1, inplace=True)
            
            return processed_data.values
        
        try:
            # Usar preprocessor carregado
            processed_data = self.preprocessor.transform(data)
            logger.info(f"Dados preprocessados: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            raise
    
    def predict(self, data: Union[Dict, pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Faz predição completa com validação e formatação.
        
        Args:
            data: Dados de entrada.
            
        Returns:
            Dicionário com resultados da predição.
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Use load_model() primeiro.")
        
        start_time = datetime.now()
        
        try:
            # 1. Validar entrada
            validated_data = self.validate_input(data)
            
            # 2. Pré-processar
            processed_data = self.preprocess_data(validated_data)
            
            # 3. Fazer predição
            predictions = self.model.predict(processed_data)
            
            # 4. Calcular intervalos de confiança (se possível)
            confidence_intervals = self._calculate_confidence_intervals(processed_data, predictions)
            
            # 5. Formatar resultados
            results = {
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else [predictions],
                'confidence_intervals': confidence_intervals,
                'input_data': validated_data.to_dict('records'),
                'model_info': self._get_model_summary(),
                'prediction_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'n_samples': len(validated_data)
                }
            }
            
            logger.info(f"Predição concluída para {len(validated_data)} amostras")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise
    
    def predict_single(self, **kwargs) -> float:
        """
        Predição simples para uma única amostra.
        
        Args:
            **kwargs: Valores das features como argumentos nomeados.
            
        Returns:
            Valor predito como float.
        """
        result = self.predict(kwargs)
        return result['predictions'][0]
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predição em lote para múltiplas amostras.
        
        Args:
            data: DataFrame com múltiplas amostras.
            
        Returns:
            DataFrame com predições adicionadas.
        """
        result = self.predict(data)
        
        # Criar DataFrame de saída
        output_df = data.copy()
        output_df['predicted_charges'] = result['predictions']
        
        # Adicionar intervalos de confiança se disponíveis
        if result['confidence_intervals']:
            output_df['confidence_lower'] = [ci[0] for ci in result['confidence_intervals']]
            output_df['confidence_upper'] = [ci[1] for ci in result['confidence_intervals']]
        
        return output_df
    
    def _calculate_confidence_intervals(self, X: np.ndarray, predictions: np.ndarray, 
                                      confidence: float = 0.95) -> Optional[List[Tuple[float, float]]]:
        """
        Calcula intervalos de confiança para as predições.
        
        Args:
            X: Features preprocessadas.
            predictions: Predições do modelo.
            confidence: Nível de confiança.
            
        Returns:
            Lista de tuplas (lower, upper) ou None.
        """
        try:
            # Para modelos que suportam predição com incerteza
            if hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
                # Random Forest - usar desvio padrão das árvores
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                std_predictions = np.std(tree_predictions, axis=0)
                
                # Aproximação do intervalo de confiança (assumindo distribuição normal)
                z_score = 1.96 if confidence == 0.95 else 2.576  # 95% ou 99%
                margin = z_score * std_predictions
                
                intervals = [(pred - margin[i], pred + margin[i]) 
                           for i, pred in enumerate(predictions)]
                
                return intervals
            
            else:
                # Para outros modelos, usar aproximação baseada na variância dos resíduos
                # Isso requer dados de treinamento, então retornamos None por enquanto
                return None
                
        except Exception as e:
            logger.warning(f"Não foi possível calcular intervalos de confiança: {e}")
            return None
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do modelo carregado.
        
        Returns:
            Dicionário com informações do modelo.
        """
        summary = {
            'model_type': type(self.model).__name__ if self.model else None,
            'features_count': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names
        }
        
        # Adicionar informações extras se disponíveis
        if self.model_info:
            summary.update({
                'training_score': self.model_info.get('score'),
                'training_timestamp': self.model_info.get('timestamp')
            })
        
        return summary
    
    def explain_prediction(self, data: Union[Dict, pd.DataFrame, pd.Series], 
                         top_n: int = 5) -> Dict[str, Any]:
        """
        Explica a predição mostrando contribuição das features.
        
        Args:
            data: Dados de entrada.
            top_n: Número de features mais importantes a mostrar.
            
        Returns:
            Dicionário com explicação da predição.
        """
        # Fazer predição normal
        result = self.predict(data)
        
        explanation = {
            'prediction': result['predictions'][0],
            'feature_contributions': None,
            'top_features': None
        }
        
        # Tentar obter importância das features do modelo
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Ordenar por importância
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            explanation['feature_contributions'] = feature_importance
            explanation['top_features'] = sorted_features[:top_n]
            
        return explanation


def load_best_model(models_dir: Path = MODELS_DIR) -> InsurancePredictor:
    """
    Carrega o melhor modelo treinado.
    
    Args:
        models_dir: Diretório dos modelos.
        
    Returns:
        Preditor configurado com o melhor modelo.
    """
    # Procurar pelo melhor modelo
    model_path = models_dir / "best_model.pkl"
    preprocessor_path = models_dir / "model_artifacts" / "preprocessor.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    
    predictor = InsurancePredictor()
    predictor.load_model(model_path)
    
    # Carregar preprocessor se existir
    if preprocessor_path.exists():
        predictor.load_preprocessor(preprocessor_path)
    else:
        logger.warning(f"Preprocessor não encontrado em: {preprocessor_path}")
    
    return predictor


def predict_insurance_premium(age: int, sex: str, bmi: float, children: int, 
                            smoker: str, region: str, 
                            model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Função conveniente para predição de prêmio de seguro.
    
    Args:
        age: Idade do segurado.
        sex: Sexo (male/female).
        bmi: Índice de massa corporal.
        children: Número de filhos.
        smoker: Fumante (yes/no).
        region: Região (northeast/northwest/southeast/southwest).
        model_path: Caminho do modelo (opcional).
        
    Returns:
        Dicionário com resultado da predição.
    """
    # Preparar dados
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Carregar modelo
    if model_path:
        predictor = InsurancePredictor(model_path)
    else:
        predictor = load_best_model()
    
    # Fazer predição
    result = predictor.predict(data)
    
    # Formatar resposta
    prediction = result['predictions'][0]
    
    return {
        'predicted_premium': round(prediction, 2),
        'input_data': data,
        'confidence_interval': result['confidence_intervals'][0] if result['confidence_intervals'] else None,
        'model_type': result['model_info']['model_type'],
        'processing_time_ms': result['prediction_metadata']['processing_time_ms']
    }


if __name__ == "__main__":
    # Exemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Dados de exemplo
        sample_data = {
            'age': 39,
            'sex': 'female',
            'bmi': 27.9,
            'children': 3,
            'smoker': 'no',
            'region': 'southeast'
        }
        
        print("🔮 Testando predição de seguro...")
        print(f"Dados de entrada: {sample_data}")
        
        # Fazer predição
        result = predict_insurance_premium(**sample_data)
        
        print(f"\n✅ Predição concluída!")
        print(f"Prêmio previsto: ${result['predicted_premium']:,.2f}")
        print(f"Tipo de modelo: {result['model_type']}")
        print(f"Tempo de processamento: {result['processing_time_ms']:.2f}ms")
        
        if result['confidence_interval']:
            lower, upper = result['confidence_interval']
            print(f"Intervalo de confiança: ${lower:,.2f} - ${upper:,.2f}")
        
    except Exception as e:
        print(f"❌ Erro: {e}") 