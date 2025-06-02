"""
Módulo de predição para modelos de seguros.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import joblib
from datetime import datetime
import warnings

from ..config.settings import Config
from ..utils.logging import get_logger

# Configurar logger
logger = get_logger(__name__)
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
        self.model_metadata = None
        
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
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"✅ Modelo carregado: {model_path}")
            
            # Tentar carregar metadados
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"✅ Metadados carregados: {metadata_path}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: Path) -> None:
        """
        Carrega preprocessor treinado.
        
        Args:
            preprocessor_path: Caminho para o preprocessor.
        """
        try:
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor não encontrado: {preprocessor_path}")
            
            preprocessor_data = joblib.load(preprocessor_path)
            
            # Recriar o preprocessor
            from ..data.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            
            # Carregar componentes
            self.preprocessor.label_encoders = preprocessor_data.get('label_encoders', {})
            self.preprocessor.feature_selector = preprocessor_data.get('feature_selector')
            self.preprocessor.selected_features = preprocessor_data.get('selected_features')
            self.preprocessor.preprocessing_stats = preprocessor_data.get('preprocessing_stats', {})
            
            logger.info(f"✅ Preprocessor carregado: {preprocessor_path}")
            if self.preprocessor.selected_features:
                logger.info(f"   Features: {len(self.preprocessor.selected_features)}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar preprocessor: {e}")
            raise
    
    def validate_input(self, data: Union[Dict, pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Valida e normaliza dados de entrada.
        
        Args:
            data: Dados de entrada.
            
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
        missing_cols = set(Config.FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltando: {missing_cols}")
        
        # Selecionar apenas colunas de features
        df = df[Config.FEATURE_COLUMNS].copy()
        
        # Validar valores categóricos
        for col in Config.CATEGORICAL_COLUMNS:
            if col in df.columns:
                invalid_values = set(df[col]) - set(Config.CATEGORICAL_VALUES[col])
                if invalid_values:
                    raise ValueError(f"Valores inválidos em {col}: {invalid_values}")
        
        # Validar valores numéricos
        for col in Config.NUMERICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar ranges
                if col in Config.NUMERICAL_RANGES:
                    min_val = Config.NUMERICAL_RANGES[col]['min']
                    max_val = Config.NUMERICAL_RANGES[col]['max']
                    
                    out_of_range = (df[col] < min_val) | (df[col] > max_val)
                    if out_of_range.any():
                        logger.warning(f"Valores fora do range em {col}: {df[col][out_of_range].tolist()}")
        
        # Verificar valores nulos
        if df.isnull().any().any():
            logger.warning("Valores nulos encontrados - serão tratados pelo preprocessor")
        
        return df
    
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
            
            # 2. Pré-processar se preprocessor disponível
            if self.preprocessor:
                processed_data = self.preprocessor.transform(validated_data)
            else:
                # Encoding básico se não houver preprocessor
                processed_data = self._basic_encoding(validated_data)
            
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
                    'n_samples': len(validated_data),
                    'preprocessor_used': self.preprocessor is not None
                }
            }
            
            logger.info(f"✅ Predição concluída para {len(validated_data)} amostras")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
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
    
    def _basic_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding básico quando não há preprocessor.
        
        Args:
            data: DataFrame de entrada.
            
        Returns:
            DataFrame codificado.
        """
        encoded_data = data.copy()
        
        # Label encoding básico para categóricas
        for col in Config.CATEGORICAL_COLUMNS:
            if col in encoded_data.columns:
                if col == 'sex':
                    encoded_data[col] = (encoded_data[col] == 'male').astype(int)
                elif col == 'smoker':
                    encoded_data[col] = (encoded_data[col] == 'yes').astype(int)
                elif col == 'region':
                    region_map = {
                        'northeast': 0, 'northwest': 1, 
                        'southeast': 2, 'southwest': 3
                    }
                    encoded_data[col] = encoded_data[col].map(region_map).fillna(0)
        
        return encoded_data
    
    def _calculate_confidence_intervals(self, X: np.ndarray, predictions: np.ndarray, 
                                      confidence: float = 0.95) -> Optional[List]:
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
            # Para Gradient Boosting, usar estimativa da variância
            if hasattr(self.model, 'estimators_'):
                # Estimar incerteza baseada na variação entre estimadores
                
                # Aproximação simples: usar desvio padrão baseado no erro típico
                # Para produção, seria melhor usar métodos mais sofisticados
                std_estimate = np.std(predictions) * 0.1  # 10% como estimativa conservadora
                
                # Intervalo de confiança
                z_score = 1.96 if confidence == 0.95 else 2.576
                margin = z_score * std_estimate
                
                intervals = [(pred - margin, pred + margin) for pred in predictions]
                return intervals
            
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
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None
        }
        
        # Adicionar informações do modelo
        if self.model and hasattr(self.model, 'n_features_in_'):
            summary['n_features_in'] = self.model.n_features_in_
        
        # Adicionar metadados se disponíveis
        if self.model_metadata:
            summary.update({
                'training_date': self.model_metadata.get('saved_at'),
                'model_version': self.model_metadata.get('model_type')
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
        if hasattr(self.model, 'feature_importances_'):
            if self.preprocessor and self.preprocessor.selected_features:
                feature_names = self.preprocessor.selected_features
            else:
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            # Ordenar por importância
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            explanation['feature_contributions'] = feature_importance
            explanation['top_features'] = sorted_features[:top_n]
        
        return explanation


def load_production_model(models_dir: Path = None) -> InsurancePredictor:
    """
    Carrega o modelo de produção.
    
    Args:
        models_dir: Diretório dos modelos.
        
    Returns:
        Preditor configurado.
    """
    if models_dir is None:
        models_dir = Config.MODELS_DIR
    
    # Procurar pelo modelo principal
    model_path = models_dir / "gradient_boosting_model.pkl"
    preprocessor_path = Config.MODEL_ARTIFACTS_DIR / "preprocessor.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de produção não encontrado em: {model_path}")
    
    predictor = InsurancePredictor()
    predictor.load_model(model_path)
    
    # Carregar preprocessor se existir
    if preprocessor_path.exists():
        predictor.load_preprocessor(preprocessor_path)
    else:
        logger.warning(f"Preprocessor não encontrado em: {preprocessor_path}")
        logger.warning("Usando encoding básico")
    
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
        predictor = load_production_model()
    
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
    # Teste do preditor
    from ..utils.logging import setup_logging
    
    setup_logging("INFO")
    
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
        
        logger.info("🔮 Testando predição de seguro...")
        logger.info(f"Dados de entrada: {sample_data}")
        
        # Fazer predição
        result = predict_insurance_premium(**sample_data)
        
        logger.info(f"✅ Predição concluída!")
        logger.info(f"Prêmio previsto: ${result['predicted_premium']:,.2f}")
        logger.info(f"Tipo de modelo: {result['model_type']}")
        logger.info(f"Tempo de processamento: {result['processing_time_ms']:.2f}ms")
        
        if result['confidence_interval']:
            lower, upper = result['confidence_interval']
            logger.info(f"Intervalo de confiança: ${lower:,.2f} - ${upper:,.2f}")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}") 