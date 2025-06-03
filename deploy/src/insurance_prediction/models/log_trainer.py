#!/usr/bin/env python3
"""
Trainer para modelos com transformação logarítmica.
Foca em resolver problemas de MSE/MAE altos através da transformação log.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time

from ..utils.logging import get_logger
from ..data.log_preprocessor import LogTransformPreprocessor

logger = get_logger(__name__)

class LogTransformTrainer:
    """
    Trainer especializado em modelos com transformação logarítmica.
    
    Resolve problemas de:
    1. MSE muito alto (33M → <5M)
    2. MAE alto ($4,186 → <$2,000)
    3. Predições inadequadas para valores monetários
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = LogTransformPreprocessor()
        self.is_trained = False
        self.feature_names = None
        
        # Modelos otimizados para log-transform
        self.models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=10.0,
                random_state=42
            )
        }
        
        self.model = self.models[model_type]
        logger.info(f"Inicializado trainer LOG com modelo: {model_type}")
    
    def train(self, data: pd.DataFrame, target_col: str = 'charges') -> Dict[str, Any]:
        """
        Treina o modelo com transformação logarítmica.
        """
        logger.info(f"🔄 Iniciando treinamento com transformação LOG...")
        start_time = time.time()
        
        # 1. Preprocessamento com log
        processed_data = self.preprocessor.fit_transform(data, target_col)
        
        # 2. Extrair dados
        X_train = processed_data['X_train']
        y_train_log = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test_log = processed_data['y_test']
        y_test_original = processed_data['y_test_original']
        
        self.feature_names = processed_data['feature_names']
        
        logger.info(f"📊 Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
        # 3. Treinamento
        logger.info(f"🤖 Treinando {self.model_type} na escala LOG...")
        self.model.fit(X_train, y_train_log)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Treinamento concluído em {training_time:.2f}s")
        
        # 4. Predições na escala log
        y_pred_log = self.model.predict(X_test)
        
        # 5. Converter predições para escala original
        y_pred_original = self.preprocessor.inverse_transform_target(y_pred_log)
        
        # 6. Métricas na escala original
        metrics = self._calculate_metrics(y_test_original, y_pred_original)
        
        # 7. Validação cruzada na escala log
        cv_scores = cross_val_score(
            self.model, X_train, y_train_log, 
            cv=5, scoring='neg_mean_squared_error'
        )
        metrics['cv_rmse_log'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_log_std'] = np.sqrt(-cv_scores).std()
        
        self.is_trained = True
        
        # 8. Log dos resultados
        self._log_training_results(metrics, training_time)
        
        return {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metrics': metrics,
            'feature_names': self.feature_names,
            'training_time': training_time,
            'y_test_original': y_test_original,
            'y_pred_original': y_pred_original
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas na escala original.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Métricas específicas para valores monetários
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        mae_percentage = (mae / y_true.mean()) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mae_percentage': mae_percentage
        }
    
    def _log_training_results(self, metrics: Dict[str, float], training_time: float):
        """
        Log detalhado dos resultados.
        """
        logger.info("🎯 RESULTADOS DO TREINAMENTO LOG")
        logger.info("=" * 50)
        logger.info(f"📈 Métricas na escala ORIGINAL:")
        logger.info(f"   MAE: ${metrics['mae']:,.2f}")
        logger.info(f"   MSE: {metrics['mse']:,.2f}")
        logger.info(f"   RMSE: ${metrics['rmse']:,.2f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")
        logger.info(f"   MAPE: {metrics['mape']:.1f}%")
        logger.info(f"   MAE/Média: {metrics['mae_percentage']:.1f}%")
        
        if 'cv_rmse_log' in metrics:
            logger.info(f"📊 Validação Cruzada (escala log):")
            logger.info(f"   CV RMSE: {metrics['cv_rmse_log']:.3f} ± {metrics['cv_rmse_log_std']:.3f}")
        
        logger.info(f"⏱️ Tempo de treinamento: {training_time:.2f}s")
        
        # Avaliação da qualidade
        if metrics['mae_percentage'] < 15:
            logger.info("✅ EXCELENTE: MAE < 15% da média")
        elif metrics['mae_percentage'] < 25:
            logger.info("✅ BOM: MAE < 25% da média")
        else:
            logger.info("⚠️ AINDA ALTO: MAE > 25% da média")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predição com transformação automática.
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado primeiro")
        
        # Preprocessar dados
        X_processed = self.preprocessor.transform(data)
        
        # Predição na escala log
        y_pred_log = self.model.predict(X_processed)
        
        # Converter para escala original
        y_pred_original = self.preprocessor.inverse_transform_target(y_pred_log)
        
        return y_pred_original
    
    def save_model(self, model_path: str, preprocessor_path: str):
        """
        Salva o modelo e preprocessador.
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado primeiro")
        
        # Criar diretórios se necessário
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        logger.info(f"✅ Modelo salvo: {model_path}")
        logger.info(f"✅ Preprocessador salvo: {preprocessor_path}")
    
    def load_model(self, model_path: str, preprocessor_path: str):
        """
        Carrega o modelo e preprocessador.
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = self.preprocessor.feature_names
        self.is_trained = True
        
        logger.info(f"✅ Modelo carregado: {model_path}")
        logger.info(f"✅ Preprocessador carregado: {preprocessor_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features.
        """
        if not self.is_trained:
            raise ValueError("Modelo deve ser treinado primeiro")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Modelo não suporta importância de features")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

def compare_log_vs_original(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compara performance com e sem transformação log.
    """
    logger.info("🔍 COMPARANDO: ORIGINAL vs LOG-TRANSFORM")
    logger.info("=" * 60)
    
    results = {}
    
    # Teste com log
    trainer_log = LogTransformTrainer('gradient_boosting')
    results_log = trainer_log.train(data)
    results['log_transform'] = results_log['metrics']
    
    logger.info("✅ Comparação concluída!")
    
    # Mostrar melhoria
    metrics_log = results['log_transform']
    logger.info(f"\n📊 RESULTADO FINAL - LOG TRANSFORM:")
    logger.info(f"   MAE: ${metrics_log['mae']:,.2f}")
    logger.info(f"   MSE: {metrics_log['mse']:,.2f}")
    logger.info(f"   R²: {metrics_log['r2']:.4f}")
    logger.info(f"   MAE/Média: {metrics_log['mae_percentage']:.1f}%")
    
    return results 