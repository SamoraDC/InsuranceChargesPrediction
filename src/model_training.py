#!/usr/bin/env python3
"""
Módulo para treinamento de modelos de machine learning para predição de prêmios de seguro.

Este módulo implementa múltiplos algoritmos de regressão com validação cruzada,
otimização de hiperparâmetros e avaliação abrangente usando várias métricas.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path se necessário
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import mlflow
import mlflow.sklearn

# Machine Learning
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost não está instalado. Instale com: pip install xgboost")

from config import Config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_bias_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Mean Bias Deviation (MBE) - viés médio das predições.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        
    Returns:
        MBE value
    """
    return np.mean(y_pred - y_true)

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calcula R² ajustado.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        n_features: Número de features
        
    Returns:
        Adjusted R² value
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  n_features: int) -> Dict[str, float]:
    """
    Calcula todas as métricas solicitadas.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        n_features: Número de features
        
    Returns:
        Dictionary com todas as métricas
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Converter para %
        'R²': r2_score(y_true, y_pred),
        'Adjusted_R²': adjusted_r2_score(y_true, y_pred, n_features),
        'MBE': mean_bias_deviation(y_true, y_pred)
    }

class InsuranceModelTrainer:
    """Classe para treinamento de modelos de predição de prêmios de seguro."""
    
    def __init__(self, use_mlflow: bool = False):
        """
        Inicializa o trainer.
        
        Args:
            use_mlflow: Se deve usar MLflow para tracking
        """
        self.use_mlflow = use_mlflow
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        
        # Definir modelos
        self._setup_models()
        
        # Configurar MLflow se solicitado
        if self.use_mlflow:
            self._setup_mlflow()
    
    def _setup_models(self):
        """Define os modelos a serem treinados."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'ElasticNet Regression': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        # Modelos ensemble
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0, random_state=42))
        ]
        
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', XGBRegressor(n_estimators=50, random_state=42, verbosity=0)))
        
        self.models['Voting Regressor'] = VotingRegressor(
            estimators=base_models
        )
        
        self.models['Bagging Regressor'] = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42),
            n_estimators=50,
            random_state=42
        )
    
    def _setup_mlflow(self):
        """Configura MLflow para tracking."""
        try:
            mlflow.set_tracking_uri(Config.MLFLOW_URI)
            mlflow.set_experiment("Insurance_Premium_Prediction")
            logger.info("MLflow configurado com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao configurar MLflow: {e}")
            self.use_mlflow = False
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define grids de hiperparâmetros para otimização."""
        grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'ElasticNet Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'Decision Tree': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            },
            'Extra Trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        return grids
    
    def train_model(self, model_name: str, model: Any, X_train: np.ndarray, 
                   y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   optimize_hyperparams: bool = True, cv_folds: int = 5) -> Dict:
        """
        Treina um modelo individual com validação cruzada e otimização opcional.
        
        Args:
            model_name: Nome do modelo
            model: Instância do modelo
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            optimize_hyperparams: Se deve otimizar hiperparâmetros
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Dictionary com resultados do modelo
        """
        logger.info(f"Treinando {model_name}...")
        
        start_time = datetime.now()
        
        try:
            # Otimização de hiperparâmetros se solicitada
            if optimize_hyperparams:
                grids = self.get_hyperparameter_grids()
                if model_name in grids:
                    logger.info(f"Otimizando hiperparâmetros para {model_name}...")
                    
                    # Usar RandomizedSearchCV para modelos complexos
                    if model_name in ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'XGBoost']:
                        search = RandomizedSearchCV(
                            model, grids[model_name], 
                            n_iter=20, cv=cv_folds, 
                            scoring='r2', random_state=42, 
                            n_jobs=-1, verbose=0
                        )
                    else:
                        search = GridSearchCV(
                            model, grids[model_name], 
                            cv=cv_folds, scoring='r2', 
                            n_jobs=-1, verbose=0
                        )
                    
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    logger.info(f"Melhores parâmetros: {search.best_params_}")
            
            # Treinar modelo final
            model.fit(X_train, y_train)
            
            # Predições
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            # Calcular métricas completas
            n_features = X_train.shape[1]
            train_metrics = calculate_comprehensive_metrics(y_train, y_pred_train, n_features)
            test_metrics = calculate_comprehensive_metrics(y_test, y_pred_test, n_features)
            
            # Métricas de validação cruzada
            cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_absolute_error')
            cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
            
            # Compilar resultados
            results = {
                'model': model,
                'model_name': model_name,
                'training_time': (datetime.now() - start_time).total_seconds(),
                
                # Métricas de treino
                'train_metrics': train_metrics,
                
                # Métricas de teste
                'test_metrics': test_metrics,
                
                # Validação cruzada
                'cv_r2_scores': cv_scores,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_mae_mean': cv_mae_scores.mean(),
                'cv_mae_std': cv_mae_scores.std(),
                'cv_rmse_mean': np.sqrt(cv_mse_scores.mean()),
                'cv_rmse_std': np.sqrt(cv_mse_scores.std()),
                
                # Score principal para comparação (R² do teste)
                'main_score': test_metrics['R²']
            }
            
            # Logging MLflow
            if self.use_mlflow:
                self._log_to_mlflow(model_name, model, results)
            
            # Atualizar melhor modelo
            if results['main_score'] > self.best_score:
                self.best_score = results['main_score']
                self.best_model = model
                self.best_model_name = model_name
                logger.info(f"🏆 Novo melhor modelo: {model_name} (R² = {self.best_score:.4f})")
            
            # Log das métricas principais
            logger.info(f"✅ {model_name} - Resultados:")
            logger.info(f"   R² Test: {test_metrics['R²']:.4f}")
            logger.info(f"   MAE Test: {test_metrics['MAE']:.2f}")
            logger.info(f"   RMSE Test: {test_metrics['RMSE']:.2f}")
            logger.info(f"   MAPE Test: {test_metrics['MAPE']:.2f}%")
            logger.info(f"   CV R² Mean: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro ao treinar {model_name}: {e}")
            return {
                'model': None,
                'model_name': model_name,
                'error': str(e),
                'main_score': -np.inf
            }
    
    def _log_to_mlflow(self, model_name: str, model: Any, results: Dict):
        """Log resultados no MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parâmetros
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    for param, value in params.items():
                        mlflow.log_param(param, value)
                
                # Log métricas de teste
                test_metrics = results['test_metrics']
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric.lower()}", value)
                
                # Log métricas de CV
                mlflow.log_metric("cv_r2_mean", results['cv_r2_mean'])
                mlflow.log_metric("cv_r2_std", results['cv_r2_std'])
                mlflow.log_metric("cv_mae_mean", results['cv_mae_mean'])
                mlflow.log_metric("training_time", results['training_time'])
                
                # Log modelo
                mlflow.sklearn.log_model(model, f"model_{model_name.lower().replace(' ', '_')}")
                
        except Exception as e:
            logger.warning(f"Erro ao fazer log no MLflow para {model_name}: {e}")
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray,
                        optimize_hyperparams: bool = True, cv_folds: int = 5) -> Dict:
        """
        Treina todos os modelos e retorna resultados comparativos.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            optimize_hyperparams: Se deve otimizar hiperparâmetros
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Dictionary com resultados de todos os modelos
        """
        logger.info("🚀 Iniciando treinamento de todos os modelos...")
        logger.info(f"Dataset: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Otimização de hiperparâmetros: {'Ativada' if optimize_hyperparams else 'Desativada'}")
        
        all_results = {}
        
        for model_name, model in self.models.items():
            try:
                result = self.train_model(
                    model_name, model, X_train, y_train, 
                    X_test, y_test, optimize_hyperparams, cv_folds
                )
                all_results[model_name] = result
                self.results[model_name] = result
                
            except Exception as e:
                logger.error(f"❌ Falha ao treinar {model_name}: {e}")
                all_results[model_name] = {
                    'model': None,
                    'error': str(e),
                    'main_score': -np.inf
                }
        
        # Resumo final
        logger.info("\n📊 RESUMO FINAL DOS MODELOS:")
        logger.info("=" * 80)
        
        # Ordenar por performance
        sorted_results = sorted(
            [(name, result) for name, result in all_results.items() 
             if result.get('main_score', -np.inf) > -np.inf],
            key=lambda x: x[1]['main_score'], 
            reverse=True
        )
        
        for i, (name, result) in enumerate(sorted_results):
            metrics = result.get('test_metrics', {})
            logger.info(f"{i+1:2d}. {name:<25} | R²: {metrics.get('R²', 0):.4f} | "
                       f"MAE: {metrics.get('MAE', 0):7.2f} | "
                       f"RMSE: {metrics.get('RMSE', 0):7.2f} | "
                       f"MAPE: {metrics.get('MAPE', 0):5.2f}%")
        
        logger.info("=" * 80)
        
        if self.best_model_name:
            best_metrics = all_results[self.best_model_name]['test_metrics']
            logger.info(f"🏆 MELHOR MODELO: {self.best_model_name}")
            logger.info(f"   R²: {best_metrics['R²']:.4f} | Adj R²: {best_metrics['Adjusted_R²']:.4f}")
            logger.info(f"   MAE: {best_metrics['MAE']:.2f} | MSE: {best_metrics['MSE']:.2f}")
            logger.info(f"   RMSE: {best_metrics['RMSE']:.2f} | MAPE: {best_metrics['MAPE']:.2f}%")
            logger.info(f"   MBE: {best_metrics['MBE']:.2f}")
        
        return all_results
    
    def save_best_model(self, save_path: str = None) -> str:
        """
        Salva o melhor modelo.
        
        Args:
            save_path: Caminho para salvar o modelo
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda!")
        
        if save_path is None:
            save_path = str(Config.MODELS_DIR / "best_model.pkl")
        
        # Criar diretório se não existir
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        joblib.dump(self.best_model, save_path)
        
        logger.info(f"✅ Melhor modelo ({self.best_model_name}) salvo em: {save_path}")
        
        # Salvar metadados
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'performance': self.results[self.best_model_name]['test_metrics'],
            'training_date': datetime.now().isoformat(),
            'cv_performance': {
                'r2_mean': self.results[self.best_model_name]['cv_r2_mean'],
                'r2_std': self.results[self.best_model_name]['cv_r2_std'],
                'mae_mean': self.results[self.best_model_name]['cv_mae_mean']
            }
        }
        
        metadata_path = str(Path(save_path).parent / "best_model_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadados salvos em: {metadata_path}")
        
        return save_path
    
    def get_model_comparison_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com comparação de todos os modelos.
        
        Returns:
            DataFrame com métricas de todos os modelos
        """
        if not self.results:
            raise ValueError("Nenhum modelo foi treinado ainda!")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            if result.get('test_metrics'):
                metrics = result['test_metrics']
                row = {
                    'Model': model_name,
                    'R²': metrics['R²'],
                    'Adjusted_R²': metrics['Adjusted_R²'],
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE (%)': metrics['MAPE'],
                    'MBE': metrics['MBE'],
                    'CV_R²_Mean': result['cv_r2_mean'],
                    'CV_R²_Std': result['cv_r2_std'],
                    'CV_MAE_Mean': result['cv_mae_mean'],
                    'Training_Time (s)': result['training_time']
                }
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por R² (descendente)
        df = df.sort_values('R²', ascending=False).reset_index(drop=True)
        
        return df

def train_insurance_models(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          save_model: bool = True, use_mlflow: bool = False,
                          optimize_hyperparams: bool = True) -> Dict:
    """
    Função principal para treinamento de modelos de seguro.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_test: Features de teste
        y_test: Target de teste
        save_model: Se deve salvar o melhor modelo
        use_mlflow: Se deve usar MLflow para tracking
        optimize_hyperparams: Se deve otimizar hiperparâmetros
        
    Returns:
        Dictionary com resultados do treinamento
    """
    # Criar trainer
    trainer = InsuranceModelTrainer(use_mlflow=use_mlflow)
    
    # Treinar todos os modelos
    results = trainer.train_all_models(
        X_train, y_train, X_test, y_test,
        optimize_hyperparams=optimize_hyperparams
    )
    
    # Salvar melhor modelo se solicitado
    if save_model and trainer.best_model is not None:
        model_path = trainer.save_best_model()
        results['saved_model_path'] = model_path
    
    # Criar DataFrame de comparação
    comparison_df = trainer.get_model_comparison_dataframe()
    results['comparison_dataframe'] = comparison_df
    
    return {
        'results': results,
        'best_model': trainer.best_model,
        'best_model_name': trainer.best_model_name,
        'trainer': trainer,
        'comparison_df': comparison_df
    }

# ... existing code ... 