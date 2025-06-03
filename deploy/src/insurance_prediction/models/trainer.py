"""
Módulo de treinamento focado em Gradient Boosting para predição de seguros.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import joblib
import warnings

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    validation_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

from ..config.settings import Config
from ..utils.logging import get_logger

# Configurar logger
logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class GradientBoostingTrainer:
    """
    Trainer especializado em Gradient Boosting para predição de prêmios de seguro.
    """
    
    def __init__(self, use_mlflow: bool = False):
        """
        Inicializa o trainer.
        
        Args:
            use_mlflow: Se deve usar MLflow para tracking.
        """
        self.use_mlflow = use_mlflow
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.cv_results = {}
        
        # Configurar MLflow se solicitado
        if self.use_mlflow:
            self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Configura MLflow para tracking."""
        try:
            import mlflow
            import mlflow.sklearn
            
            mlflow.set_tracking_uri(Config.MLFLOW_CONFIG['tracking_uri'])
            mlflow.set_experiment(Config.MLFLOW_CONFIG['experiment_name'])
            logger.info("MLflow configurado com sucesso")
            
        except ImportError:
            logger.warning("MLflow não está instalado. Tracking desabilitado.")
            self.use_mlflow = False
        except Exception as e:
            logger.warning(f"Erro ao configurar MLflow: {e}")
            self.use_mlflow = False
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas completas de avaliação.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            
        Returns:
            Dicionário com métricas.
        """
        n_samples = len(y_true)
        n_features = getattr(self.model, 'n_features_in_', 1)
        
        # Métricas básicas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # R² ajustado
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        # Bias médio
        mbe = np.mean(y_pred - y_true)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'mape': mape,
            'mbe': mbe
        }
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Treina modelo baseline com configuração padrão.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            X_test: Features de teste.
            y_test: Target de teste.
            
        Returns:
            Resultados do treinamento baseline.
        """
        logger.info("Treinando modelo Gradient Boosting baseline...")
        
        start_time = datetime.now()
        
        # Criar modelo com configuração padrão
        model = GradientBoostingRegressor(**Config.get_model_config())
        
        # Treinar
        model.fit(X_train, y_train)
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test)
        
        # Validação cruzada
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=Config.CV_FOLDS, scoring='r2'
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Compilar resultados
        results = {
            'model': model,
            'model_type': 'Gradient Boosting Baseline',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'training_time': training_time,
            'feature_importance': self._get_feature_importance(model, X_train.columns)
        }
        
        self.model = model
        
        # Log resultados
        logger.info(f"✅ Modelo baseline treinado em {training_time:.2f}s")
        logger.info(f"   R² Test: {test_metrics['r2']:.4f}")
        logger.info(f"   MAE Test: {test_metrics['mae']:.2f}")
        logger.info(f"   RMSE Test: {test_metrics['rmse']:.2f}")
        logger.info(f"   CV R² Mean: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            method: str = 'random', n_iter: int = 50) -> Dict[str, Any]:
        """
        Otimização de hiperparâmetros.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            method: Método de busca ('grid' ou 'random').
            n_iter: Número de iterações para busca aleatória.
            
        Returns:
            Resultados da otimização.
        """
        logger.info(f"Iniciando otimização de hiperparâmetros ({method})...")
        
        # Criar modelo base
        base_model = GradientBoostingRegressor(random_state=Config.RANDOM_STATE)
        
        # Grid de parâmetros
        param_grid = Config.get_param_grid()
        
        start_time = datetime.now()
        
        if method == 'grid':
            # Grid Search completo
            search = GridSearchCV(
                base_model, param_grid,
                cv=Config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
        else:
            # Random Search (mais eficiente)
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter,
                cv=Config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                random_state=Config.RANDOM_STATE,
                verbose=1
            )
        
        # Executar busca
        search.fit(X_train, y_train)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Resultados
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_,
            'optimization_time': optimization_time
        }
        
        logger.info(f"✅ Otimização concluída em {optimization_time:.2f}s")
        logger.info(f"   Melhor score CV: {search.best_score_:.4f}")
        logger.info(f"   Melhores parâmetros:")
        for param, value in search.best_params_.items():
            logger.info(f"     {param}: {value}")
        
        return results
    
    def train_optimized_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Treina modelo otimizado completo.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            X_test: Features de teste.
            y_test: Target de teste.
            optimize_hyperparams: Se deve otimizar hiperparâmetros.
            
        Returns:
            Resultados completos do treinamento.
        """
        logger.info("🚀 Iniciando treinamento de modelo otimizado...")
        
        start_time = datetime.now()
        
        # 1. Treinar baseline
        baseline_results = self.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # 2. Otimização de hiperparâmetros se solicitada
        if optimize_hyperparams:
            tuning_results = self.hyperparameter_tuning(X_train, y_train, method='random', n_iter=50)
            
            # Retreinar com melhores parâmetros e avaliar
            best_model = tuning_results['best_model']
            best_model.fit(X_train, y_train)
            
            # Predições finais
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Métricas finais
            final_train_metrics = self.calculate_metrics(y_train, y_pred_train)
            final_test_metrics = self.calculate_metrics(y_test, y_pred_test)
            
            # Validação cruzada final
            final_cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=Config.CV_FOLDS, scoring='r2'
            )
            
            self.model = best_model
            
        else:
            # Usar modelo baseline
            tuning_results = None
            final_train_metrics = baseline_results['train_metrics']
            final_test_metrics = baseline_results['test_metrics']
            final_cv_scores = np.array([baseline_results['cv_r2_mean']])
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Compilar resultados finais
        final_results = {
            'model': self.model,
            'model_type': 'Gradient Boosting Optimized',
            'baseline_results': baseline_results,
            'tuning_results': tuning_results,
            'final_train_metrics': final_train_metrics,
            'final_test_metrics': final_test_metrics,
            'final_cv_r2_mean': final_cv_scores.mean(),
            'final_cv_r2_std': final_cv_scores.std(),
            'total_training_time': total_time,
            'feature_importance': self._get_feature_importance(self.model, X_train.columns),
            'learning_curve': self._get_learning_curve(X_train, y_train)
        }
        
        # Armazenar histórico
        self.training_history = final_results
        
        # Log resultados finais
        logger.info("🏆 TREINAMENTO OTIMIZADO CONCLUÍDO!")
        logger.info(f"   Tempo total: {total_time:.2f}s")
        logger.info(f"   R² Final: {final_test_metrics['r2']:.4f}")
        logger.info(f"   Adjusted R² Final: {final_test_metrics['adjusted_r2']:.4f}")
        logger.info(f"   MAE Final: {final_test_metrics['mae']:.2f}")
        logger.info(f"   RMSE Final: {final_test_metrics['rmse']:.2f}")
        logger.info(f"   MAPE Final: {final_test_metrics['mape']:.2f}%")
        logger.info(f"   CV R² Final: {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}")
        
        # Log MLflow se habilitado
        if self.use_mlflow:
            self._log_to_mlflow(final_results)
        
        return final_results
    
    def _get_feature_importance(self, model: GradientBoostingRegressor, 
                              feature_names: List[str]) -> pd.DataFrame:
        """
        Extrai importância das features.
        
        Args:
            model: Modelo treinado.
            feature_names: Nomes das features.
            
        Returns:
            DataFrame com importância das features.
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def _get_learning_curve(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Gera curva de aprendizado.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            
        Returns:
            Dados da curva de aprendizado.
        """
        if self.model is None:
            return {}
        
        try:
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, val_scores = learning_curve(
                self.model, X_train, y_train,
                cv=3,  # CV reduzido para performance
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2',
                n_jobs=-1
            )
            
            return {
                'train_sizes': train_sizes,
                'train_scores_mean': train_scores.mean(axis=1),
                'train_scores_std': train_scores.std(axis=1),
                'val_scores_mean': val_scores.mean(axis=1),
                'val_scores_std': val_scores.std(axis=1)
            }
            
        except Exception as e:
            logger.warning(f"Erro ao gerar curva de aprendizado: {e}")
            return {}
    
    def _log_to_mlflow(self, results: Dict[str, Any]) -> None:
        """
        Log resultados no MLflow.
        
        Args:
            results: Resultados do treinamento.
        """
        try:
            import mlflow
            import mlflow.sklearn
            
            with mlflow.start_run(run_name=f"GradientBoosting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parâmetros
                if self.best_params:
                    for param, value in self.best_params.items():
                        mlflow.log_param(param, value)
                
                # Log métricas de teste
                test_metrics = results['final_test_metrics']
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric}", value)
                
                # Log métricas de CV
                mlflow.log_metric("cv_r2_mean", results['final_cv_r2_mean'])
                mlflow.log_metric("cv_r2_std", results['final_cv_r2_std'])
                mlflow.log_metric("training_time", results['total_training_time'])
                
                # Log modelo
                mlflow.sklearn.log_model(
                    self.model, 
                    "gradient_boosting_model",
                    registered_model_name=Config.MLFLOW_CONFIG['model_name']
                )
                
                logger.info("Resultados logados no MLflow com sucesso")
                
        except Exception as e:
            logger.warning(f"Erro ao fazer log no MLflow: {e}")
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar. Se None, usa padrão.
            
        Returns:
            Caminho onde o modelo foi salvo.
        """
        if self.model is None:
            raise ValueError("Nenhum modelo foi treinado ainda!")
        
        if filepath is None:
            filepath = Config.MODELS_DIR / "gradient_boosting_model.pkl"
        
        # Criar diretório se necessário
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        joblib.dump(self.model, filepath)
        
        # Salvar metadados
        metadata = {
            'model_type': 'GradientBoostingRegressor',
            'best_params': self.best_params,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat(),
            'sklearn_version': getattr(self.model, '__module__', 'unknown')
        }
        
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Modelo salvo em: {filepath}")
        logger.info(f"✅ Metadados salvos em: {metadata_path}")
        
        return filepath
    
    def load_model(self, filepath: Path) -> GradientBoostingRegressor:
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo.
            
        Returns:
            Modelo carregado.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        self.model = joblib.load(filepath)
        
        # Tentar carregar metadados
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_params = metadata.get('best_params')
                self.training_history = metadata.get('training_history', {})
        
        logger.info(f"✅ Modelo carregado de: {filepath}")
        
        return self.model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do modelo treinado.
        
        Returns:
            Dicionário com resumo.
        """
        if self.model is None:
            return {"error": "Nenhum modelo treinado"}
        
        summary = {
            'model_type': type(self.model).__name__,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'subsample': self.model.subsample,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'n_features_in': getattr(self.model, 'n_features_in_', 0),
            'training_complete': bool(self.training_history)
        }
        
        if self.training_history:
            test_metrics = self.training_history.get('final_test_metrics', {})
            summary['performance'] = {
                'r2_score': test_metrics.get('r2', 0),
                'rmse': test_metrics.get('rmse', 0),
                'mae': test_metrics.get('mae', 0)
            }
        
        return summary


def train_gradient_boosting_model(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 optimize_hyperparams: bool = True,
                                 save_model: bool = True,
                                 use_mlflow: bool = False) -> Dict[str, Any]:
    """
    Função principal para treinamento de modelo Gradient Boosting.
    
    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        X_test: Features de teste.
        y_test: Target de teste.
        optimize_hyperparams: Se deve otimizar hiperparâmetros.
        save_model: Se deve salvar o modelo.
        use_mlflow: Se deve usar MLflow.
        
    Returns:
        Resultados completos do treinamento.
    """
    # Criar trainer
    trainer = GradientBoostingTrainer(use_mlflow=use_mlflow)
    
    # Treinar modelo
    results = trainer.train_optimized_model(
        X_train, y_train, X_test, y_test,
        optimize_hyperparams=optimize_hyperparams
    )
    
    # Salvar modelo se solicitado
    if save_model:
        model_path = trainer.save_model()
        results['model_path'] = model_path
    
    # Adicionar trainer aos resultados
    results['trainer'] = trainer
    
    return results


if __name__ == "__main__":
    # Teste do trainer
    from ..utils.logging import setup_logging
    from ..data.loader import load_insurance_data
    from ..data.preprocessor import preprocess_insurance_data
    
    setup_logging("INFO")
    
    try:
        logger.info("🧪 Testando GradientBoostingTrainer...")
        
        # Carregar e preprocessar dados
        data, _ = load_insurance_data()
        processed_data = preprocess_insurance_data(data)
        
        # Treinar modelo
        results = train_gradient_boosting_model(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test'],
            optimize_hyperparams=False,  # Teste rápido
            save_model=False
        )
        
        logger.info("✅ Teste do trainer concluído!")
        logger.info(f"R² Final: {results['final_test_metrics']['r2']:.4f}")
        logger.info(f"RMSE Final: {results['final_test_metrics']['rmse']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}") 