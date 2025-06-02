"""
Módulo de treinamento de modelos para predição de prêmios de seguro.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import joblib
from datetime import datetime

# Tentar importar MLflow (opcional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow não está disponível. Tracking de experimentos será desabilitado.")

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    cross_validate, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

# Configurações do projeto
from .config import (
    RANDOM_STATE,
    CV_FOLDS,
    MODEL_CONFIGS,
    EVALUATION_METRICS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    MODEL_ARTIFACTS_DIR
)

# Configurar logging
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ModelTrainer:
    """
    Classe para treinamento e avaliação de modelos de regressão.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE, cv_folds: int = CV_FOLDS):
        """
        Inicializa o trainer de modelos.
        
        Args:
            random_state: Semente para reprodutibilidade.
            cv_folds: Número de folds para cross-validation.
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.trained_models = {}
        self.training_results = {}
        self.best_model = None
        self.best_score = -np.inf
        
        # Configurar MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configura MLflow para tracking de experimentos."""
        if not MLFLOW_AVAILABLE:
            logger.info("MLflow não disponível - tracking desabilitado")
            return
            
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Verificar se o experimento existe, senão criar
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                logger.info(f"Experimento MLflow criado: {MLFLOW_EXPERIMENT_NAME}")
            
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            logger.info(f"MLflow configurado: {MLFLOW_TRACKING_URI}")
            
        except Exception as e:
            logger.warning(f"Erro ao configurar MLflow: {e}")
            logger.warning("Continuando sem tracking MLflow")
    
    def _get_model_instance(self, model_name: str) -> Any:
        """
        Cria instância do modelo baseado no nome.
        
        Args:
            model_name: Nome do modelo.
            
        Returns:
            Instância do modelo.
        """
        model_map = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'extra_trees': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1)
        }
        
        if model_name not in model_map:
            raise ValueError(f"Modelo não suportado: {model_name}")
            
        return model_map[model_name]
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str, hyperparameter_tuning: bool = True,
                          search_type: str = 'grid') -> Dict[str, Any]:
        """
        Treina um único modelo.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            model_name: Nome do modelo a treinar.
            hyperparameter_tuning: Se deve fazer tuning de hiperparâmetros.
            search_type: Tipo de busca ('grid', 'random').
            
        Returns:
            Dicionário com resultados do treinamento.
        """
        logger.info(f"Iniciando treinamento: {model_name}")
        
        # Obter configuração do modelo
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Configuração não encontrada para modelo: {model_name}")
        
        model_config = MODEL_CONFIGS[model_name]
        base_model = self._get_model_instance(model_name)
        
        # Configurar cross-validation
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        best_model = None
        best_params = None
        cv_scores = {}
        
        # Configurar context manager para MLflow se disponível
        if MLFLOW_AVAILABLE:
            mlflow_context = mlflow.start_run(run_name=f"{model_config['name']}_{datetime.now().strftime('%H%M%S')}")
        else:
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context:
            
            try:
                if hyperparameter_tuning and model_config['params']:
                    # Hyperparameter tuning
                    logger.info(f"Realizando hyperparameter tuning para {model_name}")
                    
                    if search_type == 'grid':
                        search = GridSearchCV(
                            base_model,
                            param_grid=model_config['params'],
                            cv=cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            verbose=0
                        )
                    else:  # random search
                        search = RandomizedSearchCV(
                            base_model,
                            param_distributions=model_config['params'],
                            n_iter=20,
                            cv=cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            random_state=self.random_state,
                            verbose=0
                        )
                    
                    # Ajustar modelo
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    
                    logger.info(f"Melhores parâmetros: {best_params}")
                    
                else:
                    # Treinar com parâmetros padrão
                    best_model = base_model
                    best_model.fit(X_train, y_train)
                    best_params = best_model.get_params()
                
                # Cross-validation com melhor modelo
                scoring_metrics = {
                    'neg_mean_absolute_error': 'neg_mean_absolute_error',
                    'neg_mean_squared_error': 'neg_mean_squared_error',
                    'r2': 'r2'
                }
                
                cv_results = cross_validate(
                    best_model, X_train, y_train,
                    cv=cv, scoring=scoring_metrics,
                    return_train_score=True, n_jobs=-1
                )
                
                # Calcular estatísticas
                for metric, values in cv_results.items():
                    if metric.startswith('test_'):
                        metric_name = metric.replace('test_', '')
                        cv_scores[f'{metric_name}_mean'] = np.mean(values)
                        cv_scores[f'{metric_name}_std'] = np.std(values)
                
                # Log no MLflow se disponível
                if MLFLOW_AVAILABLE:
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_params(best_params)
                    
                    for metric, value in cv_scores.items():
                        mlflow.log_metric(metric, value)
                    
                    # Salvar modelo no MLflow
                    mlflow.sklearn.log_model(
                        best_model, 
                        f"model_{model_name}",
                        registered_model_name=f"insurance_predictor_{model_name}"
                    )
                
                logger.info(f"Modelo {model_name} treinado com sucesso")
                logger.info(f"R² médio: {cv_scores.get('r2_mean', 'N/A'):.4f} ± {cv_scores.get('r2_std', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao treinar modelo {model_name}: {e}")
                if MLFLOW_AVAILABLE:
                    mlflow.log_param("error", str(e))
                raise
        
        # Salvar resultado
        result = {
            'model_name': model_name,
            'model_display_name': model_config['name'],
            'model': best_model,
            'best_params': best_params,
            'cv_scores': cv_scores,
            'training_time': datetime.now().isoformat()
        }
        
        return result
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        models_to_train: Optional[List[str]] = None,
                        hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Treina todos os modelos configurados.
        
        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            models_to_train: Lista de modelos para treinar. Se None, treina todos.
            hyperparameter_tuning: Se deve fazer tuning de hiperparâmetros.
            
        Returns:
            Dicionário com resultados de todos os modelos.
        """
        if models_to_train is None:
            models_to_train = list(MODEL_CONFIGS.keys())
        
        logger.info(f"Iniciando treinamento de {len(models_to_train)} modelos")
        
        self.training_results = {}
        
        for model_name in models_to_train:
            try:
                result = self.train_single_model(
                    X_train, y_train, model_name, hyperparameter_tuning
                )
                self.training_results[model_name] = result
                self.trained_models[model_name] = result['model']
                
                # Verificar se é o melhor modelo
                r2_score = result['cv_scores'].get('r2_mean', -np.inf)
                if r2_score > self.best_score:
                    self.best_score = r2_score
                    self.best_model = result['model']
                    logger.info(f"Novo melhor modelo: {model_name} (R² = {r2_score:.4f})")
                
            except Exception as e:
                logger.error(f"Falha ao treinar {model_name}: {e}")
                continue
        
        logger.info(f"Treinamento concluído. Melhor modelo: R² = {self.best_score:.4f}")
        
        return self.training_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compara performance de todos os modelos treinados.
        
        Returns:
            DataFrame com comparação dos modelos.
        """
        if not self.training_results:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        comparison_data = []
        
        for model_name, result in self.training_results.items():
            cv_scores = result['cv_scores']
            
            row = {
                'Model': result['model_display_name'],
                'R²_Mean': cv_scores.get('r2_mean', np.nan),
                'R²_Std': cv_scores.get('r2_std', np.nan),
                'MAE_Mean': -cv_scores.get('neg_mean_absolute_error_mean', np.nan),
                'MAE_Std': cv_scores.get('neg_mean_absolute_error_std', np.nan),
                'MSE_Mean': -cv_scores.get('neg_mean_squared_error_mean', np.nan),
                'MSE_Std': cv_scores.get('neg_mean_squared_error_std', np.nan),
                'RMSE_Mean': np.sqrt(-cv_scores.get('neg_mean_squared_error_mean', np.nan)),
                'Model_Key': model_name
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordenar por R² (melhor primeiro)
        comparison_df = comparison_df.sort_values('R²_Mean', ascending=False)
        
        return comparison_df
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Avalia um modelo no conjunto de teste.
        
        Args:
            model: Modelo treinado.
            X_test: Features de teste.
            y_test: Target de teste.
            
        Returns:
            Dicionário com métricas de avaliação.
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Avalia todos os modelos treinados no conjunto de teste.
        
        Args:
            X_test: Features de teste.
            y_test: Target de teste.
            
        Returns:
            DataFrame com métricas de todos os modelos.
        """
        if not self.trained_models:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        evaluation_data = []
        
        for model_name, model in self.trained_models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test)
                
                row = {
                    'Model': MODEL_CONFIGS[model_name]['name'],
                    'MAE': metrics['mae'],
                    'MSE': metrics['mse'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2'],
                    'MAPE': metrics['mape'],
                    'Model_Key': model_name
                }
                
                evaluation_data.append(row)
                
            except Exception as e:
                logger.error(f"Erro ao avaliar modelo {model_name}: {e}")
                continue
        
        evaluation_df = pd.DataFrame(evaluation_data)
        evaluation_df = evaluation_df.sort_values('R²', ascending=False)
        
        return evaluation_df
    
    def save_best_model(self, filepath: Optional[Path] = None) -> None:
        """
        Salva o melhor modelo treinado.
        
        Args:
            filepath: Caminho para salvar. Se None, usa caminho padrão.
        """
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        if filepath is None:
            filepath = MODELS_DIR / "best_model.pkl"
        
        # Criar diretório se não existir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        joblib.dump(self.best_model, filepath)
        logger.info(f"Melhor modelo salvo em: {filepath}")
        
        # Salvar informações adicionais
        model_info = {
            'model': self.best_model,
            'score': self.best_score,
            'training_results': self.training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        info_filepath = filepath.parent / "model_info.pkl"
        joblib.dump(model_info, info_filepath)
        logger.info(f"Informações do modelo salvas em: {info_filepath}")
    
    def load_model(self, filepath: Path) -> Any:
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo.
            
        Returns:
            Modelo carregado.
        """
        model = joblib.load(filepath)
        logger.info(f"Modelo carregado de: {filepath}")
        return model
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Obtém importância das features para modelos que suportam.
        
        Args:
            model_name: Nome do modelo.
            feature_names: Nomes das features.
            
        Returns:
            DataFrame com importância das features.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        
        # Verificar se modelo suporta feature importance
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Modelo {model_name} não suporta feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df


def train_insurance_models(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          feature_names: List[str],
                          hyperparameter_tuning: bool = True,
                          save_models: bool = True) -> Dict[str, Any]:
    """
    Função principal para treinar modelos de predição de seguros.
    
    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        X_test: Features de teste.
        y_test: Target de teste.
        feature_names: Nomes das features.
        hyperparameter_tuning: Se deve fazer tuning de hiperparâmetros.
        save_models: Se deve salvar o melhor modelo.
        
    Returns:
        Dicionário com todos os resultados.
    """
    logger.info("Iniciando treinamento completo de modelos")
    
    # Inicializar trainer
    trainer = ModelTrainer()
    
    # Treinar todos os modelos
    training_results = trainer.train_all_models(
        X_train, y_train, hyperparameter_tuning=hyperparameter_tuning
    )
    
    # Comparar modelos
    comparison_df = trainer.compare_models()
    
    # Avaliar no conjunto de teste
    evaluation_df = trainer.evaluate_all_models(X_test, y_test)
    
    # Obter feature importance dos modelos que suportam
    feature_importance = {}
    for model_name in trainer.trained_models.keys():
        try:
            importance_df = trainer.get_feature_importance(model_name, feature_names)
            if not importance_df.empty:
                feature_importance[model_name] = importance_df
        except:
            continue
    
    # Salvar melhor modelo
    if save_models:
        trainer.save_best_model()
    
    # Compilar resultados
    results = {
        'trainer': trainer,
        'training_results': training_results,
        'comparison_df': comparison_df,
        'evaluation_df': evaluation_df,
        'feature_importance': feature_importance,
        'best_model': trainer.best_model,
        'best_score': trainer.best_score
    }
    
    logger.info("Treinamento completo finalizado")
    logger.info(f"Melhor modelo: R² = {trainer.best_score:.4f}")
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    from .data_loader import load_insurance_data
    from .preprocessing import preprocess_insurance_data
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Carregar e preprocessar dados
        data, _ = load_insurance_data()
        processed_data = preprocess_insurance_data(data)
        
        # Treinar modelos
        results = train_insurance_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test'],
            processed_data['feature_names']
        )
        
        print("✅ Treinamento concluído!")
        print(f"Melhor R²: {results['best_score']:.4f}")
        print("\n📊 Comparação de modelos:")
        print(results['comparison_df'][['Model', 'R²_Mean', 'MAE_Mean', 'RMSE_Mean']])
        
    except Exception as e:
        print(f"❌ Erro: {e}") 