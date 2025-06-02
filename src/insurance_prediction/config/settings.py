"""
Configurações centralizadas para o projeto de predição de seguros.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

class Config:
    """Classe de configuração centralizada."""
    
    # =============================================================================
    # CAMINHOS DO PROJETO
    # =============================================================================
    
    # Diretório raiz do projeto
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    
    # Diretórios de dados
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    
    # Diretórios de modelos
    MODELS_DIR = PROJECT_ROOT / "models"
    MODEL_ARTIFACTS_DIR = MODELS_DIR / "model_artifacts"
    
    # =============================================================================
    # CONFIGURAÇÕES DE DADOS
    # =============================================================================
    
    # Arquivo de dados principal
    RAW_DATA_FILE = "insurance.csv"
    RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE
    
    # Schema dos dados
    FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
    TARGET_COLUMN = "charges"
    
    CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
    NUMERICAL_COLUMNS = ["age", "bmi", "children"]
    
    # Valores válidos para variáveis categóricas
    CATEGORICAL_VALUES = {
        "sex": ["male", "female"],
        "smoker": ["yes", "no"],
        "region": ["northeast", "northwest", "southeast", "southwest"]
    }
    
    # Ranges válidos para variáveis numéricas
    NUMERICAL_RANGES = {
        "age": {"min": 18, "max": 64},
        "bmi": {"min": 15.0, "max": 55.0},
        "children": {"min": 0, "max": 5}
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE MODELO (GRADIENT BOOSTING OTIMIZADO)
    # =============================================================================
    
    # Configurações gerais
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.15
    CV_FOLDS = 5
    
    # Gradient Boosting - Configuração otimizada
    GRADIENT_BOOSTING_CONFIG = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-4
    }
    
    # Grid para otimização de hiperparâmetros
    GRADIENT_BOOSTING_PARAM_GRID = {
        "n_estimators": [150, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 6]
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE PREPROCESSING
    # =============================================================================
    
    PREPROCESSING_CONFIG = {
        "remove_outliers": True,
        "outlier_method": "iqr",
        "scaling_method": "standard",
        "feature_selection": True,
        "n_features_to_select": 15,
        "create_polynomial_features": True,
        "polynomial_degree": 2,
        "create_interaction_features": True
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE AVALIAÇÃO
    # =============================================================================
    
    EVALUATION_METRICS = [
        "mae", "mse", "rmse", "mape", "r2", "adjusted_r2", "mbe"
    ]
    
    # Thresholds para classificação de performance
    PERFORMANCE_THRESHOLDS = {
        "excellent": 0.9,
        "very_good": 0.8,
        "good": 0.7,
        "moderate": 0.6
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE MLFLOW
    # =============================================================================
    
    MLFLOW_CONFIG = {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"),
        "experiment_name": "insurance_prediction_gradient_boosting",
        "model_name": "insurance_gb_predictor"
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE LOGGING
    # =============================================================================
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "level": "DEBUG", 
                "formatter": "detailed",
                "class": "logging.FileHandler",
                "filename": str(PROJECT_ROOT / "logs" / "insurance_prediction.log"),
                "mode": "a"
            }
        },
        "loggers": {
            "insurance_prediction": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            }
        },
        "root": {
            "handlers": ["console"],
            "level": "WARNING"
        }
    }
    
    # =============================================================================
    # CONFIGURAÇÕES DE STREAMLIT
    # =============================================================================
    
    STREAMLIT_CONFIG = {
        "page_title": "🏥 Preditor de Prêmio de Seguro",
        "page_icon": "🏥",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "theme": {
            "primaryColor": "#FF6B6B",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#F0F2F6",
            "textColor": "#262730"
        }
    }
    
    @classmethod
    def get_data_schema(cls) -> Dict[str, Any]:
        """Retorna o schema completo dos dados."""
        return {
            "features": cls.FEATURE_COLUMNS,
            "target": cls.TARGET_COLUMN,
            "categorical": cls.CATEGORICAL_COLUMNS,
            "numerical": cls.NUMERICAL_COLUMNS,
            "categorical_values": cls.CATEGORICAL_VALUES,
            "numerical_ranges": cls.NUMERICAL_RANGES
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Retorna configuração do modelo principal."""
        return cls.GRADIENT_BOOSTING_CONFIG.copy()
    
    @classmethod
    def get_param_grid(cls) -> Dict[str, List]:
        """Retorna grid de parâmetros para otimização."""
        return cls.GRADIENT_BOOSTING_PARAM_GRID.copy()
    
    @classmethod
    def setup_directories(cls) -> None:
        """Cria diretórios necessários se não existirem."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INTERIM_DATA_DIR,
            cls.MODELS_DIR,
            cls.MODEL_ARTIFACTS_DIR,
            cls.PROJECT_ROOT / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True) 