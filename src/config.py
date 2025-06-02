"""
Configurações e constantes para o projeto de predição de seguros.
"""

import os
from pathlib import Path

# =============================================================================
# CAMINHOS DO PROJETO
# =============================================================================

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Diretórios de dados
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Diretórios de modelos
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "model_artifacts"

# Diretórios de notebooks e scripts
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# =============================================================================
# CONFIGURAÇÕES DE DADOS
# =============================================================================

# Nome do arquivo de dados principal
RAW_DATA_FILE = "insurance.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE

# Nomes das colunas esperadas no dataset
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET_COLUMN = "charges"

# Colunas categóricas e numéricas
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
NUMERICAL_COLUMNS = ["age", "bmi", "children"]

# Valores únicos esperados para variáveis categóricas
CATEGORICAL_VALUES = {
    "sex": ["male", "female"],
    "smoker": ["yes", "no"],
    "region": ["northeast", "northwest", "southeast", "southwest"]
}

# Ranges válidos para variáveis numéricas (baseado no dataset real)
NUMERICAL_RANGES = {
    "age": {"min": 18, "max": 64},
    "bmi": {"min": 15.0, "max": 55.0},  # Ajustado para acomodar valor máximo observado
    "children": {"min": 0, "max": 5}
}

# =============================================================================
# CONFIGURAÇÕES DE MODELO
# =============================================================================

# Semente para reprodutibilidade
RANDOM_STATE = 42

# Proporção de divisão de dados
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Configurações de cross-validation
CV_FOLDS = 5

# Métricas de avaliação
EVALUATION_METRICS = [
    "mean_absolute_error",
    "mean_squared_error", 
    "r2_score",
    "mean_absolute_percentage_error"
]

# =============================================================================
# CONFIGURAÇÕES DE MODELOS
# =============================================================================

# Modelos a serem testados
MODEL_CONFIGS = {
    "linear_regression": {
        "name": "Linear Regression",
        "params": {}
    },
    "ridge": {
        "name": "Ridge Regression", 
        "params": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "lasso": {
        "name": "Lasso Regression",
        "params": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "elastic_net": {
        "name": "Elastic Net",
        "params": {
            "alpha": [0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "decision_tree": {
        "name": "Decision Tree",
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    }
}

# =============================================================================
# CONFIGURAÇÕES DE PREPROCESSING
# =============================================================================

# Estratégias de tratamento de outliers
OUTLIER_METHODS = ["iqr", "zscore", "isolation_forest"]

# Métodos de transformação
TRANSFORMATION_METHODS = {
    "scaler": ["standard", "minmax", "robust"],
    "power": ["boxcox", "yeojohnson"],
    "polynomial": {"degree": [2, 3]}
}

# Configurações de feature selection
FEATURE_SELECTION_METHODS = {
    "univariate": {
        "k_best": [5, 10, 15, "all"]
    },
    "model_based": {
        "threshold": ["mean", "median", "0.1*mean"]
    },
    "recursive": {
        "n_features": [5, 10, 15]
    }
}

# =============================================================================
# CONFIGURAÇÕES DE MLFLOW
# =============================================================================

# MLflow configurações
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MLFLOW_EXPERIMENT_NAME = "insurance_prediction"
MLFLOW_MODEL_NAME = "insurance_predictor"

# Aliases do modelo
MODEL_ALIASES = {
    "development": "dev",
    "staging": "staging", 
    "production": "prod"
}

# =============================================================================
# CONFIGURAÇÕES DE STREAMLIT
# =============================================================================

# Configurações da aplicação Streamlit
STREAMLIT_CONFIG = {
    "page_title": "Preditor de Prêmio de Seguro",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configurações de cache
CACHE_TTL = 3600  # 1 hora em segundos

# =============================================================================
# CONFIGURAÇÕES DE LOGGING
# =============================================================================

# Configurações de logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
} 