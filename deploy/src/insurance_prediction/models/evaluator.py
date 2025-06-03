"""
Módulo de avaliação de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Classe para avaliação de modelos de regressão.
    """
    
    def __init__(self):
        """Inicializa o avaliador."""
        pass
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de avaliação.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            
        Returns:
            Dicionário com métricas.
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        } 