"""
Insurance Prediction Package

Este pacote fornece ferramentas completas para predição de prêmios de seguro
usando Gradient Boosting como modelo principal.
"""

__version__ = "1.0.0"
__author__ = "Insurance Prediction Team"

from .models.predictor import InsurancePredictor
from .models.trainer import GradientBoostingTrainer
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor

__all__ = [
    "InsurancePredictor",
    "GradientBoostingTrainer", 
    "DataLoader",
    "DataPreprocessor"
] 