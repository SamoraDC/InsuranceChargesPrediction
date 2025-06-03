"""
Sistema de logging centralizado para o projeto.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional

from ..config.settings import Config


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configura o sistema de logging.
    
    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Garantir que o diretório de logs existe
    log_dir = Config.PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuração do logging
    config = Config.LOGGING_CONFIG.copy()
    
    # Ajustar nível se fornecido
    if log_level:
        config["loggers"]["insurance_prediction"]["level"] = log_level.upper()
        config["handlers"]["console"]["level"] = log_level.upper()
    
    # Aplicar configuração
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado.
    
    Args:
        name: Nome do logger (geralmente __name__).
        
    Returns:
        Logger configurado.
    """
    # Garantir que o setup foi feito
    if not logging.getLogger("insurance_prediction").handlers:
        setup_logging()
    
    return logging.getLogger(f"insurance_prediction.{name}")


# Logger principal do módulo
logger = get_logger(__name__) 