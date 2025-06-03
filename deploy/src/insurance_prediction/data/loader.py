"""
Módulo para carregamento e validação de dados de seguros.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

from ..config.settings import Config
from ..utils.logging import get_logger

# Configurar logger
logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class DataLoader:
    """
    Classe responsável pelo carregamento e validação inicial dos dados.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Inicializa o DataLoader.
        
        Args:
            data_path: Caminho para o arquivo de dados. Se None, usa padrão.
        """
        self.data_path = data_path or Config.RAW_DATA_PATH
        self.data = None
        self._validation_report = {}
    
    def load_data(self, validate: bool = True) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV.
        
        Args:
            validate: Se True, executa validação dos dados após carregamento.
            
        Returns:
            DataFrame com os dados carregados.
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se os dados não passarem na validação.
        """
        try:
            logger.info(f"Carregando dados de: {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
            
            # Carregar dados
            self.data = pd.read_csv(self.data_path)
            
            # Log informações básicas
            logger.info(f"Dados carregados: {self.data.shape}")
            logger.info(f"Colunas: {list(self.data.columns)}")
            logger.info(f"Tipos de dados: {self.data.dtypes.to_dict()}")
            
            # Validar se solicitado
            if validate:
                self._validate_data()
                
            return self.data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _validate_data(self) -> Dict[str, Any]:
        """
        Valida a estrutura e qualidade dos dados carregados.
        
        Returns:
            Relatório de validação.
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        logger.info("Iniciando validação dos dados...")
        
        report = {
            "shape": self.data.shape,
            "missing_values": {},
            "column_validation": {},
            "data_types": {},
            "duplicates": self.data.duplicated().sum(),
            "warnings": [],
            "errors": []
        }
        
        # Validações
        self._validate_columns(report)
        self._validate_data_types(report)
        self._validate_missing_values(report)
        self._validate_value_ranges(report)
        self._validate_categorical_values(report)
        
        self._validation_report = report
        
        # Log resultados
        self._log_validation_results(report)
        
        return report
    
    def _validate_columns(self, report: Dict[str, Any]) -> None:
        """Valida se todas as colunas esperadas estão presentes."""
        expected_columns = Config.FEATURE_COLUMNS + [Config.TARGET_COLUMN]
        actual_columns = list(self.data.columns)
        
        missing_columns = set(expected_columns) - set(actual_columns)
        extra_columns = set(actual_columns) - set(expected_columns)
        
        if missing_columns:
            error_msg = f"Colunas obrigatórias ausentes: {missing_columns}"
            report["errors"].append(error_msg)
            logger.error(error_msg)
        
        if extra_columns:
            warning_msg = f"Colunas extras encontradas: {extra_columns}"
            report["warnings"].append(warning_msg)
            logger.warning(warning_msg)
        
        report["column_validation"] = {
            "expected": expected_columns,
            "actual": actual_columns,
            "missing": list(missing_columns),
            "extra": list(extra_columns)
        }
    
    def _validate_data_types(self, report: Dict[str, Any]) -> None:
        """Valida os tipos de dados das colunas."""
        data_types = {}
        
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            data_types[col] = dtype
            
            # Verificar se colunas numéricas são realmente numéricas
            if col in Config.NUMERICAL_COLUMNS:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    error_msg = f"Coluna {col} deveria ser numérica, mas é {dtype}"
                    report["errors"].append(error_msg)
                    logger.error(error_msg)
        
        report["data_types"] = data_types
    
    def _validate_missing_values(self, report: Dict[str, Any]) -> None:
        """Valida valores ausentes."""
        missing_values = {}
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            
            missing_values[col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }
            
            if missing_count > 0:
                warning_msg = f"Coluna {col}: {missing_count} valores ausentes ({missing_pct:.2f}%)"
                report["warnings"].append(warning_msg)
                logger.warning(warning_msg)
        
        report["missing_values"] = missing_values
    
    def _validate_value_ranges(self, report: Dict[str, Any]) -> None:
        """Valida se valores numéricos estão dentro dos ranges esperados."""
        for col in Config.NUMERICAL_COLUMNS:
            if col in self.data.columns:
                col_data = self.data[col].dropna()
                
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    expected_range = Config.NUMERICAL_RANGES.get(col, {})
                    expected_min = expected_range.get("min")
                    expected_max = expected_range.get("max")
                    
                    if expected_min is not None and min_val < expected_min:
                        warning_msg = f"Coluna {col}: valor mínimo {min_val} < esperado {expected_min}"
                        report["warnings"].append(warning_msg)
                        logger.warning(warning_msg)
                    
                    if expected_max is not None and max_val > expected_max:
                        warning_msg = f"Coluna {col}: valor máximo {max_val} > esperado {expected_max}"
                        report["warnings"].append(warning_msg)
                        logger.warning(warning_msg)
    
    def _validate_categorical_values(self, report: Dict[str, Any]) -> None:
        """Valida se valores categóricos estão dentro dos valores esperados."""
        for col in Config.CATEGORICAL_COLUMNS:
            if col in self.data.columns:
                unique_values = set(self.data[col].dropna().unique())
                expected_values = set(Config.CATEGORICAL_VALUES.get(col, []))
                
                unexpected_values = unique_values - expected_values
                
                if unexpected_values:
                    warning_msg = f"Coluna {col}: valores inesperados encontrados: {unexpected_values}"
                    report["warnings"].append(warning_msg)
                    logger.warning(warning_msg)
    
    def _log_validation_results(self, report: Dict[str, Any]) -> None:
        """Log dos resultados da validação."""
        logger.info("=== RELATÓRIO DE VALIDAÇÃO ===")
        logger.info(f"Shape dos dados: {report['shape']}")
        logger.info(f"Duplicatas: {report['duplicates']}")
        logger.info(f"Total de warnings: {len(report['warnings'])}")
        logger.info(f"Total de errors: {len(report['errors'])}")
        
        if report['errors']:
            logger.error("ERROS ENCONTRADOS:")
            for error in report['errors']:
                logger.error(f"  - {error}")
        
        if report['warnings']:
            logger.warning("WARNINGS ENCONTRADOS:")
            for warning in report['warnings']:
                logger.warning(f"  - {warning}")
        
        logger.info("=== FIM DO RELATÓRIO ===")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Retorna o último relatório de validação.
        
        Returns:
            Dicionário com o relatório de validação.
        """
        return self._validation_report
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo estatístico dos dados.
        
        Returns:
            Dicionário com estatísticas resumidas.
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values_total": self.data.isnull().sum().sum(),
            "duplicated_rows": self.data.duplicated().sum()
        }
        
        # Estatísticas descritivas para colunas numéricas
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = self.data[numeric_cols].describe().to_dict()
        
        # Contagem de valores únicos para colunas categóricas
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {}
            for col in categorical_cols:
                summary["categorical_summary"][col] = {
                    "unique_count": self.data[col].nunique(),
                    "unique_values": list(self.data[col].unique()),
                    "value_counts": self.data[col].value_counts().to_dict()
                }
        
        return summary
    
    def clean_data(self) -> pd.DataFrame:
        """
        Aplica limpeza básica nos dados.
        
        Returns:
            DataFrame com dados limpos.
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        logger.info("Iniciando limpeza básica dos dados...")
        
        # Copiar dados
        clean_data = self.data.copy()
        
        # Remover duplicatas
        initial_shape = clean_data.shape
        clean_data = clean_data.drop_duplicates()
        removed_duplicates = initial_shape[0] - clean_data.shape[0]
        
        if removed_duplicates > 0:
            logger.info(f"Removidas {removed_duplicates} linhas duplicadas")
        
        # Converter tipos de dados se necessário
        for col in Config.NUMERICAL_COLUMNS:
            if col in clean_data.columns:
                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
        
        # Garantir que colunas categóricas são strings
        for col in Config.CATEGORICAL_COLUMNS:
            if col in clean_data.columns:
                clean_data[col] = clean_data[col].astype(str)
        
        logger.info(f"Limpeza concluída. Shape final: {clean_data.shape}")
        
        return clean_data


def load_insurance_data(data_path: Optional[Path] = None, 
                       validate: bool = True, 
                       clean: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Função utilitária para carregar dados de seguro.
    
    Args:
        data_path: Caminho para o arquivo de dados.
        validate: Se deve validar os dados após carregamento.
        clean: Se deve aplicar limpeza básica.
        
    Returns:
        Tupla contendo (DataFrame dos dados, relatório de validação).
    """
    loader = DataLoader(data_path)
    
    # Carregar dados
    data = loader.load_data(validate=validate)
    
    # Aplicar limpeza se solicitado
    if clean:
        data = loader.clean_data()
    
    # Obter relatório
    report = loader.get_validation_report() if validate else {}
    
    return data, report


if __name__ == "__main__":
    # Configurar logging para testes
    from ..utils.logging import setup_logging
    setup_logging("INFO")
    
    try:
        # Testar carregamento
        data, validation_report = load_insurance_data()
        
        logger.info(f"✅ Dados carregados com sucesso: {data.shape}")
        logger.info(f"Primeiras 5 linhas:")
        logger.info(f"\n{data.head()}")
        
        # Testar resumo
        loader = DataLoader()
        loader.data = data
        summary = loader.get_data_summary()
        
        logger.info(f"📊 Resumo dos dados:")
        logger.info(f"Memória: {summary['memory_usage_mb']:.2f} MB")
        logger.info(f"Valores únicos por coluna categórica:")
        for col, info in summary.get('categorical_summary', {}).items():
            logger.info(f"  {col}: {info['unique_count']} valores únicos")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}") 