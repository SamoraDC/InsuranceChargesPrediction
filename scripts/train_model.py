#!/usr/bin/env python3
"""
Script principal para treinamento do modelo de predição de seguros.

Este script executa o pipeline completo:
1. Carregamento e validação dos dados
2. Pré-processamento otimizado para Gradient Boosting
3. Treinamento com otimização de hiperparâmetros
4. Avaliação e salvamento do modelo
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Adicionar src ao path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from insurance_prediction.config.settings import Config
from insurance_prediction.utils.logging import setup_logging, get_logger
from insurance_prediction.data.loader import load_insurance_data
from insurance_prediction.data.preprocessor import preprocess_insurance_data
from insurance_prediction.models.trainer import train_gradient_boosting_model

# Configurar logging
setup_logging("INFO")
logger = get_logger(__name__)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Treinar modelo de predição de prêmios de seguro",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Config.RAW_DATA_PATH,
        help="Caminho para o arquivo de dados"
    )
    
    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        default=True,
        help="Otimizar hiperparâmetros (recomendado)"
    )
    
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        default=False,
        help="Pular otimização de hiperparâmetros (mais rápido)"
    )
    
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        default=False,
        help="Usar MLflow para tracking"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Config.MODELS_DIR,
        help="Diretório para salvar o modelo"
    )
    
    args = parser.parse_args()
    
    # Configurar otimização
    optimize_hyperparams = args.optimize_hyperparams and not args.no_optimize
    
    try:
        logger.info("🚀 INICIANDO TREINAMENTO DO MODELO DE SEGUROS")
        logger.info("=" * 60)
        
        # Criar diretórios necessários
        Config.setup_directories()
        
        # 1. Carregar dados
        logger.info("📊 FASE 1: Carregamento dos dados")
        data, validation_report = load_insurance_data(
            data_path=args.data_path,
            validate=True,
            clean=True
        )
        
        logger.info(f"✅ Dados carregados: {data.shape}")
        if validation_report.get('errors'):
            logger.error("❌ Erros na validação dos dados:")
            for error in validation_report['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        # 2. Pré-processamento
        logger.info("\n🔄 FASE 2: Pré-processamento")
        processed_data = preprocess_insurance_data(data)
        
        logger.info(f"✅ Pré-processamento concluído:")
        logger.info(f"  - Features originais: {processed_data['original_shape'][1]}")
        logger.info(f"  - Features finais: {len(processed_data['feature_names'])}")
        logger.info(f"  - Dados de treino: {processed_data['X_train'].shape}")
        logger.info(f"  - Dados de teste: {processed_data['X_test'].shape}")
        
        # 3. Treinamento
        logger.info(f"\n🎯 FASE 3: Treinamento do modelo")
        logger.info(f"  - Algoritmo: Gradient Boosting")
        logger.info(f"  - Otimização de hiperparâmetros: {'SIM' if optimize_hyperparams else 'NÃO'}")
        logger.info(f"  - MLflow tracking: {'SIM' if args.use_mlflow else 'NÃO'}")
        
        training_results = train_gradient_boosting_model(
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            optimize_hyperparams=optimize_hyperparams,
            save_model=True,
            use_mlflow=args.use_mlflow
        )
        
        # 4. Salvar preprocessor
        preprocessor_path = Config.MODEL_ARTIFACTS_DIR / "preprocessor.pkl"
        processed_data['preprocessor'].save_preprocessor(preprocessor_path)
        
        # 5. Resultados finais
        logger.info("\n🏆 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 60)
        
        final_metrics = training_results['final_test_metrics']
        logger.info(f"📈 MÉTRICAS FINAIS:")
        logger.info(f"  - R²: {final_metrics['r2']:.4f}")
        logger.info(f"  - R² Ajustado: {final_metrics['adjusted_r2']:.4f}")
        logger.info(f"  - MAE: {final_metrics['mae']:.2f}")
        logger.info(f"  - RMSE: {final_metrics['rmse']:.2f}")
        logger.info(f"  - MAPE: {final_metrics['mape']:.2f}%")
        logger.info(f"  - MBE: {final_metrics['mbe']:.2f}")
        
        logger.info(f"\n💾 ARQUIVOS SALVOS:")
        if 'model_path' in training_results:
            logger.info(f"  - Modelo: {training_results['model_path']}")
        logger.info(f"  - Preprocessor: {preprocessor_path}")
        
        # Top features
        feature_importance = training_results.get('feature_importance', pd.DataFrame())
        if not feature_importance.empty:
            logger.info(f"\n🔍 TOP 10 FEATURES MAIS IMPORTANTES:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Performance classification
        r2_score = final_metrics['r2']
        if r2_score >= Config.PERFORMANCE_THRESHOLDS['excellent']:
            performance = "EXCELENTE 🌟"
        elif r2_score >= Config.PERFORMANCE_THRESHOLDS['very_good']:
            performance = "MUITO BOM 👍"
        elif r2_score >= Config.PERFORMANCE_THRESHOLDS['good']:
            performance = "BOM ✅"
        elif r2_score >= Config.PERFORMANCE_THRESHOLDS['moderate']:
            performance = "MODERADO ⚠️"
        else:
            performance = "PRECISA MELHORAR ❌"
        
        logger.info(f"\n🎯 CLASSIFICAÇÃO DA PERFORMANCE: {performance}")
        
        logger.info("\n✨ Modelo pronto para uso em produção!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Treinamento interrompido pelo usuário")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 