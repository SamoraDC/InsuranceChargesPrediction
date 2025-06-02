#!/usr/bin/env python3
"""
Script para executar o pipeline completo de treinamento e salvar artefatos.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
import pandas as pd
import joblib

# Importar módulos do projeto
from src.data_loader import load_insurance_data
from src.preprocessing import preprocess_insurance_data
from src.model_training import train_insurance_models

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Executa o pipeline completo"""
    print("🚀 Iniciando pipeline completo de ML...")
    
    try:
        # 1. Carregar dados
        print("\n📊 Carregando dados...")
        data, data_info = load_insurance_data()
        print(f"✅ Dados carregados: {data.shape}")
        print(f"   {data_info}")
        
        # 2. Preprocessar dados
        print("\n🔧 Preprocessando dados...")
        processed_data = preprocess_insurance_data(
            data,
            remove_outliers=True,
            apply_transformations=True,
            create_polynomial=True,
            create_interactions=True,
            feature_selection=True
        )
        
        print(f"✅ Dados processados:")
        print(f"   Treino: {processed_data['X_train'].shape}")
        print(f"   Teste: {processed_data['X_test'].shape}")
        print(f"   Features: {len(processed_data['feature_names'])}")
        
        # 3. Salvar preprocessor
        print("\n💾 Salvando preprocessor...")
        preprocessor_path = Path("models/model_artifacts/preprocessor.pkl")
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_data['preprocessor'].save_preprocessor(preprocessor_path)
        print(f"✅ Preprocessor salvo em: {preprocessor_path}")
        
        # 4. Treinar modelos (se ainda não foram treinados)
        if not Path("models/best_model.pkl").exists():
            print("\n🤖 Treinando modelos...")
            results = train_insurance_models(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_test'],
                processed_data['y_test'],
                processed_data['feature_names']
            )
            
            print(f"✅ Melhor modelo treinado: R² = {results['best_score']:.4f}")
        else:
            print("✅ Modelos já existem, pulando treinamento")
        
        # 5. Teste de predição
        print("\n🔮 Testando predição...")
        from src.predict import predict_insurance_premium
        
        sample_data = {
            'age': 35,
            'sex': 'male',
            'bmi': 25.0,
            'children': 2,
            'smoker': 'no',
            'region': 'northeast'
        }
        
        result = predict_insurance_premium(**sample_data)
        print(f"✅ Predição teste: ${result['predicted_premium']:,.2f}")
        print(f"   Modelo: {result['model_type']}")
        print(f"   Tempo: {result['processing_time_ms']:.2f}ms")
        
        print("\n🎉 Pipeline executado com sucesso!")
        print("\n📋 Resumo do sistema:")
        print(f"   • Dataset: {data.shape[0]} registros, {data.shape[1]} colunas")
        print(f"   • Features processadas: {len(processed_data['feature_names'])}")
        print(f"   • Modelo: {result['model_type']}")
        print(f"   • Aplicação Streamlit: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 