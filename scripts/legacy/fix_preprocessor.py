#!/usr/bin/env python3
"""
Script para corrigir e regenerar o preprocessor.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
import pandas as pd
import joblib
import numpy as np

# Importar módulos do projeto
from src.data_loader import load_insurance_data
from src.preprocessing import preprocess_insurance_data, DataPreprocessor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Regenera o preprocessor corretamente"""
    print("🔧 Regenerando preprocessor...")
    
    try:
        # 1. Carregar dados
        print("📊 Carregando dados...")
        data, _ = load_insurance_data()
        print(f"✅ Dados carregados: {data.shape}")
        
        # 2. Preprocessar dados completamente
        print("🔄 Processando dados com pipeline completo...")
        processed_data = preprocess_insurance_data(
            data,
            remove_outliers=True,
            apply_transformations=True,
            create_polynomial=True,
            create_interactions=True,
            feature_selection=True
        )
        
        print(f"✅ Processamento concluído:")
        print(f"   Treino: {processed_data['X_train'].shape}")
        print(f"   Teste: {processed_data['X_test'].shape}")
        print(f"   Features: {len(processed_data['feature_names'])}")
        
        # 3. Verificar se o preprocessor tem pipeline
        preprocessor = processed_data['preprocessor']
        print(f"🔍 Verificando preprocessor:")
        print(f"   Pipeline existe: {preprocessor.preprocessor_pipeline is not None}")
        print(f"   Features: {len(preprocessor.feature_names_) if preprocessor.feature_names_ else 0}")
        
        # 4. Se não tem pipeline, criar um novo
        if preprocessor.preprocessor_pipeline is None:
            print("⚠️ Pipeline não encontrado, criando novo...")
            
            # Criar novo preprocessor
            new_preprocessor = DataPreprocessor()
            
            # Processar dados do zero para criar pipeline
            X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].copy()
            y = data['charges'].copy()
            
            # Aplicar todas as transformações step by step
            X_clean = new_preprocessor.remove_duplicates(X)
            
            # Detectar e tratar outliers
            outliers_info = new_preprocessor.detect_outliers(X_clean)
            X_no_outliers = new_preprocessor.handle_outliers(X_clean, method='cap', outliers_info=outliers_info)
            
            # Aplicar transformações
            X_transformed = new_preprocessor.apply_transformations(X_no_outliers)
            
            # Criar features polinomiais
            X_poly = new_preprocessor.create_polynomial_features(X_transformed, degree=2)
            
            # Criar features de interação
            X_interactions = new_preprocessor.create_interaction_features(X_poly)
            
            # Criar pipeline final
            pipeline = new_preprocessor.create_preprocessing_pipeline(
                include_feature_selection=True,
                feature_selection_method='selectkbest',
                n_features=15
            )
            
            # Treinar pipeline
            pipeline.fit(X_interactions, y)
            new_preprocessor.preprocessor_pipeline = pipeline
            
            # Obter nomes das features após pipeline
            # Como o pipeline inclui seleção de features, precisamos obter os nomes corretos
            if hasattr(pipeline.named_steps['feature_selection'], 'get_support'):
                # Para SelectKBest e outros seletores
                feature_support = pipeline.named_steps['feature_selection'].get_support()
                
                # Primeiro, obter nomes após transformação categórica
                preprocessor_step = pipeline.named_steps['preprocessor']
                if hasattr(preprocessor_step, 'get_feature_names_out'):
                    all_feature_names = preprocessor_step.get_feature_names_out()
                else:
                    # Fallback para nomes manuais
                    categorical_names = []
                    for col in ['sex', 'smoker', 'region']:
                        if col == 'sex':
                            categorical_names.append('sex_male')
                        elif col == 'smoker':
                            categorical_names.append('smoker_yes')
                        elif col == 'region':
                            categorical_names.extend(['region_northwest', 'region_southeast', 'region_southwest'])
                    
                    numerical_names = ['age', 'bmi', 'children']
                    all_feature_names = numerical_names + categorical_names
                
                # Aplicar máscara de seleção
                selected_features = [name for i, name in enumerate(all_feature_names) if feature_support[i]]
                new_preprocessor.feature_names_ = selected_features
            else:
                # Usar nomes padrão se não conseguir extrair
                new_preprocessor.feature_names_ = [f"feature_{i}" for i in range(15)]
            
            print(f"✅ Novo pipeline criado com {len(new_preprocessor.feature_names_)} features")
            preprocessor = new_preprocessor
        
        # 5. Salvar preprocessor
        print("💾 Salvando preprocessor...")
        preprocessor_path = Path("models/model_artifacts/preprocessor.pkl")
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor.save_preprocessor(preprocessor_path)
        print(f"✅ Preprocessor salvo em: {preprocessor_path}")
        
        # 6. Testar carregamento
        print("🧪 Testando carregamento...")
        test_data = joblib.load(preprocessor_path)
        print(f"   Pipeline carregado: {test_data.get('pipeline') is not None}")
        print(f"   Features: {len(test_data.get('feature_names', []))}")
        
        # 7. Teste de predição
        print("🔮 Testando predição...")
        from src.predict import predict_insurance_premium
        
        result = predict_insurance_premium(35, 'male', 25.0, 2, 'no', 'northeast')
        print(f"✅ Predição teste: ${result['predicted_premium']:,.2f}")
        
        print("\n🎉 Preprocessor regenerado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 