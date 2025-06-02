#!/usr/bin/env python3
"""
Script simplificado para corrigir o pipeline do preprocessor.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression

def main():
    """Corrige o pipeline do preprocessor"""
    print("🔧 Corrigindo pipeline do preprocessor...")
    
    try:
        # 1. Carregar preprocessor existente
        print("📊 Carregando preprocessor existente...")
        preprocessor_path = Path("models/model_artifacts/preprocessor.pkl")
        
        if not preprocessor_path.exists():
            print("❌ Preprocessor não encontrado!")
            return 1
        
        preprocessor_data = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor carregado: {list(preprocessor_data.keys())}")
        
        # 2. Criar pipeline simples baseado no que foi salvo
        print("🔨 Criando pipeline simples...")
        
        # Definir colunas
        numerical_cols = ['age', 'bmi', 'children']
        categorical_cols = ['sex', 'smoker', 'region']
        
        # Criar transformadores
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # Criar ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Criar pipeline com seleção de features
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression, k=15))
        ])
        
        # 3. Treinar pipeline com dados simples
        print("🎯 Treinando pipeline...")
        
        # Carregar dados
        sys.path.append('src')
        from src.data_loader import load_insurance_data
        data, _ = load_insurance_data()
        
        # Preparar dados
        X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].copy()
        y = data['charges'].copy()
        
        # Treinar pipeline
        pipeline.fit(X, y)
        print("✅ Pipeline treinado!")
        
        # 4. Obter nomes das features
        print("📝 Extraindo nomes das features...")
        
        # Transformar dados para obter nomes
        X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X)
        
        # Obter nomes das features do ColumnTransformer
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Aplicar seleção de features
        feature_selector = pipeline.named_steps['feature_selection']
        selected_mask = feature_selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        print(f"✅ Features selecionadas: {len(selected_features)}")
        print(f"   Nomes: {selected_features[:5]}...")
        
        # 5. Atualizar dados do preprocessor
        print("💾 Atualizando preprocessor...")
        
        preprocessor_data['pipeline'] = pipeline
        preprocessor_data['feature_names'] = selected_features
        
        # Salvar preprocessor atualizado
        joblib.dump(preprocessor_data, preprocessor_path)
        print(f"✅ Preprocessor atualizado salvo em: {preprocessor_path}")
        
        # 6. Testar carregamento
        print("🧪 Testando carregamento...")
        test_data = joblib.load(preprocessor_path)
        test_pipeline = test_data.get('pipeline')
        
        print(f"   Pipeline existe: {test_pipeline is not None}")
        print(f"   Features: {len(test_data.get('feature_names', []))}")
        
        # 7. Teste de transformação
        print("🔮 Testando transformação...")
        
        # Dados de teste
        test_sample = pd.DataFrame([{
            'age': 35,
            'sex': 'male',
            'bmi': 25.0,
            'children': 2,
            'smoker': 'no',
            'region': 'northeast'
        }])
        
        # Transformar
        transformed = test_pipeline.transform(test_sample)
        print(f"✅ Transformação bem-sucedida: {transformed.shape}")
        
        # 8. Teste de predição completa
        print("🎯 Testando predição completa...")
        from src.predict import predict_insurance_premium
        
        result = predict_insurance_premium(35, 'male', 25.0, 2, 'no', 'northeast')
        print(f"✅ Predição teste: ${result['predicted_premium']:,.2f}")
        
        print("\n🎉 Preprocessor corrigido com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 