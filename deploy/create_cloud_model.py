#!/usr/bin/env python3
"""
Criar modelo especificamente para Streamlit Cloud
Compatível com sklearn 1.5.2 e numpy 2.x
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_cloud_compatible_model():
    """
    Cria um modelo compatível com Streamlit Cloud
    """
    logger.info("🚀 Criando modelo compatível com Streamlit Cloud...")
    
    # Carregar dados
    csv_path = Path(__file__).parent / "insurance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"📊 Dados carregados: {df.shape}")
    
    # Preparar features exatamente como no modelo original
    X = df[['age', 'bmi', 'children']].copy()
    
    # Encoding categórico
    le_sex = LabelEncoder()
    X['sex'] = le_sex.fit_transform(df['sex'])
    
    le_smoker = LabelEncoder()
    X['smoker'] = le_smoker.fit_transform(df['smoker'])
    
    le_region = LabelEncoder()
    X['region'] = le_region.fit_transform(df['region'])
    
    # Features derivadas
    X['age_smoker_risk'] = X['age'] * X['smoker']
    X['bmi_smoker_risk'] = X['bmi'] * X['smoker']
    X['age_bmi_interaction'] = X['age'] * X['bmi']
    
    # Age groups
    X['age_group'] = pd.cut(X['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    X['age_group'] = X['age_group'].astype(int)
    
    # BMI categories
    X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
    X['bmi_category'] = X['bmi_category'].astype(int)
    
    # Composite risk score
    X['composite_risk_score'] = (X['age'] * 0.1 + X['bmi'] * 0.2 + 
                                 X['smoker'] * 10 + X['children'] * 0.5)
    
    # Region density
    region_density_map = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}  # ne, nw, se, sw
    X['region_density'] = X['region'].map(region_density_map)
    
    y = df['charges']
    
    logger.info(f"📊 Features preparadas: {X.shape}")
    logger.info(f"📊 Colunas: {list(X.columns)}")
    
    # Treinar modelo com random_state fixo para compatibilidade
    model = GradientBoostingRegressor(
        max_depth=6,
        max_features='sqrt',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=200,
        n_iter_no_change=10,
        random_state=42,  # Fixo para compatibilidade
        subsample=0.8
    )
    
    logger.info("⚡ Treinando modelo...")
    model.fit(X, y)
    
    # Verificar se está treinado
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Modelo não foi treinado corretamente!")
    
    score = model.score(X, y)
    logger.info(f"📊 R² do modelo: {score:.4f}")
    
    # Teste do modelo
    test_sample = X.iloc[0:1]
    test_pred = model.predict(test_sample)[0]
    logger.info(f"🧪 Teste: ${test_pred:.2f}")
    
    # Salvar modelo
    model_path = Path(__file__).parent / "gradient_boosting_model_CLOUD.pkl"
    joblib.dump(model, model_path)
    logger.info(f"💾 Modelo salvo em: {model_path}")
    
    # Salvar encoders
    encoders = {
        'sex': le_sex,
        'smoker': le_smoker,
        'region': le_region
    }
    
    encoders_path = Path(__file__).parent / "encoders_CLOUD.pkl"
    joblib.dump(encoders, encoders_path)
    logger.info(f"💾 Encoders salvos em: {encoders_path}")
    
    # Salvar metadados
    metadata = {
        'model_type': 'cloud_compatible',
        'sklearn_version': '1.5.2',
        'features': list(X.columns),
        'r2_score': score,
        'training_samples': len(X),
        'test_prediction': float(test_pred)
    }
    
    metadata_path = Path(__file__).parent / "gradient_boosting_model_CLOUD_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"💾 Metadata salvos em: {metadata_path}")
    
    logger.info("🎉 Modelo compatível com Streamlit Cloud criado com sucesso!")
    
    return model, encoders, metadata

if __name__ == "__main__":
    try:
        model, encoders, metadata = create_cloud_compatible_model()
        print("✅ SUCESSO: Modelo compatível com Streamlit Cloud criado!")
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc() 