#!/usr/bin/env python3
"""
Criar modelo robusto para deploy usando dados reais do insurance.csv
Compatível com scikit-learn 1.5.x e sem dependências do src/
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_robust_model():
    """Criar modelo robusto usando dados reais"""
    
    print("🏗️ CRIANDO MODELO ROBUSTO PARA DEPLOY...")
    
    # 1. Carregar dados reais
    data_path = Path("data/insurance.csv")
    if not data_path.exists():
        raise FileNotFoundError("❌ Arquivo insurance.csv não encontrado!")
    
    df = pd.read_csv(data_path)
    print(f"✅ Dados carregados: {len(df)} registros")
    print(f"📊 Colunas: {list(df.columns)}")
    
    # 2. Preparar features
    X = df[['age', 'bmi', 'children']].copy()
    
    # Encoding das categóricas
    encoders = {}
    
    # Sex
    le_sex = LabelEncoder()
    X['sex'] = le_sex.fit_transform(df['sex'])
    encoders['sex'] = le_sex
    
    # Smoker  
    le_smoker = LabelEncoder()
    X['smoker'] = le_smoker.fit_transform(df['smoker'])
    encoders['smoker'] = le_smoker
    
    # Region
    le_region = LabelEncoder()
    X['region'] = le_region.fit_transform(df['region'])
    encoders['region'] = le_region
    
    # Features derivadas
    X['bmi_smoker'] = X['bmi'] * X['smoker']
    X['age_smoker'] = X['age'] * X['smoker']
    
    y = df['charges']
    
    print(f"📋 Features: {list(X.columns)}")
    print(f"🎯 Target: charges (min: ${y.min():.2f}, max: ${y.max():.2f})")
    
    # 3. Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Treinar modelo robusto
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    print("🔄 Treinando modelo...")
    model.fit(X_train, y_train)
    
    # 5. Avaliar performance
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"📊 PERFORMANCE:")
    print(f"   R² Train: {r2_train:.4f}")
    print(f"   R² Test:  {r2_test:.4f}")
    print(f"   MAE Train: ${mae_train:.2f}")
    print(f"   MAE Test:  ${mae_test:.2f}")
    
    # 6. Teste com amostra específica
    test_sample = {
        'age': 19,
        'sex': 'female',
        'bmi': 27.9,
        'children': 0,
        'smoker': 'yes',
        'region': 'southwest'
    }
    
    # Preparar features de teste
    test_features = [
        test_sample['age'],
        le_sex.transform([test_sample['sex']])[0],
        test_sample['bmi'],
        test_sample['children'],
        le_smoker.transform([test_sample['smoker']])[0],
        le_region.transform([test_sample['region']])[0]
    ]
    test_features.append(test_features[2] * test_features[4])  # bmi_smoker
    test_features.append(test_features[0] * test_features[4])  # age_smoker
    
    pred_test = model.predict([test_features])[0]
    print(f"🧪 TESTE AMOSTRA:")
    print(f"   Entrada: {test_sample}")
    print(f"   Predição: ${pred_test:.2f}")
    
    # 7. Criar metadados
    metadata = {
        'model_type': 'robust_deploy',
        'algorithm': 'GradientBoostingRegressor',
        'sklearn_version': '1.5.x',
        'n_estimators': 100,
        'max_depth': 4,
        'r2_score': float(r2_test),
        'mae': float(mae_test),
        'features': list(X.columns),
        'n_features': len(X.columns),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'encoders_info': {
            'sex': list(le_sex.classes_),
            'smoker': list(le_smoker.classes_),
            'region': list(le_region.classes_)
        },
        'test_prediction': {
            'input': test_sample,
            'prediction': float(pred_test)
        }
    }
    
    # 8. Salvar arquivos
    deploy_path = Path("deploy")
    deploy_path.mkdir(exist_ok=True)
    
    # Modelo
    model_path = deploy_path / "robust_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo: {model_path}")
    
    # Metadados
    metadata_path = deploy_path / "robust_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadados salvos: {metadata_path}")
    
    # Encoders
    encoders_path = deploy_path / "robust_encoders.pkl"
    joblib.dump(encoders, encoders_path)
    print(f"✅ Encoders salvos: {encoders_path}")
    
    print("🎉 MODELO ROBUSTO CRIADO COM SUCESSO!")
    return model, encoders, metadata

if __name__ == "__main__":
    create_robust_model() 