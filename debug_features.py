#!/usr/bin/env python3
"""
Debug das features para identificar diferenças entre modelo local e cloud
"""
import sys
import numpy as np
sys.path.append('.')

# Importar o sistema local
from src.insurance_prediction.models.predictor import InsurancePredictor

# Importar o sistema cloud
from deploy.model_utils import prepare_features_exact_model
import joblib

# Dados de teste
test_data = {
    'age': 19,
    'sex': 'female', 
    'bmi': 27.9,
    'children': 0,
    'smoker': 'yes',
    'region': 'southwest'
}

print("🔍 ANÁLISE DETALHADA DAS FEATURES")
print(f"📋 Dados: {test_data}")
print()

# 1. Obter features do sistema LOCAL
print("1️⃣ SISTEMA LOCAL:")
local_predictor = InsurancePredictor()
local_predictor.load_model("models/gradient_boosting_model.pkl")

# Processar os dados
import pandas as pd
df = pd.DataFrame([test_data])
X_local = local_predictor.preprocessor.fit_transform(df, None)
print(f"Features locais: {X_local[0]}")
print(f"Feature names: {local_predictor.preprocessor.feature_names_out_}")
print()

# 2. Obter features do sistema CLOUD
print("2️⃣ SISTEMA CLOUD:")
preprocessor_cloud = joblib.load('deploy/models/model_artifacts/preprocessor_exact.pkl')
X_cloud = prepare_features_exact_model(test_data, preprocessor_cloud)
print(f"Features cloud: {X_cloud[0]}")
print()

# 3. Comparar feature por feature
print("3️⃣ COMPARAÇÃO FEATURE POR FEATURE:")
for i, (local_val, cloud_val) in enumerate(zip(X_local[0], X_cloud[0])):
    diff = abs(local_val - cloud_val)
    status = "✅" if diff < 0.001 else "❌"
    print(f"{status} Feature {i}: Local={local_val:.6f}, Cloud={cloud_val:.6f}, Diff={diff:.6f}")

print()

# 4. Teste de predição
print("4️⃣ PREDIÇÕES:")
pred_local = local_predictor.predict(df)
print(f"Local: ${pred_local[0]['predicted_premium']:.2f}")

model_cloud = joblib.load('deploy/gradient_boosting_model_exact.pkl')
pred_cloud = model_cloud.predict(X_cloud)[0]
print(f"Cloud: ${pred_cloud:.2f}")
print(f"Diferença: ${abs(pred_local[0]['predicted_premium'] - pred_cloud):.2f}") 