#!/usr/bin/env python3
"""
Análise das diferenças entre os modelos local e deployment
"""

import joblib
import json

# Verificar modelo local
try:
    local_model = joblib.load('models/gradient_boosting_model.pkl')
    print('=== MODELO LOCAL (gradient_boosting_model.pkl) ===')
    print(f'Tipo: {type(local_model).__name__}')
    print(f'Features de entrada: {local_model.n_features_in_}')
    if hasattr(local_model, 'feature_importances_'):
        print(f'Número de features: {len(local_model.feature_importances_)}')
    
    # Verificar metadados
    with open('models/gradient_boosting_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print(f'Performance (R²): {metadata.get("r2_score", "N/A")}')
    print(f'MAE: {metadata.get("mae", "N/A")}')
    print(f'Data criação: {metadata.get("saved_at", "N/A")}')
    
except Exception as e:
    print(f'Erro modelo local: {e}')

print()

# Verificar modelo deployment
try:
    deploy_model_data = joblib.load('models/production_model_optimized.pkl')
    print('=== MODELO DEPLOYMENT (production_model_optimized.pkl) ===')
    print(f'Estrutura: {list(deploy_model_data.keys())}')
    model = deploy_model_data['model']
    print(f'Tipo: {type(model).__name__}')
    print(f'Features de entrada: {model.n_features_in_}')
    print(f'Feature names: {deploy_model_data.get("feature_names", "N/A")}')
    print(f'Metrics: {deploy_model_data.get("metrics", "N/A")}')
    
except Exception as e:
    print(f'Erro modelo deployment: {e}')

# Verificar preprocessor
print()
print('=== PREPROCESSOR ===')
try:
    preprocessor = joblib.load('models/model_artifacts/preprocessor.pkl')
    print(f'Estrutura preprocessor: {list(preprocessor.keys())}')
    print(f'Selected features: {len(preprocessor.get("selected_features", []))} features')
    print(f'Features: {preprocessor.get("selected_features", [])}')
except Exception as e:
    print(f'Erro preprocessor: {e}') 