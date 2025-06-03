#!/usr/bin/env python3
"""
Script para comparar os modelos local e deployment
"""

# Teste modelo deployment
print("=== TESTE DEPLOYMENT MODEL ===")
try:
    from deploy.model_utils import predict_premium as predict_deploy
    
    test_data = {
        'age': 25,
        'sex': 'male', 
        'bmi': 22.6,
        'children': 0,
        'smoker': 'no',
        'region': 'southeast'
    }
    
    result_deploy = predict_deploy(test_data)
    if result_deploy['success']:
        print(f"Predição Deployment: ${result_deploy['predicted_premium']:,.2f}")
        print(f"Mensal Deployment: ${result_deploy['monthly_premium']:,.2f}")
        print(f"Model Info: {result_deploy['model_info']}")
        
        # Verificar se é modelo dummy
        if 'dummy' in str(result_deploy.get('model_info', {})).lower():
            print("⚠️  USANDO MODELO DUMMY!")
    else:
        print(f"Erro deployment: {result_deploy['error']}")
        
except Exception as e:
    print(f"Erro ao testar deployment: {e}")

# Teste modelo local
print("\n=== TESTE LOCAL MODEL ===")
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('.') / 'src'))
    
    from insurance_prediction.models.predictor import predict_insurance_premium
    
    result_local = predict_insurance_premium(
        age=25,
        sex='male',
        bmi=22.6,
        children=0,
        smoker='no',
        region='southeast'
    )
    
    if 'predictions' in result_local:
        prediction = result_local['predictions'][0]
        print(f"Predição Local: ${prediction:,.2f}")
        print(f"Mensal Local: ${prediction/12:,.2f}")
        print(f"Model Info: {result_local.get('model_info', {})}")
    else:
        print(f"Resultado local: {result_local}")
        
except Exception as e:
    print(f"Erro ao testar local: {e}")

# Verificar se o modelo do deployment existe
print("\n=== VERIFICAÇÃO ARQUIVOS ===")
import os
deploy_model = "deploy/production_model_optimized.pkl"
local_model = "models/production_model_optimized.pkl"

print(f"Deploy model existe: {os.path.exists(deploy_model)}")
print(f"Local model existe: {os.path.exists(local_model)}")

if os.path.exists(deploy_model) and os.path.exists(local_model):
    deploy_size = os.path.getsize(deploy_model)
    local_size = os.path.getsize(local_model)
    print(f"Deploy model size: {deploy_size} bytes")
    print(f"Local model size: {local_size} bytes")
    print(f"Arquivos são iguais: {deploy_size == local_size}") 