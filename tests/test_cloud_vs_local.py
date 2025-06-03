#!/usr/bin/env python3
"""
Teste de comparação: Modelo Cloud vs Local
"""

import sys
from pathlib import Path
import os

def test_models_comparison():
    """Compara predições entre modelo cloud e local"""
    print("🔍 COMPARANDO MODELO CLOUD vs LOCAL...")
    
    # Teste modelo local
    print("\n1️⃣ TESTANDO MODELO LOCAL:")
    try:
        sys.path.insert(0, 'src')
        from insurance_prediction.models.predictor import predict_insurance_premium
        
        test_data = {
            'age': 30,
            'sex': 'male',
            'bmi': 25.0,
            'children': 1,
            'smoker': 'no',
            'region': 'southwest'
        }
        
        local_result = predict_insurance_premium(**test_data)
        local_premium = local_result['predicted_premium']
        print(f"✅ Local: ${local_premium:.2f}")
        
    except Exception as e:
        print(f"❌ Erro local: {e}")
        local_premium = None
    
    # Teste modelo cloud
    print("\n2️⃣ TESTANDO MODELO CLOUD:")
    try:
        os.chdir('deploy')
        sys.path.insert(0, '.')
        from model_utils import load_model, predict_premium
        
        model_data = load_model()
        
        test_data = {
            'age': 30,
            'sex': 'male',
            'bmi': 25.0,
            'children': 1,
            'smoker': 'no',
            'region': 'southwest'
        }
        
        cloud_result = predict_premium(test_data, model_data)
        cloud_premium = cloud_result['predicted_premium']
        print(f"✅ Cloud: ${cloud_premium:.2f}")
        print(f"   Tipo: {cloud_result['model_type']}")
        print(f"   Features: {cloud_result['features_used']}")
        
    except Exception as e:
        print(f"❌ Erro cloud: {e}")
        cloud_premium = None
    finally:
        os.chdir('..')
    
    # Comparação
    if local_premium and cloud_premium:
        diff = abs(local_premium - cloud_premium)
        diff_pct = (diff / local_premium) * 100
        
        print(f"\n📊 COMPARAÇÃO:")
        print(f"   Local:  ${local_premium:.2f}")
        print(f"   Cloud:  ${cloud_premium:.2f}")
        print(f"   Diferença: ${diff:.2f} ({diff_pct:.2f}%)")
        
        if diff_pct < 10:
            print("✅ MODELOS COMPATÍVEIS - Diferença aceitável")
            return True
        else:
            print("⚠️ MODELOS DIFERENTES - Diferença significativa")
            return False
    else:
        print("❌ Não foi possível comparar")
        return False

if __name__ == "__main__":
    success = test_models_comparison()
    print(f"\n{'✅ TESTE APROVADO' if success else '❌ TESTE FALHOU'}")
    sys.exit(0 if success else 1) 