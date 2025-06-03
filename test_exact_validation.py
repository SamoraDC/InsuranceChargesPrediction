# Teste com dados exatos do create_exact_cloud_model.py
import sys
sys.path.append('.')
from deploy.model_utils import load_model, predict_premium

# Dados específicos que foram validados no script
test_data = {
    'age': 19,
    'sex': 'female', 
    'bmi': 27.9,
    'children': 0,
    'smoker': 'yes',
    'region': 'southwest'
}

print("🧪 TESTE DE VALIDAÇÃO EXATA")
print(f"📋 Dados: {test_data}")

model_data = load_model()
result = predict_premium(test_data, model_data)
print(f'🔍 Cloud Exact: ${result["predicted_premium"]:.2f}')
print(f'✅ Esperado: $5999.56')
print(f'📊 Diferença: ${abs(result["predicted_premium"] - 5999.56):.2f}')

# Verificar se é 100% idêntico
if abs(result["predicted_premium"] - 5999.56) < 0.01:
    print("🎉 MODELO EXATO FUNCIONANDO PERFEITAMENTE!")
else:
    print("⚠️ Ainda há diferença - precisamos investigar") 