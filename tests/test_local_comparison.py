#!/usr/bin/env python3
"""
Teste de comparação entre modelo local e robusto
"""
import sys
sys.path.append('src')
from insurance_prediction.models.predictor import InsurancePredictor
import pandas as pd

# Teste com dados específicos
test_data = {
    'age': 19,
    'sex': 'female', 
    'bmi': 27.9,
    'children': 0,
    'smoker': 'yes',
    'region': 'southwest'
}

print("🔍 TESTE COMPARATIVO")
print(f"📋 Dados: {test_data}")
print()

try:
    # Modelo Local
    print("1️⃣ MODELO LOCAL:")
    predictor = InsurancePredictor()
    predictor.load_model('models/gradient_boosting_model.pkl')
    
    df = pd.DataFrame([test_data])
    result = predictor.predict(df)
    local_pred = result[0]['predicted_premium']
    print(f"   Predição: ${local_pred:.2f}")
    
except Exception as e:
    print(f"❌ Erro no modelo local: {e}")

# Modelo Robusto  
print("\n2️⃣ MODELO ROBUSTO:")
from deploy.model_utils import load_model, predict_premium

model_data = load_model()
result = predict_premium(test_data, model_data)
robust_pred = result['predicted_premium']
print(f"   Predição: ${robust_pred:.2f}")

print(f"\n📊 COMPARAÇÃO:")
print(f"   Local:   ${local_pred:.2f}")
print(f"   Robusto: ${robust_pred:.2f}")
print(f"   Diferença: ${abs(local_pred - robust_pred):.2f}")

# Verificar qual está correto usando dados reais
print(f"\n🔍 VERIFICAÇÃO COM DADOS REAIS:")
import pandas as pd
df_real = pd.read_csv('data/insurance.csv')

# Encontrar registros similares no dataset real
similar = df_real[
    (df_real['age'] == 19) & 
    (df_real['sex'] == 'female') & 
    (df_real['smoker'] == 'yes') & 
    (df_real['region'] == 'southwest')
]

if len(similar) > 0:
    print(f"   Registros similares encontrados: {len(similar)}")
    for i, row in similar.iterrows():
        print(f"   Age:{row['age']}, BMI:{row['bmi']:.1f}, Charges:${row['charges']:.2f}")
else:
    print("   Nenhum registro exatamente similar encontrado")

# Registros de fumantes jovens para referência
young_smokers = df_real[
    (df_real['age'] <= 25) & 
    (df_real['smoker'] == 'yes')
]

print(f"\n📈 FUMANTES JOVENS (≤25 anos) para referência:")
print(f"   Total: {len(young_smokers)}")
print(f"   Charges médio: ${young_smokers['charges'].mean():.2f}")
print(f"   Charges min: ${young_smokers['charges'].min():.2f}")
print(f"   Charges max: ${young_smokers['charges'].max():.2f}") 