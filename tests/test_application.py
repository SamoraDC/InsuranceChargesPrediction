#!/usr/bin/env python3
"""
Script para testar todas as funcionalidades da aplicação.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def test_data_loading():
    """Testa carregamento de dados"""
    print("📊 Testando carregamento de dados...")
    try:
        from src.data_loader import load_insurance_data
        data, info = load_insurance_data()
        print(f"✅ Dados carregados: {data.shape}")
        return True
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return False

def test_prediction():
    """Testa predição"""
    print("🔮 Testando predição...")
    try:
        from src.predict import predict_insurance_premium
        
        # Teste básico
        result = predict_insurance_premium(35, 'male', 25.0, 2, 'no', 'northeast')
        
        print(f"✅ Predição básica: ${result['predicted_premium']:,.2f}")
        
        # Teste com diferentes perfis
        test_cases = [
            (25, 'female', 22.0, 0, 'no', 'southwest'),
            (55, 'male', 30.0, 3, 'yes', 'southeast'),
            (45, 'female', 28.5, 1, 'no', 'northwest')
        ]
        
        print("🧪 Testando casos variados:")
        for i, (age, sex, bmi, children, smoker, region) in enumerate(test_cases):
            result = predict_insurance_premium(age, sex, bmi, children, smoker, region)
            print(f"   Caso {i+1}: ${result['predicted_premium']:,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_prediction():
    """Testa predição em lote"""
    print("📊 Testando predição em lote...")
    try:
        from src.predict import InsurancePredictor
        
        predictor = InsurancePredictor()
        predictor.load_model(Path("models/best_model.pkl"))
        predictor.load_preprocessor(Path("models/model_artifacts/preprocessor.pkl"))
        
        # Criar dados de teste
        test_data = pd.DataFrame([
            {'age': 25, 'sex': 'male', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'northeast'},
            {'age': 35, 'sex': 'female', 'bmi': 28.0, 'children': 2, 'smoker': 'no', 'region': 'southwest'},
            {'age': 45, 'sex': 'male', 'bmi': 32.0, 'children': 1, 'smoker': 'yes', 'region': 'southeast'}
        ])
        
        results = predictor.predict_batch(test_data)
        print(f"✅ Predição em lote: {len(results)} amostras processadas")
        print(f"   Prêmios: ${results['predicted_charges'].min():,.2f} - ${results['predicted_charges'].max():,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na predição em lote: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_components():
    """Testa componentes da aplicação"""
    print("🎨 Testando componentes da aplicação...")
    try:
        sys.path.append(str(Path(__file__).parent / "app"))
        
        # Testar helpers
        from app.utils.helpers import format_currency, validate_input_data, get_risk_category
        
        # Teste format_currency
        assert format_currency(12345.67) == "$12,345.67"
        print("✅ format_currency funcionando")
        
        # Teste validate_input_data
        valid_data = {
            'age': 35,
            'sex': 'male',
            'bmi': 25.0,
            'children': 2,
            'smoker': 'no',
            'region': 'northeast'
        }
        is_valid, errors = validate_input_data(valid_data)
        assert is_valid == True
        print("✅ validate_input_data funcionando")
        
        # Teste get_risk_category
        category, color, emoji = get_risk_category(15000)
        print(f"✅ get_risk_category: {category} {emoji}")
        
        return True
    except Exception as e:
        print(f"❌ Erro nos componentes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_charts():
    """Testa geração de gráficos"""
    print("📈 Testando geração de gráficos...")
    try:
        sys.path.append(str(Path(__file__).parent / "app"))
        from app.components.charts import create_prediction_gauge, create_feature_importance_chart
        
        # Teste gauge
        gauge_fig = create_prediction_gauge(15000)
        print("✅ Gráfico gauge criado")
        
        # Teste feature importance (simulado)
        feature_importance = {
            'age': 0.3,
            'bmi': 0.25,
            'smoker_yes': 0.4,
            'children': 0.05
        }
        importance_fig = create_feature_importance_chart(feature_importance)
        if importance_fig:
            print("✅ Gráfico de importância criado")
        
        return True
    except Exception as e:
        print(f"❌ Erro nos gráficos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_performance():
    """Testa performance do modelo"""
    print("🎯 Testando performance do modelo...")
    try:
        from src.data_loader import load_insurance_data
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        import joblib
        
        # Carregar dados
        data, _ = load_insurance_data()
        X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
        y = data['charges']
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Carregar modelo e preprocessor
        model = joblib.load("models/best_model.pkl")
        preprocessor_data = joblib.load("models/model_artifacts/preprocessor.pkl")
        pipeline = preprocessor_data['pipeline']
        
        # Testar performance
        X_test_transformed = pipeline.transform(X_test)
        y_pred = model.predict(X_test_transformed)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✅ Performance do modelo:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: ${mae:,.2f}")
        
        # Classificar performance
        if r2 > 0.7:
            print("✅ Performance BOA")
        elif r2 > 0.6:
            print("⚠️ Performance MODERADA")
        else:
            print("❌ Performance RUIM")
        
        return True
    except Exception as e:
        print(f"❌ Erro na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes"""
    print("🧪 INICIANDO TESTES DA APLICAÇÃO")
    print("=" * 50)
    
    tests = [
        ("Carregamento de Dados", test_data_loading),
        ("Predição Individual", test_prediction),
        ("Predição em Lote", test_batch_prediction),
        ("Componentes da App", test_app_components),
        ("Geração de Gráficos", test_charts),
        ("Performance do Modelo", test_model_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print(f"\n{'='*20} RESUMO DOS TESTES {'='*20}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:<25} {status}")
    
    print(f"\n📊 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Aplicação está funcionando corretamente.")
        print("\n🚀 A aplicação está pronta para uso:")
        print("   • Streamlit App: http://localhost:8501")
        print("   • Predições funcionando")
        print("   • Componentes carregados")
        print("   • Modelo treinado e avaliado")
    else:
        print(f"⚠️ {total - passed} teste(s) falharam. Verifique os erros acima.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main()) 