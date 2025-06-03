#!/usr/bin/env python3
"""
Teste completo do pipeline com transformação logarítmica.
Compara métricas antes e depois da transformação log.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

from insurance_prediction.data.loader import load_insurance_data
from insurance_prediction.data.preprocessor import preprocess_insurance_data
from insurance_prediction.models.trainer import GradientBoostingTrainer
from insurance_prediction.models.log_trainer import LogTransformTrainer, compare_log_vs_original

def test_current_model():
    """Testa o modelo atual (sem log)."""
    print("🔍 TESTANDO MODELO ATUAL (SEM LOG)")
    print("=" * 60)
    
    # Carregar dados
    data, _ = load_insurance_data()
    
    # Preprocessar com método atual
    processed_data = preprocess_insurance_data(data)
    
    # Treinar modelo atual - CORRIGIDO: usar train_baseline_model
    trainer = GradientBoostingTrainer()
    results = trainer.train_baseline_model(
        processed_data['X_train'], 
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    # Extrair métricas de teste
    test_metrics = results['test_metrics']
    
    print(f"📊 MÉTRICAS MODELO ATUAL:")
    print(f"   MAE: ${test_metrics['mae']:,.2f}")
    print(f"   MSE: {test_metrics['mse']:,.2f}")
    print(f"   RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"   R²: {test_metrics['r2']:.4f}")
    print(f"   MAE/Média: {test_metrics['mae']/processed_data['y_test'].mean()*100:.1f}%")
    
    return test_metrics

def test_log_transform_model():
    """Testa o novo modelo com transformação log."""
    print("\n🔄 TESTANDO MODELO COM TRANSFORMAÇÃO LOG")
    print("=" * 60)
    
    # Carregar dados
    data, _ = load_insurance_data()
    
    # Treinar com log transform
    trainer_log = LogTransformTrainer('gradient_boosting')
    results_log = trainer_log.train(data)
    
    return results_log

def compare_all_models():
    """Compara todos os modelos disponíveis com log transform."""
    print("\n🤖 TESTANDO DIFERENTES MODELOS COM LOG")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    
    models = ['gradient_boosting', 'random_forest', 'ridge']
    results = {}
    
    for model_name in models:
        print(f"\n📈 Testando {model_name.upper()}...")
        
        trainer = LogTransformTrainer(model_name)
        result = trainer.train(data)
        
        results[model_name] = result['metrics']
        
        # Log resumido
        metrics = result['metrics']
        print(f"   MAE: ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.1f}%)")
        print(f"   R²: {metrics['r2']:.4f}")
    
    return results

def create_prediction_examples():
    """Cria exemplos de predição com o modelo log."""
    print("\n🎯 TESTANDO PREDIÇÕES COM MODELO LOG")
    print("=" * 60)
    
    # Carregar e treinar modelo
    data, _ = load_insurance_data()
    trainer = LogTransformTrainer('gradient_boosting')
    trainer.train(data)
    
    # Exemplos de teste
    test_cases = [
        {
            'name': 'Jovem não fumante',
            'age': 25, 'sex': 'male', 'bmi': 22.0, 'children': 0, 
            'smoker': 'no', 'region': 'southwest'
        },
        {
            'name': 'Adulto fumante',
            'age': 45, 'sex': 'female', 'bmi': 28.0, 'children': 2, 
            'smoker': 'yes', 'region': 'northeast'
        },
        {
            'name': 'Idoso obeso fumante',
            'age': 60, 'sex': 'male', 'bmi': 35.0, 'children': 1, 
            'smoker': 'yes', 'region': 'southeast'
        }
    ]
    
    for case in test_cases:
        # Criar DataFrame
        test_df = pd.DataFrame([case])
        test_df = test_df.drop('name', axis=1)
        
        # Predição
        start_time = time.time()
        prediction = trainer.predict(test_df)[0]
        pred_time = (time.time() - start_time) * 1000
        
        print(f"📋 {case['name']}:")
        print(f"   Perfil: {case['age']}a, {case['sex']}, BMI {case['bmi']}, {case['children']} filhos, {case['smoker']}")
        print(f"   Predição: ${prediction:,.2f} (tempo: {pred_time:.1f}ms)")

def analyze_feature_importance():
    """Analisa importância das features no modelo log."""
    print("\n📊 IMPORTÂNCIA DAS FEATURES (MODELO LOG)")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    trainer = LogTransformTrainer('gradient_boosting')
    trainer.train(data)
    
    importance = trainer.get_feature_importance()
    
    print("🏆 TOP 10 FEATURES MAIS IMPORTANTES:")
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

def main():
    """Executa todos os testes."""
    print("🔍 TESTE COMPLETO: TRANSFORMAÇÃO LOGARÍTMICA")
    print("=" * 80)
    
    # 1. Teste modelo atual
    current_results = test_current_model()
    
    # 2. Teste modelo com log
    log_results = test_log_transform_model()
    
    # 3. Comparação
    print("\n📊 COMPARAÇÃO FINAL: ATUAL vs LOG")
    print("=" * 60)
    
    current_mae = current_results['mae']
    log_mae = log_results['metrics']['mae']
    
    current_mse = current_results['mse']
    log_mse = log_results['metrics']['mse']
    
    current_r2 = current_results['r2']
    log_r2 = log_results['metrics']['r2']
    
    print(f"📈 MAE:")
    print(f"   Atual: ${current_mae:,.2f}")
    print(f"   Log:   ${log_mae:,.2f}")
    print(f"   Melhoria: {((current_mae - log_mae)/current_mae)*100:+.1f}%")
    
    print(f"\n📈 MSE:")
    print(f"   Atual: {current_mse:,.2f}")
    print(f"   Log:   {log_mse:,.2f}")
    print(f"   Melhoria: {((current_mse - log_mse)/current_mse)*100:+.1f}%")
    
    print(f"\n📈 R²:")
    print(f"   Atual: {current_r2:.4f}")
    print(f"   Log:   {log_r2:.4f}")
    print(f"   Melhoria: {((log_r2 - current_r2)/current_r2)*100:+.1f}%")
    
    # 4. Testes adicionais
    compare_all_models()
    create_prediction_examples()
    analyze_feature_importance()
    
    # 5. Conclusão
    print("\n🎯 CONCLUSÃO")
    print("=" * 40)
    if log_mae < current_mae and log_mse < current_mse:
        print("✅ TRANSFORMAÇÃO LOG É SUPERIOR!")
        print("   Recomenda-se implementar no pipeline principal.")
        
        # Calcular melhoria percentual
        mae_improvement = ((current_mae - log_mae) / current_mae) * 100
        mse_improvement = ((current_mse - log_mse) / current_mse) * 100
        
        print(f"   🎯 Melhorias obtidas:")
        print(f"      MAE: -{mae_improvement:.1f}%")
        print(f"      MSE: -{mse_improvement:.1f}%")
        
        if mae_improvement > 20 or mse_improvement > 30:
            print("   🚀 MELHORIA SIGNIFICATIVA!")
    else:
        print("⚠️ Resultados mistos. Análise adicional necessária.")
    
    # Salvar o melhor modelo
    if log_mae < current_mae:
        print("\n💾 Salvando modelo LOG otimizado...")
        data, _ = load_insurance_data()
        trainer = LogTransformTrainer('gradient_boosting')
        trainer.train(data)
        
        trainer.save_model(
            'models/gradient_boosting_log_model.pkl',
            'models/log_preprocessor.pkl'
        )
        print("✅ Modelo salvo com sucesso!")

if __name__ == "__main__":
    main() 