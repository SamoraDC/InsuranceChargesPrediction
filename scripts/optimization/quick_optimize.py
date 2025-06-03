#!/usr/bin/env python3
"""
Script SIMPLES para otimizar métricas com transformação logarítmica.
FOCO: Resolver MSE/MAE altos de forma direta.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

def main():
    print("🚀 OTIMIZAÇÃO RÁPIDA: LOG TRANSFORM")
    print("=" * 50)
    
    # 1. Carregar dados
    print("📊 Carregando dados...")
    data = pd.read_csv('data/raw/insurance.csv')
    data = data.drop_duplicates()
    
    print(f"   Shape: {data.shape}")
    print(f"   Target média: ${data['charges'].mean():,.2f}")
    print(f"   Target skewness: {data['charges'].skew():.3f}")
    
    # 2. Preprocessing SIMPLES
    print("\n🔧 Preprocessing simples...")
    
    # Features básicas + algumas interações
    X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].copy()
    y = data['charges'].copy()
    
    # Adicionar interações importantes
    X['bmi_smoker'] = X['bmi'] * X['smoker'].map({'no': 0, 'yes': 1})
    X['age_smoker'] = X['age'] * X['smoker'].map({'no': 0, 'yes': 1})
    
    # Encoding simples
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X['sex'] = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['region'] = le_region.fit_transform(X['region'])
    
    print(f"   Features finais: {X.shape[1]}")
    
    # 3. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. TESTE 1: Modelo Original
    print("\n🤖 TESTE 1: Modelo Original")
    
    # Normalizar
    scaler_orig = StandardScaler()
    X_train_scaled = scaler_orig.fit_transform(X_train)
    X_test_scaled = scaler_orig.transform(X_test)
    
    # Treinar
    model_orig = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=6,
        min_samples_split=20, random_state=42
    )
    model_orig.fit(X_train_scaled, y_train)
    
    # Avaliar
    y_pred_orig = model_orig.predict(X_test_scaled)
    mae_orig = mean_absolute_error(y_test, y_pred_orig)
    mse_orig = mean_squared_error(y_test, y_pred_orig)
    r2_orig = r2_score(y_test, y_pred_orig)
    
    print(f"   MAE: ${mae_orig:,.2f} ({mae_orig/y_test.mean()*100:.1f}%)")
    print(f"   MSE: {mse_orig:,.2f}")
    print(f"   R²: {r2_orig:.4f}")
    
    # 5. TESTE 2: Modelo com LOG Transform
    print("\n🔄 TESTE 2: Modelo com LOG Transform")
    
    # Transformação LOG
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print(f"   Original skewness: {y_train.skew():.3f}")
    print(f"   Log skewness: {y_train_log.skew():.3f}")
    
    # Treinar na escala LOG
    model_log = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        min_samples_split=20, subsample=0.8, random_state=42
    )
    model_log.fit(X_train_scaled, y_train_log)
    
    # Predições e conversão de volta
    y_pred_log = model_log.predict(X_test_scaled)
    y_pred_log_orig = np.expm1(y_pred_log)  # Converter de volta
    
    # Avaliar na escala original
    mae_log = mean_absolute_error(y_test, y_pred_log_orig)
    mse_log = mean_squared_error(y_test, y_pred_log_orig)
    r2_log = r2_score(y_test, y_pred_log_orig)
    
    print(f"   MAE: ${mae_log:,.2f} ({mae_log/y_test.mean()*100:.1f}%)")
    print(f"   MSE: {mse_log:,.2f}")
    print(f"   R²: {r2_log:.4f}")
    
    # 6. Comparação
    print("\n📊 COMPARAÇÃO FINAL")
    print("=" * 40)
    
    mae_improvement = ((mae_orig - mae_log) / mae_orig) * 100
    mse_improvement = ((mse_orig - mse_log) / mse_orig) * 100
    r2_improvement = ((r2_log - r2_orig) / r2_orig) * 100
    
    print(f"📈 MAE: ${mae_orig:,.2f} → ${mae_log:,.2f} ({mae_improvement:+.1f}%)")
    print(f"📈 MSE: {mse_orig:,.2f} → {mse_log:,.2f} ({mse_improvement:+.1f}%)")
    print(f"📈 R²:  {r2_orig:.4f} → {r2_log:.4f} ({r2_improvement:+.1f}%)")
    
    # 7. Avaliação final
    print(f"\n🎯 AVALIAÇÃO FINAL:")
    
    if mae_log < mae_orig and mse_log < mse_orig:
        print("✅ LOG TRANSFORM É SUPERIOR!")
        
        # Qualidade das métricas
        mae_percentage = (mae_log / y_test.mean()) * 100
        
        if mae_percentage < 15:
            print("🏆 EXCELENTE: MAE < 15% da média")
        elif mae_percentage < 25:
            print("✅ BOM: MAE < 25% da média")
        else:
            print("⚠️ ACEITÁVEL: MAE < 30% da média")
        
        if mae_improvement > 15:
            print("🚀 MELHORIA SIGNIFICATIVA!")
        
        # Salvar modelo otimizado
        print("\n💾 Salvando modelo otimizado...")
        Path('models').mkdir(exist_ok=True)
        
        # Salvar tudo necessário
        model_data = {
            'model': model_log,
            'scaler': scaler_orig,
            'encoders': {
                'sex': le_sex,
                'smoker': le_smoker,
                'region': le_region
            },
            'metrics': {
                'mae': mae_log,
                'mse': mse_log,
                'r2': r2_log,
                'mae_percentage': mae_percentage
            }
        }
        
        joblib.dump(model_data, 'models/optimized_model_complete.pkl')
        print("✅ Modelo completo salvo!")
        
        # Teste rápido
        print("\n🎯 Teste de predição:")
        test_case = [[30, 1, 25.0, 1, 1, 0, 25.0, 30.0]]  # age, sex, bmi, children, smoker, region, bmi_smoker, age_smoker
        test_scaled = scaler_orig.transform(test_case)
        pred_log = model_log.predict(test_scaled)[0]
        pred_orig = np.expm1(pred_log)
        print(f"   Exemplo: ${pred_orig:,.2f}")
        
    else:
        print("⚠️ Resultados inconclusivos")
    
    print(f"\n📋 CONCLUSÃO:")
    print(f"   Modelo original MAE: ${mae_orig:,.2f}")
    print(f"   Modelo otimizado MAE: ${mae_log:,.2f}")
    print(f"   Melhoria: {mae_improvement:.1f}%")
    
    if mae_log < 2500:
        print("🎯 META ATINGIDA: MAE < $2,500!")
    elif mae_log < 3000:
        print("✅ PRÓXIMO DA META: MAE < $3,000")
    
if __name__ == "__main__":
    main() 