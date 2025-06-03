#!/usr/bin/env python3
"""
Script simples para treinar o modelo otimizado.
Execute apenas se não tiver modelo ou quiser retreinar.
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
    print("🤖 Treinando modelo otimizado...")
    
    # Carregar dados
    data = pd.read_csv('data/raw/insurance.csv')
    data = data.drop_duplicates()
    
    # Preprocessing simples
    X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].copy()
    y = data['charges'].copy()
    
    # Interações essenciais
    X['bmi_smoker'] = X['bmi'] * X['smoker'].map({'no': 0, 'yes': 1})
    X['age_smoker'] = X['age'] * X['smoker'].map({'no': 0, 'yes': 1})
    
    # Encoding
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X['sex'] = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['region'] = le_region.fit_transform(X['region'])
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo otimizado
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=20,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Validar
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ MAE: ${mae:,.2f} ({mae/y_test.mean()*100:.1f}%)")
    print(f"✅ MSE: {mse:,.0f}")
    print(f"✅ R²: {r2:.4f}")
    
    # Salvar modelo completo
    Path('models').mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'encoders': {
            'sex': le_sex,
            'smoker': le_smoker,
            'region': le_region
        },
        'feature_names': list(X.columns),
        'metrics': {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mae_percentage': (mae / y_test.mean()) * 100
        }
    }
    
    joblib.dump(model_data, 'models/production_model_optimized.pkl')
    print("💾 Modelo salvo: models/production_model_optimized.pkl")
    
    # Teste rápido
    print("\n🎯 Teste:")
    test_data = [[35, 1, 25.0, 2, 0, 0, 0, 0]]  # não fumante
    test_scaled = scaler.transform(test_data)
    pred = model.predict(test_scaled)[0]
    print(f"   Predição: ${pred:,.2f}")

if __name__ == "__main__":
    main() 