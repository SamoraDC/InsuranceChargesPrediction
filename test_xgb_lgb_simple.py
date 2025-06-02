#!/usr/bin/env python3
"""
Script simples para testar XGBoost e LightGBM especificamente.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# XGBoost e LightGBM
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Função para carregar dados simples
def load_simple_data():
    """Carrega dados de forma simples."""
    data_path = Path("data/raw/insurance.csv")
    data = pd.read_csv(data_path)
    return data

# Preprocessamento simples
def simple_preprocessing(data):
    """Preprocessamento básico."""
    # Separar features e target
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    # Encoding simples para categorias
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X_processed = X.copy()
    X_processed['sex'] = le_sex.fit_transform(X['sex'])
    X_processed['smoker'] = le_smoker.fit_transform(X['smoker'])
    X_processed['region'] = le_region.fit_transform(X['region'])
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred, n_features):
    """Calcula todas as métricas."""
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R²': r2,
        'Adjusted_R²': 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1),
        'MBE': np.mean(y_pred - y_true)
    }

def test_model(model_name, model, X_train, y_train, X_test, y_test):
    """Testa um modelo específico."""
    print(f"\n🎯 Testando {model_name}...")
    
    start_time = datetime.now()
    
    try:
        # Treinar
        print(f"   Treinando...")
        model.fit(X_train, y_train)
        
        # Predizer
        print(f"   Fazendo predições...")
        y_pred = model.predict(X_test)
        
        # Métricas
        print(f"   Calculando métricas...")
        metrics = calculate_metrics(y_test, y_pred, X_train.shape[1])
        
        # Cross-validation
        print(f"   Validação cruzada...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')  # Reduzir para 3 folds
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Resultados
        print(f"✅ {model_name} - Concluído em {training_time:.2f}s")
        print(f"   R²: {metrics['R²']:.4f}")
        print(f"   Adjusted R²: {metrics['Adjusted_R²']:.4f}")
        print(f"   MAE: {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   MBE: {metrics['MBE']:.2f}")
        print(f"   CV R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'training_time': training_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Erro em {model_name}: {e}")
        return {
            'model': None,
            'error': str(e),
            'success': False
        }

def main():
    """Função principal."""
    print("🚀 Teste SIMPLES - XGBoost vs LightGBM")
    print("=" * 50)
    
    try:
        # 1. Carregar dados
        print("📊 Carregando dados...")
        data = load_simple_data()
        print(f"✅ Dados carregados: {data.shape}")
        
        # 2. Preprocessamento simples
        print("🔄 Preprocessamento simples...")
        X_train, X_test, y_train, y_test = simple_preprocessing(data)
        print(f"✅ Preprocessamento concluído:")
        print(f"   Treino: {X_train.shape}")
        print(f"   Teste: {X_test.shape}")
        
        # 3. Testar modelos
        models = {
            'XGBoost': XGBRegressor(
                n_estimators=50,  # Reduzir para ser mais rápido
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                eval_metric='rmse'
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=50,  # Reduzir para ser mais rápido
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
                metric='rmse'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            result = test_model(model_name, model, X_train, y_train, X_test, y_test)
            results[model_name] = result
        
        # 4. Comparar resultados
        print("\n📊 COMPARAÇÃO FINAL:")
        print("=" * 60)
        print(f"{'Model':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Time':<8}")
        print("-" * 60)
        
        best_model = None
        best_r2 = -1
        
        for name, result in results.items():
            if result['success']:
                metrics = result['metrics']
                time_str = f"{result['training_time']:.1f}s"
                
                print(f"{name:<12} {metrics['R²']:<8.4f} {metrics['MAE']:<8.2f} "
                      f"{metrics['RMSE']:<8.2f} {metrics['MAPE']:<8.2f} {time_str:<8}")
                
                if metrics['R²'] > best_r2:
                    best_r2 = metrics['R²']
                    best_model = name
            else:
                print(f"{name:<12} ERRO")
        
        print("=" * 60)
        
        if best_model:
            print(f"🏆 MELHOR MODELO: {best_model}")
            best_metrics = results[best_model]['metrics']
            print(f"   R²: {best_metrics['R²']:.4f}")
            print(f"   MAE: {best_metrics['MAE']:.2f}")
            print(f"   RMSE: {best_metrics['RMSE']:.2f}")
            
            # Salvar melhor modelo
            model_path = Path("models/best_model_simple.pkl")
            model_path.parent.mkdir(exist_ok=True)
            joblib.dump(results[best_model]['model'], model_path)
            print(f"✅ Melhor modelo salvo: {model_path}")
            
        return results
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("\n🎉 Teste concluído com sucesso!")
    else:
        print("\n❌ Teste falhou!") 