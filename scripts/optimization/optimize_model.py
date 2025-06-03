#!/usr/bin/env python3
"""
Script otimizado para implementar a melhor solução baseada em transformação logarítmica.
FOCO: Resolver problemas de MSE/MAE altos através de transformação de dados.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
from typing import Dict, Any, Tuple

def load_and_clean_data():
    """Carrega e limpa os dados básicos."""
    print("📊 Carregando dados...")
    
    # Carregar dados
    data = pd.read_csv('data/raw/insurance.csv')
    
    # Limpeza básica
    data = data.drop_duplicates()
    
    print(f"   Dados carregados: {data.shape}")
    print(f"   Target - Média: ${data['charges'].mean():,.2f}")
    print(f"   Target - Skewness: {data['charges'].skew():.3f}")
    
    return data

def create_optimized_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria features otimizadas (menos redundância)."""
    df = data.copy()
    
    # Apenas interações essenciais
    df['bmi_smoker'] = df['bmi'] * df['smoker'].map({'no': 0, 'yes': 1})
    df['age_smoker'] = df['age'] * df['smoker'].map({'no': 0, 'yes': 1})
    
    # Categorias simples - CORRIGIDO: usar cut com duplicates='drop'
    try:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2], duplicates='drop')
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 25, 30, 50], labels=[0, 1, 2], duplicates='drop')
    except Exception:
        # Fallback para categorização manual
        df['age_group'] = 0
        df.loc[df['age'] >= 30, 'age_group'] = 1
        df.loc[df['age'] >= 50, 'age_group'] = 2
        
        df['bmi_category'] = 0
        df.loc[df['bmi'] >= 25, 'bmi_category'] = 1
        df.loc[df['bmi'] >= 30, 'bmi_category'] = 2
    
    # Risk score
    df['risk_score'] = (
        df['age'] * 0.1 + 
        df['bmi'] * 0.2 + 
        df['smoker'].map({'no': 0, 'yes': 10})
    )
    
    # Verificar e tratar NaN
    if df.isnull().any().any():
        print(f"   ⚠️ Encontrados NaN, preenchendo com médias...")
        df = df.fillna(df.mean(numeric_only=True))
    
    return df

def preprocess_data_original(data: pd.DataFrame) -> Dict[str, Any]:
    """Preprocessamento original (sem log)."""
    print("\n🔧 Preprocessamento ORIGINAL...")
    
    # Features
    df = create_optimized_features(data)
    
    # Separar target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Encoding categórico
    categorical_cols = ['sex', 'smoker', 'region']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': label_encoders
    }

def preprocess_data_log(data: pd.DataFrame) -> Dict[str, Any]:
    """Preprocessamento com transformação LOG."""
    print("\n🔄 Preprocessamento com LOG TRANSFORM...")
    
    # Features
    df = create_optimized_features(data)
    
    # Separar target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # TRANSFORMAÇÃO LOG na target
    y_log = np.log1p(y)  # log(1+x)
    
    print(f"   Original Skewness: {y.skew():.3f}")
    print(f"   Log Skewness: {y_log.skew():.3f}")
    
    # Encoding categórico
    categorical_cols = ['sex', 'smoker', 'region']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Divisão treino/teste
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Seleção de features (reduzir overfitting)
    selector = SelectKBest(f_regression, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train_log)
    X_test_selected = selector.transform(X_test)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train_log': y_train_log,
        'y_test_log': y_test_log,
        'y_train_orig': y_train_orig,
        'y_test_orig': y_test_orig,
        'scaler': scaler,
        'selector': selector,
        'encoders': label_encoders
    }

def train_and_evaluate_original(data_dict: Dict[str, Any]) -> Dict[str, float]:
    """Treina modelo original."""
    print("\n🤖 Treinando modelo ORIGINAL...")
    
    # Modelo otimizado
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    # Treinar
    start_time = time.time()
    model.fit(data_dict['X_train'], data_dict['y_train'])
    training_time = time.time() - start_time
    
    # Predições
    y_pred = model.predict(data_dict['X_test'])
    
    # Métricas
    mae = mean_absolute_error(data_dict['y_test'], y_pred)
    mse = mean_squared_error(data_dict['y_test'], y_pred)
    r2 = r2_score(data_dict['y_test'], y_pred)
    mae_percentage = (mae / data_dict['y_test'].mean()) * 100
    
    print(f"   ⏱️ Tempo: {training_time:.2f}s")
    print(f"   📈 MAE: ${mae:,.2f} ({mae_percentage:.1f}%)")
    print(f"   📈 MSE: {mse:,.2f}")
    print(f"   📈 R²: {r2:.4f}")
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'mae_percentage': mae_percentage,
        'training_time': training_time,
        'model': model
    }

def train_and_evaluate_log(data_dict: Dict[str, Any]) -> Dict[str, float]:
    """Treina modelo com transformação LOG."""
    print("\n🔄 Treinando modelo LOG TRANSFORM...")
    
    # Modelo otimizado para log
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    # Treinar na escala LOG
    start_time = time.time()
    model.fit(data_dict['X_train'], data_dict['y_train_log'])
    training_time = time.time() - start_time
    
    # Predições na escala LOG
    y_pred_log = model.predict(data_dict['X_test'])
    
    # Converter de volta para escala original
    y_pred_orig = np.expm1(y_pred_log)  # exp(x) - 1
    
    # Métricas na escala original
    mae = mean_absolute_error(data_dict['y_test_orig'], y_pred_orig)
    mse = mean_squared_error(data_dict['y_test_orig'], y_pred_orig)
    r2 = r2_score(data_dict['y_test_orig'], y_pred_orig)
    mae_percentage = (mae / data_dict['y_test_orig'].mean()) * 100
    
    # Validação cruzada na escala log
    cv_scores = cross_val_score(
        model, data_dict['X_train'], data_dict['y_train_log'], 
        cv=5, scoring='r2'
    )
    
    print(f"   ⏱️ Tempo: {training_time:.2f}s")
    print(f"   📈 MAE: ${mae:,.2f} ({mae_percentage:.1f}%)")
    print(f"   📈 MSE: {mse:,.2f}")
    print(f"   📈 R²: {r2:.4f}")
    print(f"   📊 CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'mae_percentage': mae_percentage,
        'training_time': training_time,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'model': model
    }

def save_best_model(model_dict: Dict[str, Any], preprocessing_dict: Dict[str, Any]):
    """Salva o melhor modelo."""
    print("\n💾 Salvando modelo otimizado...")
    
    # Criar diretório
    Path('models').mkdir(exist_ok=True)
    
    # Salvar modelo
    joblib.dump(model_dict['model'], 'models/optimized_log_model.pkl')
    joblib.dump(preprocessing_dict, 'models/optimized_preprocessor.pkl')
    
    print("✅ Modelo salvo em models/optimized_log_model.pkl")
    print("✅ Preprocessador salvo em models/optimized_preprocessor.pkl")

def test_predictions(model_dict: Dict[str, Any], preprocessing_dict: Dict[str, Any]):
    """Testa predições com exemplos."""
    print("\n🎯 Testando predições otimizadas...")
    
    # Exemplos
    examples = [
        {'age': 25, 'sex': 'male', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'southwest'},
        {'age': 45, 'sex': 'female', 'bmi': 28.0, 'children': 2, 'smoker': 'yes', 'region': 'northeast'},
        {'age': 60, 'sex': 'male', 'bmi': 35.0, 'children': 1, 'smoker': 'yes', 'region': 'southeast'}
    ]
    
    for i, example in enumerate(examples, 1):
        # Criar features
        df = pd.DataFrame([example])
        df = create_optimized_features(df)
        df = df.drop('charges', axis=1)
        
        # Encoding
        for col in ['sex', 'smoker', 'region']:
            df[col] = preprocessing_dict['encoders'][col].transform(df[col])
        
        # Seleção e scaling
        X_selected = preprocessing_dict['selector'].transform(df)
        X_scaled = preprocessing_dict['scaler'].transform(X_selected)
        
        # Predição em log e conversão
        y_pred_log = model_dict['model'].predict(X_scaled)[0]
        y_pred_orig = np.expm1(y_pred_log)
        
        print(f"   Exemplo {i}: ${y_pred_orig:,.2f}")

def main():
    """Executa otimização completa."""
    print("🚀 OTIMIZAÇÃO COMPLETA DO SISTEMA DE PREDIÇÃO")
    print("=" * 70)
    
    # 1. Carregar dados
    data = load_and_clean_data()
    
    # 2. Testar abordagem original
    data_orig = preprocess_data_original(data)
    results_orig = train_and_evaluate_original(data_orig)
    
    # 3. Testar abordagem com log
    data_log = preprocess_data_log(data)
    results_log = train_and_evaluate_log(data_log)
    
    # 4. Comparar resultados
    print("\n📊 COMPARAÇÃO FINAL")
    print("=" * 50)
    
    mae_improvement = ((results_orig['mae'] - results_log['mae']) / results_orig['mae']) * 100
    mse_improvement = ((results_orig['mse'] - results_log['mse']) / results_orig['mse']) * 100
    r2_improvement = ((results_log['r2'] - results_orig['r2']) / results_orig['r2']) * 100
    
    print(f"📈 MAE: ${results_orig['mae']:,.2f} → ${results_log['mae']:,.2f} ({mae_improvement:+.1f}%)")
    print(f"📈 MSE: {results_orig['mse']:,.2f} → {results_log['mse']:,.2f} ({mse_improvement:+.1f}%)")
    print(f"📈 R²:  {results_orig['r2']:.4f} → {results_log['r2']:.4f} ({r2_improvement:+.1f}%)")
    
    # 5. Conclusão
    print(f"\n🎯 CONCLUSÃO:")
    if results_log['mae'] < results_orig['mae'] and results_log['mse'] < results_orig['mse']:
        print("✅ TRANSFORMAÇÃO LOG É SUPERIOR!")
        
        if mae_improvement > 15 and mse_improvement > 20:
            print("🚀 MELHORIA SIGNIFICATIVA OBTIDA!")
        
        # Avaliação das métricas finais
        if results_log['mae_percentage'] < 15:
            print("🏆 EXCELENTE: MAE < 15% da média")
        elif results_log['mae_percentage'] < 25:
            print("✅ BOM: MAE < 25% da média")
        
        # Salvar melhor modelo
        save_best_model(results_log, data_log)
        test_predictions(results_log, data_log)
        
    else:
        print("⚠️ Resultados inconclusivos")
    
    print(f"\n📋 RECOMENDAÇÕES FINAIS:")
    print(f"   1. ✅ Usar transformação logarítmica na target")
    print(f"   2. ✅ Aplicar seleção de features (k=10)")
    print(f"   3. ✅ Normalizar todas as features")
    print(f"   4. ✅ Usar validação cruzada")
    print(f"   5. ✅ Monitorar MAE% (atual: {results_log['mae_percentage']:.1f}%)")

if __name__ == "__main__":
    main() 