#!/usr/bin/env python3
"""
Comparação de Performance: Modelo Local vs Deploy
Métricas: R², MAE, MSE, RMSE, MAPE, MBE
USANDO DADOS REAIS DO INSURANCE.CSV
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path('.') / 'src'))

def calculate_metrics(y_true, y_pred):
    """Calcula todas as métricas de performance."""
    
    # Converter para numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # R² (R-squared)
    r2 = r2_score(y_true, y_pred)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # MBE (Mean Bias Error)
    mbe = np.mean(y_pred - y_true)
    
    return {
        'R²': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MBE': mbe
    }

def load_real_data():
    """Carrega os dados reais do insurance.csv"""
    
    print("📂 Carregando dados reais do insurance.csv...")
    
    # Tentar diferentes caminhos
    possible_paths = [
        "data/insurance.csv",
        "data/raw/insurance.csv", 
        "../data/insurance.csv"
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"✅ Dados carregados de: {path}")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Colunas: {list(df.columns)}")
            return df
        except Exception as e:
            continue
    
    raise FileNotFoundError("❌ Arquivo insurance.csv não encontrado!")

def prepare_test_data(df, test_size=0.3):
    """Prepara dados de teste a partir do dataset real."""
    
    print(f"🔄 Preparando dados de teste (tamanho do teste: {test_size})")
    
    # Separar features e target
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    target = 'charges'
    
    X = df[features]
    y = df[target]
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=df['smoker']
    )
    
    print(f"📈 Dados de treino: {len(X_train)} amostras")
    print(f"📊 Dados de teste: {len(X_test)} amostras")
    
    # Converter para formato adequado
    test_data = []
    for idx, row in X_test.iterrows():
        test_data.append({
            'age': int(row['age']),
            'sex': row['sex'],
            'bmi': float(row['bmi']),
            'children': int(row['children']),
            'smoker': row['smoker'],
            'region': row['region'],
            'true_charges': float(y_test.iloc[X_test.index.get_loc(idx)])
        })
    
    return test_data

def test_local_model(test_data):
    """Testa o modelo local."""
    
    print("🏠 Testando Modelo Local...")
    
    try:
        from insurance_prediction.models.predictor import predict_insurance_premium
        
        predictions = []
        true_values = []
        errors = 0
        
        for i, data in enumerate(test_data):
            try:
                result = predict_insurance_premium(
                    age=data['age'],
                    sex=data['sex'],
                    bmi=data['bmi'],
                    children=data['children'],
                    smoker=data['smoker'],
                    region=data['region']
                )
                
                predictions.append(result['predicted_premium'])
                true_values.append(data['true_charges'])
                
            except Exception as e:
                errors += 1
                if errors <= 3:  # Mostra apenas os primeiros 3 erros
                    print(f"Erro na predição local {i+1}: {e}")
                continue
        
        print(f"✅ Modelo Local: {len(predictions)}/{len(test_data)} predições (erros: {errors})")
        return true_values, predictions
        
    except Exception as e:
        print(f"❌ Erro no modelo local: {e}")
        return [], []

def test_deploy_model(test_data):
    """Testa o modelo de deploy."""
    
    print("☁️ Testando Modelo Deploy...")
    
    try:
        from deploy.model_utils import load_model, predict_premium
        
        # Carregar modelo uma vez
        model_data = load_model()
        print(f"🔧 Modelo carregado: {type(model_data['model']).__name__}")
        print(f"🎯 Features: {model_data.get('feature_names', 'N/A')}")
        
        predictions = []
        true_values = []
        errors = 0
        
        for i, data in enumerate(test_data):
            try:
                input_data = {
                    'age': data['age'],
                    'sex': data['sex'],
                    'bmi': data['bmi'],
                    'children': data['children'],
                    'smoker': data['smoker'],
                    'region': data['region']
                }
                
                result = predict_premium(input_data, model_data)
                
                if result['success']:
                    predictions.append(result['predicted_premium'])
                    true_values.append(data['true_charges'])
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"Erro na predição deploy {i+1}: {result.get('error', 'Desconhecido')}")
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"Erro na predição deploy {i+1}: {e}")
                continue
        
        print(f"✅ Modelo Deploy: {len(predictions)}/{len(test_data)} predições (erros: {errors})")
        return true_values, predictions
        
    except Exception as e:
        print(f"❌ Erro no modelo deploy: {e}")
        return [], []

def analyze_data_distribution(df):
    """Analisa a distribuição dos dados reais."""
    
    print("\n📊 ANÁLISE DOS DADOS REAIS:")
    print("=" * 50)
    
    print(f"Total de amostras: {len(df)}")
    print(f"Charges médio: ${df['charges'].mean():,.2f}")
    print(f"Charges mediano: ${df['charges'].median():,.2f}")
    print(f"Charges mín: ${df['charges'].min():,.2f}")
    print(f"Charges máx: ${df['charges'].max():,.2f}")
    print(f"Desvio padrão: ${df['charges'].std():,.2f}")
    
    print("\n🚭 Distribuição por fumante:")
    smoker_stats = df.groupby('smoker')['charges'].agg(['count', 'mean', 'std'])
    print(smoker_stats)
    
    print("\n👥 Distribuição por sexo:")
    sex_stats = df.groupby('sex')['charges'].agg(['count', 'mean', 'std'])
    print(sex_stats)
    
    print("\n🗺️ Distribuição por região:")
    region_stats = df.groupby('region')['charges'].agg(['count', 'mean', 'std'])
    print(region_stats)

def compare_models():
    """Compara a performance dos dois modelos usando dados REAIS."""
    
    print("🔍 COMPARAÇÃO DE PERFORMANCE: LOCAL vs DEPLOY")
    print("📄 USANDO DADOS REAIS DO INSURANCE.CSV")
    print("=" * 60)
    
    # Carregar dados reais
    df = load_real_data()
    
    # Analisar distribuição dos dados
    analyze_data_distribution(df)
    
    # Preparar dados de teste
    print("\n" + "="*50)
    test_data = prepare_test_data(df, test_size=0.2)  # 20% para teste
    
    # Testar modelo local
    print("\n" + "="*40)
    true_local, pred_local = test_local_model(test_data)
    
    # Testar modelo deploy
    print("\n" + "="*40)
    true_deploy, pred_deploy = test_deploy_model(test_data)
    
    # Calcular métricas se ambos funcionaram
    if len(pred_local) > 0 and len(pred_deploy) > 0:
        print("\n" + "="*60)
        print("📈 RESULTADOS DA COMPARAÇÃO COM DADOS REAIS")
        print("="*60)
        
        # Calcular métricas para cada modelo
        metrics_local = calculate_metrics(true_local, pred_local)
        metrics_deploy = calculate_metrics(true_deploy, pred_deploy)
        
        # Exibir resultados
        print(f"\n🏠 MODELO LOCAL ({len(pred_local)} amostras):")
        print("-" * 40)
        for metric, value in metrics_local.items():
            if metric == 'R²':
                print(f"{metric:>8}: {value:>12.4f}")
            elif metric in ['MAPE']:
                print(f"{metric:>8}: {value:>11.2f}%")
            else:
                print(f"{metric:>8}: ${value:>11,.2f}")
        
        print(f"\n☁️ MODELO DEPLOY ({len(pred_deploy)} amostras):")
        print("-" * 40)
        for metric, value in metrics_deploy.items():
            if metric == 'R²':
                print(f"{metric:>8}: {value:>12.4f}")
            elif metric in ['MAPE']:
                print(f"{metric:>8}: {value:>11.2f}%")
            else:
                print(f"{metric:>8}: ${value:>11,.2f}")
        
        # Comparação direta
        print(f"\n🏆 COMPARAÇÃO DIRETA:")
        print("-" * 60)
        print(f"{'Métrica':>10} | {'Local':>12} | {'Deploy':>12} | {'Melhor':>12}")
        print("-" * 60)
        
        for metric in metrics_local.keys():
            local_val = metrics_local[metric]
            deploy_val = metrics_deploy[metric]
            
            # Determinar qual é melhor (maior R², menor para outras)
            if metric == 'R²':
                better = 'LOCAL' if local_val > deploy_val else 'DEPLOY'
                if abs(local_val - deploy_val) < 0.001:
                    better = 'EMPATE'
            else:
                better = 'LOCAL' if local_val < deploy_val else 'DEPLOY'
                if abs(local_val - deploy_val) < 1:
                    better = 'EMPATE'
            
            if metric == 'R²':
                print(f"{metric:>10} | {local_val:>12.4f} | {deploy_val:>12.4f} | {better:>12}")
            elif metric == 'MAPE':
                print(f"{metric:>10} | {local_val:>11.2f}% | {deploy_val:>11.2f}% | {better:>12}")
            else:
                print(f"{metric:>10} | ${local_val:>11,.0f} | ${deploy_val:>11,.0f} | {better:>12}")
        
        # Score geral
        print("\n🎯 SCORE GERAL:")
        print("-" * 40)
        
        local_wins = 0
        deploy_wins = 0
        ties = 0
        
        for metric in metrics_local.keys():
            if metric == 'R²':
                if metrics_local[metric] > metrics_deploy[metric] + 0.001:
                    local_wins += 1
                elif metrics_deploy[metric] > metrics_local[metric] + 0.001:
                    deploy_wins += 1
                else:
                    ties += 1
            else:
                if metrics_local[metric] < metrics_deploy[metric] - 1:
                    local_wins += 1
                elif metrics_deploy[metric] < metrics_local[metric] - 1:
                    deploy_wins += 1
                else:
                    ties += 1
        
        print(f"Local ganha em: {local_wins}/6 métricas")
        print(f"Deploy ganha em: {deploy_wins}/6 métricas")
        print(f"Empates: {ties}/6 métricas")
        
        if local_wins > deploy_wins:
            print("🏆 VENCEDOR: MODELO LOCAL")
        elif deploy_wins > local_wins:
            print("🏆 VENCEDOR: MODELO DEPLOY")
        else:
            print("🤝 EMPATE")
        
        # Análise detalhada
        print(f"\n🔍 ANÁLISE DETALHADA:")
        print("-" * 40)
        print(f"Diferença R²: {abs(metrics_local['R²'] - metrics_deploy['R²']):.4f}")
        print(f"Diferença MAE: ${abs(metrics_local['MAE'] - metrics_deploy['MAE']):,.2f}")
        print(f"Diferença RMSE: ${abs(metrics_local['RMSE'] - metrics_deploy['RMSE']):,.2f}")
        
        # Recomendações
        if deploy_wins > local_wins:
            print("\n💡 RECOMENDAÇÃO: Use o modelo DEPLOY em produção")
        elif local_wins > deploy_wins:
            print("\n💡 RECOMENDAÇÃO: Considere atualizar o modelo DEPLOY com o modelo LOCAL")
        else:
            print("\n💡 RECOMENDAÇÃO: Ambos modelos têm performance similar")
        
    else:
        print("❌ Não foi possível comparar - um ou ambos modelos falharam")
        if len(pred_local) == 0:
            print("   - Modelo LOCAL falhou")
        if len(pred_deploy) == 0:
            print("   - Modelo DEPLOY falhou")

if __name__ == "__main__":
    compare_models() 