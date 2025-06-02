#!/usr/bin/env python3
"""
Script para treinar todos os modelos incluindo XGBoost e identificar o melhor modelo
baseado em TODAS as métricas do notebook FIAP_Tech_Challenge_01.ipynb.

Métricas utilizadas: MAE, MSE, RMSE, MAPE, R², Adjusted R², MBE
Sistema de pontuação robusto que considera todas as métricas e estabilidade CV.
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
import logging
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# XGBoost
from xgboost import XGBRegressor

# LightGBM (opcional)
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM não está instalado.")

# Imports do projeto
from src.data_loader import load_insurance_data
from src.preprocessing import preprocess_insurance_data

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mean_bias_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Bias Deviation (MBE)."""
    return np.mean(y_pred - y_true)

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Calcula R² ajustado."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    """Calcula todas as métricas solicitadas."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Converter para %
        'R²': r2_score(y_true, y_pred),
        'Adjusted_R²': adjusted_r2_score(y_true, y_pred, n_features),
        'MBE': mean_bias_deviation(y_true, y_pred)
    }

def calculate_composite_score(results: dict) -> dict:
    """
    Calcula um score composto baseado em todas as métricas para determinar o melhor modelo.
    
    Args:
        results: Dictionary com resultados de todos os modelos
        
    Returns:
        Dictionary com scores compostos e ranking
    """
    # Filtrar modelos válidos
    valid_results = {name: result for name, result in results.items() 
                    if result.get('test_metrics', {}).get('R²', -np.inf) > -np.inf}
    
    if not valid_results:
        return {}
    
    # Extrair métricas para normalização
    metrics_data = []
    model_names = []
    
    for name, result in valid_results.items():
        metrics = result['test_metrics']
        cv_data = {
            'r2_mean': result['cv_r2_mean'],
            'r2_std': result['cv_r2_std'],
            'mae_mean': result['cv_mae_mean'],
            'mae_std': result['cv_mae_std']
        }
        
        row = {
            'model': name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R²': metrics['R²'],
            'Adjusted_R²': metrics['Adjusted_R²'],
            'MBE_abs': abs(metrics['MBE']),  # Valor absoluto do viés
            'CV_R²_std': cv_data['r2_std'],  # Estabilidade (menor é melhor)
            'CV_MAE_std': cv_data['mae_std']  # Estabilidade (menor é melhor)
        }
        metrics_data.append(row)
        model_names.append(name)
    
    df = pd.DataFrame(metrics_data)
    
    # Definir pesos para cada métrica (soma = 1.0)
    weights = {
    'R²': 0.15,          # Menor foco no R² simples
    'Adjusted_R²': 0.20,  # Maior foco no R² ajustado para poder explicativo robusto
    'MAE': 0.15,         # Erro absoluto médio, balanceado com RMSE
    'RMSE': 0.25,        # Penaliza mais erros grandes, importante para custos altos
    'MAPE': 0.05,        # Erro percentual, útil se a escala varia muito, mas com cautela
    'MBE_abs': 0.10,     # Importante para evitar viés sistemático (sub/superestimação)
    'CV_R²_std': 0.05,   # Estabilidade do R²
    'CV_MAE_std': 0.05   # Estabilidade do MAE (ou poderia ser CV_RMSE_std)
}
    
    # Normalizar métricas (0-1 scale)
    normalized_scores = {}
    
    # Métricas para MAXIMIZAR (maior é melhor)
    maximize_metrics = ['R²', 'Adjusted_R²']
    for metric in maximize_metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            normalized_scores[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_scores[metric] = pd.Series([1.0] * len(df))
    
    # Métricas para MINIMIZAR (menor é melhor)
    minimize_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MBE_abs', 'CV_R²_std', 'CV_MAE_std']
    for metric in minimize_metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            # Inverter: menor valor = maior score
            normalized_scores[metric] = 1 - ((df[metric] - min_val) / (max_val - min_val))
        else:
            normalized_scores[metric] = pd.Series([1.0] * len(df))
    
    # Calcular score composto ponderado
    composite_scores = pd.Series([0.0] * len(df))
    score_breakdown = {}
    
    for metric, weight in weights.items():
        weighted_score = normalized_scores[metric] * weight
        composite_scores += weighted_score
        score_breakdown[metric] = normalized_scores[metric]
    
    # Criar resultado final
    final_results = {}
    for i, model_name in enumerate(model_names):
        final_results[model_name] = {
            'composite_score': composite_scores.iloc[i],
            'normalized_scores': {metric: score_breakdown[metric].iloc[i] for metric in weights.keys()},
            'raw_metrics': valid_results[model_name]['test_metrics'],
            'cv_metrics': {
                'r2_mean': valid_results[model_name]['cv_r2_mean'],
                'r2_std': valid_results[model_name]['cv_r2_std'],
                'mae_mean': valid_results[model_name]['cv_mae_mean'],
                'mae_std': valid_results[model_name]['cv_mae_std']
            },
            'rank': 0  # Será preenchido depois
        }
    
    # Ordenar por score composto e atribuir ranks
    sorted_models = sorted(final_results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        final_results[model_name]['rank'] = rank
    
    return final_results

def train_and_evaluate_model(model_name: str, model, X_train, y_train, X_test, y_test, n_features: int):
    """Treina e avalia um modelo individual."""
    logger.info(f"Treinando {model_name}...")
    
    start_time = datetime.now()
    
    try:
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        metrics = calculate_all_metrics(y_test, y_pred_test, n_features)
        
        # Validação cruzada para R²
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        # Compilar resultados
        result = {
            'model': model,
            'model_name': model_name,
            'test_metrics': metrics,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_mae_mean': cv_mae_scores.mean(),
            'cv_mae_std': cv_mae_scores.std(),
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Log resultados
        logger.info(f"✅ {model_name} - Resultados:")
        logger.info(f"   Test R²: {metrics['R²']:.4f}")
        logger.info(f"   Test Adjusted R²: {metrics['Adjusted_R²']:.4f}")
        logger.info(f"   Test MAE: {metrics['MAE']:.2f}")
        logger.info(f"   Test MSE: {metrics['MSE']:.2f}")
        logger.info(f"   Test RMSE: {metrics['RMSE']:.2f}")
        logger.info(f"   Test MAPE: {metrics['MAPE']:.2f}%")
        logger.info(f"   Test MBE: {metrics['MBE']:.2f}")
        logger.info(f"   CV R² Mean: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro ao treinar {model_name}: {e}")
        return {
            'model': None,
            'model_name': model_name,
            'error': str(e),
            'test_metrics': {
                'R²': -np.inf,
                'MAE': np.inf,
                'MSE': np.inf,
                'RMSE': np.inf,
                'MAPE': np.inf,
                'Adjusted_R²': -np.inf,
                'MBE': np.inf
            }
        }

def print_detailed_ranking(scoring_results: dict):
    """Imprime ranking detalhado com todas as métricas."""
    print("\n📊 RANKING DETALHADO DOS MODELOS (Baseado em TODAS as Métricas):")
    print("=" * 100)
    
    # Ordenar por composite score
    sorted_models = sorted(scoring_results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
    
    # Header
    print(f"{'Rank':<4} {'Model':<20} {'Score':<8} {'R²':<8} {'Adj R²':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<7} {'|MBE|':<7}")
    print("-" * 100)
    
    for model_name, data in sorted_models:
        metrics = data['raw_metrics']
        rank = data['rank']
        score = data['composite_score']
        
        print(f"{rank:<4} {model_name:<20} {score:<8.4f} {metrics['R²']:<8.4f} {metrics['Adjusted_R²']:<8.4f} "
              f"{metrics['MAE']:<8.2f} {metrics['RMSE']:<8.2f} {metrics['MAPE']:<7.2f} {abs(metrics['MBE']):<7.2f}")
    
    print("=" * 100)
    print("Score = Score Composto baseado em TODAS as métricas ponderadas")
    print("Pesos: R²(25%), Adj R²(15%), MAE(20%), RMSE(15%), MAPE(10%), |MBE|(5%), Estabilidade CV(10%)")

def main():
    """Função principal."""
    print("🚀 Iniciando treinamento de modelos com XGBoost...")
    print("🎯 Sistema de avaliação ROBUSTO baseado em TODAS as métricas!")
    print("=" * 70)
    
    try:
        # 1. Carregar dados
        print("📊 Carregando dados...")
        data, _ = load_insurance_data()
        print(f"✅ Dados carregados: {data.shape}")
        
        # 2. Preprocessar dados
        print("🔄 Preprocessando dados...")
        processed_data = preprocess_insurance_data(
            data,
            remove_outliers=True,
            apply_transformations=True,
            create_polynomial=True,
            create_interactions=True,
            feature_selection=True
        )
        
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        n_features = X_train.shape[1]
        
        print(f"✅ Preprocessamento concluído:")
        print(f"   Treino: {X_train.shape}")
        print(f"   Teste: {X_test.shape}")
        print(f"   Features: {n_features}")
        
        # 3. Definir modelos (baseado no notebook)
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'ElasticNet Regression': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        }
        
        # Adicionar LightGBM se disponível
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        
        # 4. Treinar todos os modelos
        print(f"\n🎯 Treinando {len(models)} modelos...")
        print("=" * 70)
        
        results = {}
        for model_name, model in models.items():
            result = train_and_evaluate_model(
                model_name, model, X_train, y_train, X_test, y_test, n_features
            )
            results[model_name] = result
            print("-" * 50)
        
        # 5. Calcular scores compostos baseados em TODAS as métricas
        print("\n🧮 Calculando scores compostos baseados em TODAS as métricas...")
        scoring_results = calculate_composite_score(results)
        
        # 6. Mostrar ranking detalhado
        print_detailed_ranking(scoring_results)
        
        # 7. Identificar melhor modelo (baseado no score composto)
        if scoring_results:
            # Encontrar modelo com maior score composto
            best_model_name = max(scoring_results.keys(), 
                                key=lambda x: scoring_results[x]['composite_score'])
            
            best_result = results[best_model_name]
            best_metrics = best_result['test_metrics']
            best_score_data = scoring_results[best_model_name]
            
            print("\n" + "=" * 70)
            print(f"🏆 MELHOR MODELO (Score Composto): {best_model_name}")
            print("=" * 70)
            print(f"Score Composto:  {best_score_data['composite_score']:.4f}")
            print(f"Rank:            #{best_score_data['rank']}")
            print("\n📊 Métricas Detalhadas:")
            print(f"R²:              {best_metrics['R²']:.4f}")
            print(f"Adjusted R²:     {best_metrics['Adjusted_R²']:.4f}")
            print(f"MAE:             {best_metrics['MAE']:.2f}")
            print(f"MSE:             {best_metrics['MSE']:.2f}")
            print(f"RMSE:            {best_metrics['RMSE']:.2f}")
            print(f"MAPE:            {best_metrics['MAPE']:.2f}%")
            print(f"MBE:             {best_metrics['MBE']:.2f}")
            print(f"CV R²:           {best_result['cv_r2_mean']:.4f} ± {best_result['cv_r2_std']:.4f}")
            
            print("\n🎯 Scores Normalizados por Categoria:")
            norm_scores = best_score_data['normalized_scores']
            print(f"Performance R²:  {norm_scores['R²']:.3f}")
            print(f"Performance Adj R²: {norm_scores['Adjusted_R²']:.3f}")
            print(f"Erro MAE:        {norm_scores['MAE']:.3f}")
            print(f"Erro RMSE:       {norm_scores['RMSE']:.3f}")
            print(f"Erro MAPE:       {norm_scores['MAPE']:.3f}")
            print(f"Viés (|MBE|):    {norm_scores['MBE_abs']:.3f}")
            print(f"Estabilidade R²: {norm_scores['CV_R²_std']:.3f}")
            print(f"Estabilidade MAE:{norm_scores['CV_MAE_std']:.3f}")
            
            # 8. Salvar melhor modelo
            print("\n💾 Salvando melhor modelo...")
            
            # Criar diretórios
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            artifacts_dir = Path("models/model_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            
            # Salvar modelo
            model_path = models_dir / "best_model.pkl"
            joblib.dump(best_result['model'], model_path)
            print(f"✅ Modelo salvo: {model_path}")
            
            # Salvar metadados extendidos
            metadata = {
                'model_name': best_model_name,
                'model_type': type(best_result['model']).__name__,
                'selection_method': 'composite_score_all_metrics',
                'composite_score': best_score_data['composite_score'],
                'rank': best_score_data['rank'],
                'performance': best_metrics,
                'normalized_scores': best_score_data['normalized_scores'],
                'cv_performance': {
                    'r2_mean': best_result['cv_r2_mean'],
                    'r2_std': best_result['cv_r2_std'],
                    'mae_mean': best_result['cv_mae_mean'],
                    'mae_std': best_result['cv_mae_std']
                },
                'training_date': datetime.now().isoformat(),
                'features_count': n_features,
                'scoring_weights': {
                    'R²': 0.25, 'Adjusted_R²': 0.15, 'MAE': 0.20, 'RMSE': 0.15,
                    'MAPE': 0.10, 'MBE_abs': 0.05, 'CV_R²_std': 0.05, 'CV_MAE_std': 0.05
                }
            }
            
            import json
            metadata_path = models_dir / "best_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadados salvos: {metadata_path}")
            
            # 9. Criar DataFrame de comparação completa
            comparison_data = []
            for name, score_data in scoring_results.items():
                metrics = score_data['raw_metrics']
                cv_data = score_data['cv_metrics']
                
                row = {
                    'Rank': score_data['rank'],
                    'Model': name,
                    'Composite_Score': score_data['composite_score'],
                    'R²': metrics['R²'],
                    'Adjusted_R²': metrics['Adjusted_R²'],
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE (%)': metrics['MAPE'],
                    'MBE': metrics['MBE'],
                    'CV_R²_Mean': cv_data['r2_mean'],
                    'CV_R²_Std': cv_data['r2_std'],
                    'CV_MAE_Mean': cv_data['mae_mean'],
                    'CV_MAE_Std': cv_data['mae_std'],
                    'Training_Time (s)': results[name]['training_time']
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
            
            # Salvar comparação
            comparison_path = models_dir / "model_comparison_complete.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"✅ Comparação completa salva: {comparison_path}")
            
            print("\n🎉 Treinamento concluído com sucesso!")
            print(f"🏆 Melhor modelo: {best_model_name}")
            print(f"🎯 Score composto: {best_score_data['composite_score']:.4f}")
            print(f"📈 R²: {best_metrics['R²']:.4f}")
            print(f"📉 MAE: {best_metrics['MAE']:.2f}")
            
            # Comparar com resultados do notebook para XGBoost
            if best_model_name == 'XGBoost':
                notebook_xgb_metrics = {
                    'R²': 0.8787, 'MAE': 2462.06, 'MSE': 18834571.78,
                    'RMSE': 4339.88, 'MAPE': 30.65
                }
                
                print("\n📋 Comparação XGBoost com resultados do notebook:")
                print("-" * 60)
                print(f"{'Métrica':<10} {'Atual':<12} {'Notebook':<12} {'Diferença':<12}")
                print("-" * 60)
                
                for metric in ['R²', 'MAE', 'RMSE', 'MAPE']:
                    atual = best_metrics[metric] if metric != 'MAPE' else best_metrics['MAPE']
                    notebook = notebook_xgb_metrics[metric]
                    diff = atual - notebook
                    print(f"{metric:<10} {atual:<12.2f} {notebook:<12.2f} {diff:<12.2f}")
            
            return {
                'best_model': best_result['model'],
                'best_model_name': best_model_name,
                'best_metrics': best_metrics,
                'composite_score': best_score_data['composite_score'],
                'all_results': results,
                'scoring_results': scoring_results,
                'comparison_df': comparison_df
            }
        
        else:
            print("❌ Nenhum modelo foi treinado com sucesso!")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🏆 RESULTADO FINAL:")
        print(f"   Melhor modelo: {result['best_model_name']}")
        print(f"   Score composto: {result['composite_score']:.4f}")
        print(f"   R²: {result['best_metrics']['R²']:.4f}")
        print(f"   MAE: {result['best_metrics']['MAE']:.2f}")
        print("\n🎯 Modelo selecionado baseado em TODAS as métricas!")
    else:
        print("❌ Treinamento falhou!") 