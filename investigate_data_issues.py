#!/usr/bin/env python3
"""
Investigação dos problemas de escala e transformação dos dados.
FOCO: Transformação logarítmica, escalas, feature engineering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from insurance_prediction.data.loader import load_insurance_data
from insurance_prediction.data.preprocessor import preprocess_insurance_data
from insurance_prediction.models.trainer import GradientBoostingTrainer

def analyze_target_distribution():
    """Analisa a distribuição da variável target."""
    print("🔍 ANÁLISE DA DISTRIBUIÇÃO DA VARIÁVEL TARGET")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    charges = data['charges']
    
    # Estatísticas básicas
    print(f"📊 ESTATÍSTICAS BÁSICAS:")
    print(f"   Média: ${charges.mean():,.2f}")
    print(f"   Mediana: ${charges.median():,.2f}")
    print(f"   Desvio Padrão: ${charges.std():,.2f}")
    print(f"   Coef. Variação: {charges.std()/charges.mean():.2f}")
    print(f"   Mín: ${charges.min():,.2f}")
    print(f"   Máx: ${charges.max():,.2f}")
    print(f"   Amplitude: ${charges.max() - charges.min():,.2f}")
    
    # Análise de assimetria
    skewness = charges.skew()
    print(f"\n📈 ASSIMETRIA (Skewness): {skewness:.2f}")
    if skewness > 1:
        print("   ⚠️ ALTA ASSIMETRIA POSITIVA - PRECISA TRANSFORMAÇÃO LOG!")
    elif skewness > 0.5:
        print("   ⚠️ ASSIMETRIA MODERADA - RECOMENDA TRANSFORMAÇÃO LOG")
    
    # Análise por fumante (principal driver)
    print(f"\n🚬 ANÁLISE POR FUMANTE:")
    smoker_stats = data.groupby('smoker')['charges'].agg(['mean', 'std', 'min', 'max', 'count'])
    for smoker, stats in smoker_stats.iterrows():
        print(f"   {smoker}: Média=${stats['mean']:,.2f}, Std=${stats['std']:,.2f}, Range=${stats['min']:,.2f}-${stats['max']:,.2f}")
    
    ratio = smoker_stats.loc['yes', 'mean'] / smoker_stats.loc['no', 'mean']
    print(f"   📊 Ratio Fumante/Não-fumante: {ratio:.2f}x")
    
    return data

def test_log_transformation():
    """Testa transformação logarítmica."""
    print("\n🔄 TESTANDO TRANSFORMAÇÃO LOGARÍTMICA")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    charges = data['charges']
    
    # Transformação log
    log_charges = np.log1p(charges)  # log(1+x) para evitar log(0)
    
    print(f"📊 COMPARAÇÃO ORIGINAL vs LOG:")
    print(f"   Original - Skewness: {charges.skew():.3f}")
    print(f"   Log      - Skewness: {log_charges.skew():.3f}")
    print(f"   Original - Std/Mean: {charges.std()/charges.mean():.3f}")
    print(f"   Log      - Std/Mean: {log_charges.std()/log_charges.mean():.3f}")
    
    # Retornar dados para treino
    return data, log_charges

def compare_models_with_without_log():
    """Compara modelos com e sem transformação log."""
    print("\n🤖 COMPARANDO MODELOS: ORIGINAL vs LOG-TRANSFORMED")
    print("=" * 60)
    
    data, log_charges = test_log_transformation()
    
    # Preparar features
    X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].copy()
    
    # Encoding simples para comparação rápida
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Divisão treino/teste
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X_encoded, data['charges'], test_size=0.2, random_state=42
    )
    
    _, _, y_train_log, y_test_log = train_test_split(
        X_encoded, log_charges, test_size=0.2, random_state=42
    )
    
    # Modelo 1: Original
    print("\n📈 MODELO 1: TARGET ORIGINAL")
    model_orig = GradientBoostingRegressor(random_state=42, n_estimators=100)
    model_orig.fit(X_train, y_train_orig)
    y_pred_orig = model_orig.predict(X_test)
    
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    
    print(f"   MAE: ${mae_orig:,.2f}")
    print(f"   MSE: {mse_orig:,.2f}")
    print(f"   RMSE: ${np.sqrt(mse_orig):,.2f}")
    print(f"   R²: {r2_orig:.4f}")
    print(f"   MAE/Média: {mae_orig/y_test_orig.mean()*100:.1f}%")
    
    # Modelo 2: Log-transformed
    print("\n📈 MODELO 2: TARGET LOG-TRANSFORMED")
    model_log = GradientBoostingRegressor(random_state=42, n_estimators=100)
    model_log.fit(X_train, y_train_log)
    y_pred_log = model_log.predict(X_test)
    
    # Converter predições de volta para escala original
    y_pred_log_orig = np.expm1(y_pred_log)  # exp(x) - 1
    y_test_log_orig = np.expm1(y_test_log)
    
    mae_log = mean_absolute_error(y_test_log_orig, y_pred_log_orig)
    mse_log = mean_squared_error(y_test_log_orig, y_pred_log_orig)
    r2_log = r2_score(y_test_log_orig, y_pred_log_orig)
    
    print(f"   MAE: ${mae_log:,.2f}")
    print(f"   MSE: {mse_log:,.2f}")
    print(f"   RMSE: ${np.sqrt(mse_log):,.2f}")
    print(f"   R²: {r2_log:.4f}")
    print(f"   MAE/Média: {mae_log/y_test_log_orig.mean()*100:.1f}%")
    
    # Comparação
    print(f"\n🎯 MELHORIA COM LOG-TRANSFORM:")
    print(f"   MAE: {((mae_orig - mae_log)/mae_orig)*100:+.1f}%")
    print(f"   MSE: {((mse_orig - mse_log)/mse_orig)*100:+.1f}%")
    print(f"   R²: {((r2_log - r2_orig)/r2_orig)*100:+.1f}%")
    
    return model_orig, model_log, (mae_orig, mse_orig, r2_orig), (mae_log, mse_log, r2_log)

def analyze_current_preprocessing():
    """Analisa o preprocessamento atual."""
    print("\n🔧 ANÁLISE DO PREPROCESSAMENTO ATUAL")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    processed_data = preprocess_insurance_data(data)
    
    print(f"📊 FEATURES CRIADAS:")
    print(f"   Features originais: 6")
    print(f"   Features finais: {len(processed_data['feature_names'])}")
    print(f"   Features criadas: {len(processed_data['feature_names']) - 6}")
    
    print(f"\n📝 NOMES DAS FEATURES:")
    for i, feature in enumerate(processed_data['feature_names']):
        print(f"   {i+1:2d}. {feature}")
    
    # Analisar escalas das features
    X_train = processed_data['X_train']
    print(f"\n📏 ESCALAS DAS FEATURES:")
    print(f"   Mín global: {X_train.min().min():.3f}")
    print(f"   Máx global: {X_train.max().max():.3f}")
    print(f"   Std médio: {X_train.std().mean():.3f}")
    
    # Features com escalas muito diferentes
    feature_stats = pd.DataFrame({
        'feature': processed_data['feature_names'],
        'min': X_train.min(),
        'max': X_train.max(),
        'std': X_train.std(),
        'range': X_train.max() - X_train.min()
    })
    
    print(f"\n⚠️ FEATURES COM MAIOR VARIAÇÃO:")
    top_range = feature_stats.nlargest(5, 'range')
    for _, row in top_range.iterrows():
        print(f"   {row['feature']}: Range={row['range']:.2f}, Std={row['std']:.2f}")

def check_outliers_impact():
    """Verifica impacto dos outliers."""
    print("\n🎯 ANÁLISE DE OUTLIERS")
    print("=" * 60)
    
    data, _ = load_insurance_data()
    charges = data['charges']
    
    # Outliers por IQR
    Q1 = charges.quantile(0.25)
    Q3 = charges.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (charges < lower_bound) | (charges > upper_bound)
    outlier_count = outliers.sum()
    
    print(f"📊 OUTLIERS (IQR 1.5):")
    print(f"   Total: {outlier_count} ({outlier_count/len(charges)*100:.1f}%)")
    print(f"   Limite inferior: ${lower_bound:,.2f}")
    print(f"   Limite superior: ${upper_bound:,.2f}")
    
    # Impacto na média
    charges_no_outliers = charges[~outliers]
    print(f"\n💰 IMPACTO NA MÉDIA:")
    print(f"   Com outliers: ${charges.mean():,.2f}")
    print(f"   Sem outliers: ${charges_no_outliers.mean():,.2f}")
    print(f"   Diferença: ${charges.mean() - charges_no_outliers.mean():,.2f}")
    
    # Outliers extremos
    extreme_outliers = charges > (Q3 + 3 * IQR)
    print(f"\n🚨 OUTLIERS EXTREMOS (Q3 + 3*IQR):")
    print(f"   Total: {extreme_outliers.sum()} valores")
    if extreme_outliers.sum() > 0:
        print(f"   Valores: {charges[extreme_outliers].tolist()}")

def main():
    """Executa toda a investigação."""
    print("🔍 INVESTIGAÇÃO COMPLETA DOS PROBLEMAS DE DADOS")
    print("=" * 80)
    
    # Análises
    data = analyze_target_distribution()
    test_log_transformation()
    model_orig, model_log, metrics_orig, metrics_log = compare_models_with_without_log()
    analyze_current_preprocessing()
    check_outliers_impact()
    
    # Conclusões
    print("\n🎯 CONCLUSÕES E RECOMENDAÇÕES")
    print("=" * 60)
    
    mae_orig, mse_orig, r2_orig = metrics_orig
    mae_log, mse_log, r2_log = metrics_log
    
    if mae_log < mae_orig:
        print("✅ TRANSFORMAÇÃO LOG MELHORA AS MÉTRICAS!")
        print(f"   Redução MAE: ${mae_orig - mae_log:,.2f}")
        print(f"   Redução MSE: {mse_orig - mse_log:,.2f}")
    
    print(f"\n📋 AÇÕES RECOMENDADAS:")
    print(f"   1. ✅ Implementar transformação logarítmica na target")
    print(f"   2. ✅ Revisar feature engineering (reduzir features redundantes)")
    print(f"   3. ✅ Aplicar normalização nas features")
    print(f"   4. ✅ Tratamento mais agressivo de outliers")
    print(f"   5. ✅ Usar validação cruzada com log-transform")

if __name__ == "__main__":
    main() 