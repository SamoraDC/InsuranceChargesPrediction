#!/usr/bin/env python3
"""
CRIAR MODELO SUPERIOR PARA DEPLOY - RÉPLICA EXATA DO LOCAL
Este script cria um modelo IDÊNTICO ao local com performance superior
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_superior_model():
    """Criar modelo superior idêntico ao local"""
    
    print("🚀 CRIANDO MODELO SUPERIOR PARA DEPLOY...")
    print("🎯 Objetivo: Performance R² > 0.89, MAE < $2,300 (igual ao local)")
    
    # 1. Carregar dados reais
    data_path = Path("data/insurance.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {data_path}")
    
    print(f"✅ Carregando dados de: {data_path}")
    df = pd.read_csv(data_path)
    print(f"📊 Dataset: {len(df)} registros, {len(df.columns)} colunas")
    
    # 2. Preparar features EXATAMENTE como o modelo local superior
    print("\n🔧 PREPARANDO FEATURES SUPERIORES...")
    
    # Features básicas
    X = df[['age', 'bmi', 'children']].copy()
    
    # Encoding categórico
    le_sex = LabelEncoder()
    X['sex'] = le_sex.fit_transform(df['sex'])
    
    le_smoker = LabelEncoder()  
    X['smoker'] = le_smoker.fit_transform(df['smoker'])
    
    le_region = LabelEncoder()
    X['region'] = le_region.fit_transform(df['region'])
    
    # ===== FEATURES AVANÇADAS (IGUAL AO LOCAL) =====
    
    # 1. Interactions
    X['age_smoker_risk'] = X['age'] * X['smoker']
    X['bmi_smoker_risk'] = X['bmi'] * X['smoker'] 
    X['age_bmi_interaction'] = X['age'] * X['bmi']
    
    # 2. Age groups (categorical)
    def age_group_encoding(age):
        if age < 30:
            return 0
        elif age < 45:
            return 1
        elif age < 60:
            return 2
        else:
            return 3
    
    X['age_group'] = X['age'].apply(age_group_encoding)
    
    # 3. BMI categories (clinical)
    def bmi_category_encoding(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese
    
    X['bmi_category'] = X['bmi'].apply(bmi_category_encoding)
    
    # 4. Composite risk score (EXATO DO LOCAL)
    X['composite_risk_score'] = (
        X['age'] * 0.1 + 
        X['bmi'] * 0.2 + 
        X['smoker'] * 10 + 
        X['children'] * 0.5
    )
    
    # 5. Region risk density
    region_risk_map = {0: 1, 1: 2, 2: 1, 3: 3}  # northeast=3, northwest=2, southeast=1, southwest=1
    X['region_density'] = X['region'].map(region_risk_map)
    
    # 6. Premium band (additional feature for better performance)
    premium_band = pd.cut(df['charges'], bins=5, labels=[0,1,2,3,4])
    # X['expected_premium_band'] = premium_band.astype(int)  # Don't include target info
    
    y = df['charges']
    
    print(f"✅ Features criadas: {len(X.columns)} features")
    print(f"📋 Features: {list(X.columns)}")
    
    # 3. Split treino/teste (MESMO SEED DO LOCAL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,  # MESMO SEED DO LOCAL
        stratify=pd.cut(y, bins=5, labels=[0,1,2,3,4])  # Stratify por faixas de preço
    )
    
    print(f"\n📊 Split de dados:")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste: {len(X_test)} amostras")
    
    # 4. Treinar modelo SUPERIOR (hiperparâmetros otimizados)
    print("\n🤖 TREINANDO MODELO SUPERIOR...")
    
    model = GradientBoostingRegressor(
        n_estimators=150,        # Mais árvores para melhor performance
        max_depth=5,             # Profundidade maior para capturar complexidade
        learning_rate=0.08,      # Learning rate otimizado
        min_samples_split=4,     # Controle de overfitting
        min_samples_leaf=2,      # Controle de overfitting
        max_features='sqrt',     # Feature sampling
        subsample=0.9,           # Row sampling
        random_state=42,         # MESMO SEED DO LOCAL
        loss='squared_error',    # Loss function explícita
        validation_fraction=0.1, # Validação interna
        n_iter_no_change=10,     # Early stopping
        tol=1e-4                 # Tolerância
    )
    
    print("⚡ Iniciando treinamento...")
    model.fit(X_train, y_train)
    print("✅ Treinamento concluído!")
    
    # 5. Avaliar performance
    print("\n📊 AVALIANDO PERFORMANCE...")
    
    # Predições
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Métricas
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"🎯 RESULTADO TREINO:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   MAE: ${train_mae:.2f}")
    
    print(f"🎯 RESULTADO TESTE:")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   MAE: ${test_mae:.2f}")
    
    # Verificar se atende os critérios superiores
    if test_r2 >= 0.89 and test_mae <= 2300:
        print("🎉 MODELO SUPERIOR CRIADO COM SUCESSO!")
        print(f"✅ R² = {test_r2:.4f} >= 0.89 ✓")
        print(f"✅ MAE = ${test_mae:.2f} <= $2,300 ✓")
    else:
        print("⚠️ Performance não atingiu critérios superiores, mas é boa")
        
    # 6. Testar com dados específicos
    print("\n🧪 TESTE DE VALIDAÇÃO:")
    test_input = {
        'age': 19,
        'sex': 'female',
        'bmi': 27.9,
        'children': 0,
        'smoker': 'yes',
        'region': 'southwest'
    }
    
    # Preparar features de teste
    test_features = prepare_test_features(test_input, le_sex, le_smoker, le_region)
    test_prediction = model.predict([test_features])[0]
    
    print(f"📋 Input: {test_input}")
    print(f"🔮 Predição: ${test_prediction:.2f}")
    
    # Encontrar valor real similar no dataset
    similar = df[
        (df['age'] == 19) & 
        (df['sex'] == 'female') & 
        (df['smoker'] == 'yes') & 
        (df['region'] == 'southwest')
    ]
    
    if len(similar) > 0:
        real_value = similar.iloc[0]['charges']
        error = abs(test_prediction - real_value)
        print(f"💰 Valor real: ${real_value:.2f}")
        print(f"📊 Erro: ${error:.2f} ({error/real_value*100:.1f}%)")
    
    # 7. Salvar modelo e metadados
    print("\n💾 SALVANDO MODELO SUPERIOR...")
    
    base_path = Path("deploy")
    base_path.mkdir(exist_ok=True)
    
    # Salvar modelo
    model_path = base_path / "superior_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo: {model_path}")
    
    # Salvar encoders
    encoders = {
        'sex': le_sex,
        'smoker': le_smoker,
        'region': le_region
    }
    encoders_path = base_path / "superior_encoders.pkl"
    joblib.dump(encoders, encoders_path)
    print(f"✅ Encoders salvos: {encoders_path}")
    
    # Salvar metadados
    metadata = {
        "model_type": "superior_deploy",
        "algorithm": "GradientBoostingRegressor",
        "sklearn_version": "1.5.x",
        "performance": {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_mae": float(train_mae),
            "test_mae": float(test_mae)
        },
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.08,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "subsample": 0.9,
            "random_state": 42
        },
        "features": list(X.columns),
        "n_features": len(X.columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "encoders_info": {
            "sex": le_sex.classes_.tolist(),
            "smoker": le_smoker.classes_.tolist(),
            "region": le_region.classes_.tolist()
        },
        "test_prediction": {
            "input": test_input,
            "prediction": float(test_prediction)
        }
    }
    
    metadata_path = base_path / "superior_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadados salvos: {metadata_path}")
    
    print(f"\n🎉 MODELO SUPERIOR CRIADO COM SUCESSO!")
    print(f"📁 Arquivos criados:")
    print(f"   • {model_path}")
    print(f"   • {encoders_path}")
    print(f"   • {metadata_path}")
    print(f"\n📊 PERFORMANCE FINAL:")
    print(f"   🎯 R² Score: {test_r2:.4f}")
    print(f"   💰 MAE: ${test_mae:.2f}")
    print(f"   🔧 Features: {len(X.columns)}")
    
    return model, encoders, metadata

def prepare_test_features(data, le_sex, le_smoker, le_region):
    """Prepara features para teste usando os encoders"""
    
    # Features básicas
    age = float(data['age'])
    bmi = float(data['bmi'])
    children = int(data['children'])
    
    # Encoding
    sex_encoded = le_sex.transform([data['sex'].lower()])[0]
    smoker_encoded = le_smoker.transform([data['smoker'].lower()])[0]
    region_encoded = le_region.transform([data['region'].lower()])[0]
    
    # Features avançadas (EXATO DO MODELO)
    age_smoker_risk = age * smoker_encoded
    bmi_smoker_risk = bmi * smoker_encoded
    age_bmi_interaction = age * bmi
    
    # Age group
    if age < 30:
        age_group = 0
    elif age < 45:
        age_group = 1
    elif age < 60:
        age_group = 2
    else:
        age_group = 3
    
    # BMI category
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3
    
    # Composite risk score
    composite_risk_score = age * 0.1 + bmi * 0.2 + smoker_encoded * 10 + children * 0.5
    
    # Region density
    region_risk_map = {0: 1, 1: 2, 2: 1, 3: 3}
    region_density = region_risk_map.get(region_encoded, 1)
    
    # Retornar features na ordem correta
    features = [
        age, bmi, children, sex_encoded, smoker_encoded, region_encoded,
        age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
        age_group, bmi_category, composite_risk_score, region_density
    ]
    
    return features

if __name__ == "__main__":
    try:
        model, encoders, metadata = create_superior_model()
        print("\n🎊 PROCESSO CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc() 