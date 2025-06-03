#!/usr/bin/env python3
"""
Cria modelo compatível com Streamlit Cloud
Usado EXATAMENTE o mesmo pipeline do local, mas sem dependências de random state
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys

# Add src to path
sys.path.insert(0, str(Path('.') / 'src'))

def create_features(df):
    """
    Cria EXATAMENTE as mesmas features do modelo local
    """
    df_copy = df.copy()
    
    # Encode categóricas
    df_copy['sex'] = df_copy['sex'].map({'male': 1, 'female': 0})
    df_copy['smoker'] = df_copy['smoker'].map({'yes': 1, 'no': 0})
    df_copy['region'] = df_copy['region'].map({
        'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3
    })
    
    # Features avançadas (EXATAS do modelo local)
    df_copy['age_smoker_risk'] = df_copy['age'] * df_copy['smoker']
    df_copy['bmi_smoker_risk'] = df_copy['bmi'] * df_copy['smoker'] 
    df_copy['age_bmi_interaction'] = df_copy['age'] * df_copy['bmi']
    df_copy['age_group'] = (df_copy['age'] >= 50).astype(int)
    df_copy['bmi_category'] = (df_copy['bmi'] >= 30).astype(int)
    df_copy['composite_risk_score'] = (
        df_copy['age'] * 0.1 + 
        df_copy['bmi'] * 0.2 + 
        df_copy['smoker'] * 50
    )
    
    # Region density (EXATO do modelo local)
    region_density_map = {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.5}  # sw, se, nw, ne
    df_copy['region_density'] = df_copy['region'].map(region_density_map)
    
    # Ordem das features (EXATA do modelo local)
    feature_columns = [
        'age', 'sex', 'bmi', 'children', 'smoker', 'region',
        'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
        'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
    ]
    
    return df_copy[feature_columns]

def create_compatible_model():
    """
    Cria modelo compatível com Streamlit Cloud usando exatamente o mesmo algoritmo do local
    """
    print("🏗️ CRIANDO MODELO COMPATÍVEL COM STREAMLIT CLOUD...")
    
    # 1. Carregar dados (mesmos dados do local)
    data_path = Path('data/insurance.csv')
    if not data_path.exists():
        data_path = Path('data/raw/insurance.csv')
    
    if not data_path.exists():
        raise FileNotFoundError("Arquivo insurance.csv não encontrado!")
    
    df = pd.read_csv(data_path)
    print(f"✅ Dados carregados: {len(df)} amostras")
    
    # 2. Preparar features (EXATO do modelo local)
    X = create_features(df)
    y = df['charges']
    
    print(f"✅ Features criadas: {list(X.columns)}")
    print(f"   Total features: {X.shape[1]}")
    
    # 3. Split treino/teste (mesma proporção do local)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Treinar modelo (EXATOS parâmetros do local)
    print("🤖 Treinando modelo...")
    
    # Usar exatamente os mesmos parâmetros do modelo local
    model = GradientBoostingRegressor(
        n_estimators=200,           # Mesmo número de estimadores  
        learning_rate=0.1,          # Mesma taxa de aprendizado
        max_depth=6,                # Mesma profundidade
        min_samples_split=10,       # Mesmo min_samples_split
        min_samples_leaf=4,         # Mesmo min_samples_leaf
        subsample=0.8,              # Mesmo subsample
        random_state=42             # Seed para reproducibilidade
    )
    
    model.fit(X_train, y_train)
    
    # 5. Avaliar modelo
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"📊 Performance do modelo:")
    print(f"   R² treino: {r2_train:.4f}")
    print(f"   R² teste: {r2_test:.4f}")
    print(f"   MAE teste: ${mae_test:.2f}")
    print(f"   RMSE teste: ${rmse_test:.2f}")
    
    # 6. Criar preprocessor mock (compatível)
    preprocessor_data = {
        'feature_names': list(X.columns),
        'label_encoders': {
            'sex': {'male': 1, 'female': 0},
            'smoker': {'yes': 1, 'no': 0},
            'region': {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        },
        'selected_features': list(X.columns),
        'preprocessing_stats': {
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    }
    
    # 7. Metadados
    metadata = {
        'model_type': 'GradientBoostingRegressor',
        'algorithm': 'Gradient Boosting',
        'version': '2.0.0-cloud-compatible',
        'features': list(X.columns),
        'n_features': X.shape[1],
        'r2_score': r2_test,
        'mae': mae_test,
        'rmse': rmse_test,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'created_for': 'streamlit_cloud_deploy'
    }
    
    # 8. Salvar arquivos
    deploy_dir = Path('deploy')
    deploy_dir.mkdir(exist_ok=True)
    
    # Modelo
    model_path = deploy_dir / 'gradient_boosting_model_cloud.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo: {model_path}")
    
    # Metadados  
    metadata_path = deploy_dir / 'gradient_boosting_model_cloud_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadados salvos: {metadata_path}")
    
    # Preprocessor
    preprocessor_dir = deploy_dir / 'models' / 'model_artifacts'
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = preprocessor_dir / 'preprocessor_cloud.pkl'
    joblib.dump(preprocessor_data, preprocessor_path)
    print(f"✅ Preprocessor salvo: {preprocessor_path}")
    
    # 9. Teste rápido
    print("\n🧪 TESTE RÁPIDO:")
    test_sample = X_test.iloc[0:1]
    prediction = model.predict(test_sample)[0]
    actual = y_test.iloc[0]
    
    print(f"   Predição: ${prediction:.2f}")
    print(f"   Real: ${actual:.2f}")
    print(f"   Diferença: ${abs(prediction - actual):.2f}")
    
    return model, metadata, preprocessor_data

if __name__ == "__main__":
    try:
        model, metadata, preprocessor = create_compatible_model()
        print("\n🎉 MODELO COMPATÍVEL CRIADO COM SUCESSO!")
        print(f"📊 Performance: R²={metadata['r2_score']:.3f}, MAE=${metadata['mae']:.0f}")
        print("🚀 Pronto para deploy no Streamlit Cloud!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc() 