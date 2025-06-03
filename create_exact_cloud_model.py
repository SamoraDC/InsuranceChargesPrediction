#!/usr/bin/env python3
"""
Cria modelo EXATO do local para Streamlit Cloud
Replicando EXATAMENTE o mesmo preprocessamento e modelo
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path('.') / 'src'))

def test_local_preprocessing():
    """Testa como o modelo local processa os dados"""
    print("🔍 ANALISANDO PREPROCESSAMENTO LOCAL...")
    
    from insurance_prediction.models.predictor import predict_insurance_premium
    
    # Dados de teste
    test_data = {
        'age': 30,
        'sex': 'male',
        'bmi': 25.0,
        'children': 1,
        'smoker': 'no',
        'region': 'southwest'
    }
    
    result = predict_insurance_premium(**test_data)
    expected_value = result['predicted_premium']
    
    print(f"✅ Valor esperado do local: ${expected_value:.2f}")
    return expected_value, test_data

def create_exact_replica():
    """Cria réplica EXATA do modelo local"""
    print("🎯 CRIANDO RÉPLICA EXATA DO MODELO LOCAL...")
    
    # 1. Obter valor esperado
    expected_value, test_data = test_local_preprocessing()
    
    # 2. Carregar dados originais
    data_path = Path('data/insurance.csv')
    if not data_path.exists():
        data_path = Path('data/raw/insurance.csv')
    
    df = pd.read_csv(data_path)
    print(f"✅ Dados carregados: {len(df)} amostras")
    
    # 3. Usar EXATAMENTE o mesmo preprocessador do local
    try:
        from insurance_prediction.data.preprocessor import DataPreprocessor
        from insurance_prediction.config.settings import Config
        
        # Criar preprocessor igual ao local
        preprocessor = DataPreprocessor()
        config = Config()
        
        # Preparar dados EXATAMENTE como o local
        X = df.drop(columns=['charges'])
        y = df['charges']
        
        # Aplicar preprocessamento EXATO
        result = preprocessor.fit_transform(X, y)
        if isinstance(result, tuple):
            X_processed = result[0]  # fit_transform retorna (X_transformed, y_transformed)
        else:
            X_processed = result
        
        print(f"✅ Preprocessamento aplicado: {X_processed.shape}")
        print(f"   Features: {preprocessor.selected_features}")
        
        # 4. Usar split EXATO do local (mesma seed)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # 5. Carregar modelo local treinado
        local_model_path = Path('models/gradient_boosting_model.pkl')
        if local_model_path.exists():
            # Usar o modelo já treinado
            model = joblib.load(local_model_path)
            print("✅ Modelo local carregado")
        else:
            # Treinar novo modelo com parâmetros EXATOS
            from sklearn.ensemble import GradientBoostingRegressor
            
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train, y_train)
            print("✅ Modelo treinado")
        
        # 6. Teste com dados de exemplo
        test_df = pd.DataFrame([test_data])
        test_processed = preprocessor.transform(test_df)
        prediction = model.predict(test_processed)[0]
        
        print(f"🧪 TESTE VALIDAÇÃO:")
        print(f"   Esperado: ${expected_value:.2f}")
        print(f"   Obtido:   ${prediction:.2f}")
        print(f"   Diferença: ${abs(expected_value - prediction):.2f}")
        
        if abs(expected_value - prediction) < 1.0:
            print("✅ MODELO REPLICADO COM SUCESSO!")
            
            # 7. Salvar modelo cloud EXATO
            deploy_dir = Path('deploy')
            deploy_dir.mkdir(exist_ok=True)
            
            # Modelo
            model_path = deploy_dir / 'gradient_boosting_model_exact.pkl'
            joblib.dump(model, model_path)
            
            # Preprocessor
            preprocessor_dir = deploy_dir / 'models' / 'model_artifacts'
            preprocessor_dir.mkdir(parents=True, exist_ok=True)
            preprocessor_path = preprocessor_dir / 'preprocessor_exact.pkl'
            
            # Salvar preprocessor como dados serializáveis
            preprocessor_data = {
                'label_encoders': preprocessor.label_encoders,
                'selected_features': preprocessor.selected_features,
                'preprocessing_stats': preprocessor.preprocessing_stats,
                'feature_selector': preprocessor.feature_selector
            }
            joblib.dump(preprocessor_data, preprocessor_path)
            
            # Metadados
            from sklearn.metrics import r2_score, mean_absolute_error
            y_pred = model.predict(X_test)
            
            metadata = {
                'model_type': 'GradientBoostingRegressor',
                'algorithm': 'Gradient Boosting',
                'version': '2.0.0-exact-replica',
                'features': preprocessor.selected_features,
                'n_features': len(preprocessor.selected_features),
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'created_for': 'exact_local_replica',
                'validation_test': {
                    'input': test_data,
                    'expected': expected_value,
                    'obtained': prediction,
                    'difference': abs(expected_value - prediction)
                }
            }
            
            metadata_path = deploy_dir / 'gradient_boosting_model_exact_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Arquivos salvos:")
            print(f"   - {model_path}")
            print(f"   - {metadata_path}")
            print(f"   - {preprocessor_path}")
            
            return True
        else:
            print("❌ FALHA NA REPLICAÇÃO - Diferença muito grande")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = create_exact_replica()
        if success:
            print("\n🎉 MODELO EXATO CRIADO COM SUCESSO!")
            print("🚀 Pronto para deploy no Streamlit Cloud!")
        else:
            print("\n❌ FALHA NA CRIAÇÃO DO MODELO EXATO")
        
    except Exception as e:
        print(f"\n❌ ERRO GERAL: {e}")
        import traceback
        traceback.print_exc() 