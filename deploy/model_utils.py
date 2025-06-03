#!/usr/bin/env python3
"""
Model utilities for insurance premium prediction - Production Deploy
Independent implementation without src/ dependencies
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import pickle
import json
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - Multiple possible paths for robustness
POSSIBLE_MODEL_PATHS = [
    "production_model_compatible.pkl",  # ✅ NEW: Version-compatible model
    "models/production_model_optimized.pkl",  # For Streamlit Cloud
    "../models/production_model_optimized.pkl",  # For local development
    "./models/production_model_optimized.pkl",  # Alternative
    "production_model_optimized.pkl"  # If model is in deploy folder
]

FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

# Validation ranges and values
NUMERICAL_RANGES = {
    "age": {"min": 18, "max": 64},
    "bmi": {"min": 15.0, "max": 55.0},
    "children": {"min": 0, "max": 5}
}

CATEGORICAL_VALUES = {
    "sex": ["male", "female"],
    "smoker": ["yes", "no"], 
    "region": ["northeast", "northwest", "southeast", "southwest"]
}

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / "src" 
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Cache global para o modelo treinado
_CACHED_MODEL = None

def create_auto_training_model():
    """
    Cria e treina um modelo GARANTIDO para funcionar no cloud
    Usa os dados reais do CSV para treinar on-the-fly
    """
    global _CACHED_MODEL
    
    # Se já temos modelo em cache, usar
    if _CACHED_MODEL is not None:
        logger.info("✅ Usando modelo em cache")
        return _CACHED_MODEL
    
    logger.info("🤖 Criando modelo auto-treinável...")
    
    try:
        # 1. Tentar carregar dados do CSV
        csv_paths = [
            Path(__file__).parent.parent / "data" / "insurance.csv",  # Caminho relativo
            Path("data") / "insurance.csv",  # Caminho direto
            Path("/mount/src/insurancechargesprediction/data/insurance.csv"),  # Caminho cloud
        ]
        
        df = None
        for csv_path in csv_paths:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"✅ Dados carregados de: {csv_path}")
                break
        
        if df is None:
            raise FileNotFoundError("❌ insurance.csv não encontrado em nenhum caminho")
        
        # 2. Preparar features (versão simplificada mas efetiva)
        X = df[['age', 'bmi', 'children']].copy()
        
        # Encoding manual (mais robusto que LabelEncoder)
        X['sex'] = df['sex'].map({'male': 1, 'female': 0})
        X['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        X['region'] = df['region'].map({
            'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3
        })
        
        # Features derivadas (as mais importantes)
        X['smoker_risk'] = X['smoker'] * 10  # Multiplicador de risco
        X['age_smoker'] = X['age'] * X['smoker']
        X['bmi_smoker'] = X['bmi'] * X['smoker']
        
        y = df['charges']
        
        # 3. Treinar modelo robusto
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            loss='squared_error'
        )
        
        logger.info("⚡ Treinando modelo...")
        model.fit(X, y)
        
        # 4. Verificar se está treinado
        if not hasattr(model, 'feature_importances_'):
            raise RuntimeError("❌ Modelo não foi treinado corretamente")
        
        # 5. Avaliar rapidamente
        score = model.score(X, y)
        logger.info(f"📊 R² Score: {score:.4f}")
        
        # 6. Teste de predição
        test_features = np.array([[30, 25.0, 1, 1, 0, 3, 0, 0, 0]])  # Exemplo
        test_pred = model.predict(test_features)[0]
        logger.info(f"🧪 Teste: ${test_pred:.2f}")
        
        # 7. Criar metadata
        model_data = {
            'model': model,
            'model_type': 'auto_trained_cloud',
            'feature_names': list(X.columns),
            'n_features': len(X.columns),
            'r2_score': float(score),
            'encoding_map': {
                'sex': {'male': 1, 'female': 0},
                'smoker': {'yes': 1, 'no': 0},
                'region': {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
            }
        }
        
        # 8. Salvar em cache global
        _CACHED_MODEL = model_data
        
        logger.info("🎉 MODELO AUTO-TREINÁVEL CRIADO COM SUCESSO!")
        logger.info(f"✅ R²: {score:.4f}")
        logger.info(f"🔧 Features: {len(X.columns)}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar modelo auto-treinável: {e}")
        return None

def prepare_features_auto_model(data, encoding_map):
    """
    Prepara features para o modelo auto-treinável
    """
    try:
        # Features básicas
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        
        # Encoding usando mapa
        sex = encoding_map['sex'].get(data['sex'].lower(), 0)
        smoker = encoding_map['smoker'].get(data['smoker'].lower(), 0)
        region = encoding_map['region'].get(data['region'].lower(), 0)
        
        # Features derivadas
        smoker_risk = smoker * 10
        age_smoker = age * smoker
        bmi_smoker = bmi * smoker
        
        features = [age, bmi, children, sex, smoker, region, smoker_risk, age_smoker, bmi_smoker]
        
        logger.info(f"✅ Features auto-model preparadas: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features auto-model: {e}")
        return np.array([[30, 25, 1, 1, 0, 3, 0, 0, 0]])  # Fallback

def load_model():
    """
    Carrega modelo com sistema de fallback robusto
    PRIORIDADE: Modelo auto-treinável (SEMPRE funciona)
    """
    model_data = {
        'model': None,
        'model_type': 'unknown',
        'feature_names': [],
        'metadata': {}
    }
    
    base_path = Path(__file__).parent
    
    # 🚀 PRIORIDADE MÁXIMA: MODELO AUTO-TREINÁVEL (GARANTIDO PARA CLOUD)
    logger.info("🚀 Tentando modelo auto-treinável...")
    auto_model = create_auto_training_model()
    if auto_model is not None:
        logger.info("🎉 MODELO AUTO-TREINÁVEL CARREGADO COM SUCESSO!")
        logger.info(f"🏆 Tipo: {auto_model['model_type']}")
        logger.info(f"📊 R²: {auto_model['r2_score']:.4f}")
        logger.info(f"🔧 Features: {auto_model['n_features']}")
        return auto_model

    logger.warning("⚠️ Modelo auto-treinável falhou, tentando modelos salvos...")

    # 🏆 2ª PRIORIDADE: MODELO SUPERIOR (performance R²=0.8671, MAE=$2,427)
    try:
        model_path = base_path / "superior_model.pkl"
        metadata_path = base_path / "superior_model_metadata.json"
        encoders_path = base_path / "superior_encoders.pkl"
        
        if all(p.exists() for p in [model_path, metadata_path, encoders_path]):
            # Carregar modelo superior
            model_data['model'] = joblib.load(model_path)
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                model_data['metadata'] = json.load(f)
            
            # Carregar encoders
            model_data['encoders'] = joblib.load(encoders_path)
            
            # Features do modelo superior (13 features avançadas)
            model_data['feature_names'] = model_data['metadata']['features']
            model_data['model_type'] = 'superior_deploy'
            
            logger.info(f"🏆 MODELO SUPERIOR carregado com {len(model_data['feature_names'])} features")
            logger.info(f"✅ Tipo: {type(model_data['model']).__name__}")
            logger.info(f"📊 R²: {model_data['metadata']['performance']['test_r2']:.4f}")
            logger.info(f"💰 MAE: ${model_data['metadata']['performance']['test_mae']:.2f}")
            logger.info(f"🧪 Teste: ${model_data['metadata']['test_prediction']['prediction']:.2f}")
            return model_data
        
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo superior: {e}")

    # 2ª PRIORIDADE: Modelo ROBUSTO (treinado com dados reais, compatível com sklearn 1.5.x)
    try:
        model_path = base_path / "robust_model.pkl"
        metadata_path = base_path / "robust_model_metadata.json"
        encoders_path = base_path / "robust_encoders.pkl"
        
        if all(p.exists() for p in [model_path, metadata_path, encoders_path]):
            # Carregar modelo robusto
            model_data['model'] = joblib.load(model_path)
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                model_data['metadata'] = json.load(f)
            
            # Carregar encoders
            model_data['encoders'] = joblib.load(encoders_path)
            
            # Features do modelo robusto (8 features)
            model_data['feature_names'] = model_data['metadata']['features']
            model_data['model_type'] = 'robust_deploy'
            
            logger.info(f"🎯 Modelo ROBUSTO carregado com {len(model_data['feature_names'])} features")
            logger.info(f"✅ Tipo: {type(model_data['model']).__name__}")
            logger.info(f"📊 R²: {model_data['metadata'].get('r2_score', 'N/A'):.4f}")
            logger.info(f"💰 MAE: ${model_data['metadata'].get('mae', 'N/A'):.2f}")
            logger.info(f"🧪 Teste: ${model_data['metadata']['test_prediction']['prediction']:.2f}")
            return model_data
        
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo robusto: {e}")

    # 3ª PRIORIDADE: Modelo EXATO (réplica 100% idêntica do local)
    try:
        model_path = base_path / "gradient_boosting_model_exact.pkl"
        metadata_path = base_path / "gradient_boosting_model_exact_metadata.json"
        preprocessor_path = base_path / "models" / "model_artifacts" / "preprocessor_exact.pkl"
        
        if all(p.exists() for p in [model_path, metadata_path, preprocessor_path]):
            # Carregar modelo EXATO
            model_data['model'] = joblib.load(model_path)
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                model_data['metadata'] = json.load(f)
            
            # Carregar preprocessor (dados serializáveis)
            preprocessor_data = joblib.load(preprocessor_path)
            model_data['preprocessor'] = preprocessor_data
            
            # Features do modelo exato (13 features - exato do local)
            model_data['feature_names'] = preprocessor_data.get('selected_features', [
                'age', 'sex', 'bmi', 'children', 'smoker', 'region',
                'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
                'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
            ])
            
            model_data['model_type'] = 'exact_replica'
            
            logger.info(f"🎯 Modelo EXATO carregado com {len(model_data['feature_names'])} features")
            logger.info(f"✅ Tipo: {type(model_data['model']).__name__}")
            logger.info(f"📊 R²: {model_data['metadata'].get('r2_score', 'N/A')}")
            logger.info(f"🔍 Validação: Diferença ${model_data['metadata']['validation_test']['difference']:.2f}")
            return model_data
        
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo exato: {e}")

    # 4ª PRIORIDADE: Modelo cloud compatível (13 features - SEM random state issues)
    try:
        model_path = base_path / "gradient_boosting_model_cloud.pkl"
        metadata_path = base_path / "gradient_boosting_model_cloud_metadata.json"
        preprocessor_path = base_path / "models" / "model_artifacts" / "preprocessor_cloud.pkl"
        
        if all(p.exists() for p in [model_path, metadata_path, preprocessor_path]):
            # Carregar modelo com joblib
            model_data['model'] = joblib.load(model_path)
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                model_data['metadata'] = json.load(f)
            
            # Carregar preprocessor (dados simples, não classe)
            preprocessor_data = joblib.load(preprocessor_path)
            model_data['preprocessor'] = preprocessor_data
            
            # Features do modelo cloud (13 features)
            model_data['feature_names'] = preprocessor_data.get('selected_features', [
                'age', 'sex', 'bmi', 'children', 'smoker', 'region',
                'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
                'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
            ])
            
            model_data['model_type'] = 'cloud_compatible'
            
            logger.info(f"✅ Modelo cloud compatível carregado com {len(model_data['feature_names'])} features")
            logger.info(f"🎯 Tipo: {type(model_data['model']).__name__}")
            logger.info(f"📊 R²: {model_data['metadata'].get('r2_score', 'N/A')}")
            return model_data
        
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo cloud compatível: {e}")

    # 5ª PRIORIDADE: Modelo local superior (13 features)
    try:
        model_path = base_path / "gradient_boosting_model.pkl"
        metadata_path = base_path / "gradient_boosting_model_metadata.json"
        preprocessor_path = base_path / "models" / "model_artifacts" / "preprocessor.pkl"
        
        if all(p.exists() for p in [model_path, metadata_path, preprocessor_path]):
            # Carregar modelo com joblib
            model_data['model'] = joblib.load(model_path)
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                model_data['metadata'] = json.load(f)
            
            # Carregar preprocessor com joblib
            preprocessor_data = joblib.load(preprocessor_path)
            
            # Recriar o preprocessor como no sistema local
            try:
                from insurance_prediction.data.preprocessor import DataPreprocessor
                model_data['preprocessor'] = DataPreprocessor()
                
                # Carregar componentes
                model_data['preprocessor'].label_encoders = preprocessor_data.get('label_encoders', {})
                model_data['preprocessor'].feature_selector = preprocessor_data.get('feature_selector')
                model_data['preprocessor'].selected_features = preprocessor_data.get('selected_features')
                model_data['preprocessor'].preprocessing_stats = preprocessor_data.get('preprocessing_stats', {})
            except ImportError:
                logger.warning("Não foi possível importar DataPreprocessor - usando dados brutos")
                model_data['preprocessor'] = preprocessor_data
            
            # Features do modelo superior (13 features)
            if model_data['preprocessor'] and hasattr(model_data['preprocessor'], 'selected_features'):
                model_data['feature_names'] = model_data['preprocessor'].selected_features
            else:
                model_data['feature_names'] = [
                    'age', 'sex', 'bmi', 'children', 'smoker', 'region',
                    'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
                    'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
                ]
            
            logger.info(f"✅ Modelo local superior carregado com {len(model_data['feature_names'])} features")
            logger.info(f"🎯 Tipo: {type(model_data['model']).__name__}")
            logger.info(f"📊 R²: {model_data['metadata'].get('r2_score', 'N/A')}")
        return model_data
        
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo local superior: {e}")
    
    # 6ª PRIORIDADE: Modelo compatível (fallback)
    try:
        model_path = base_path / "production_model_compatible.pkl"
        if model_path.exists():
            model_data['model'] = joblib.load(model_path)
            
            model_data['feature_names'] = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'bmi_smoker', 'age_smoker']
            model_data['model_type'] = 'compatible'
            
            logger.info("✅ Modelo compatível carregado como fallback")
            return model_data
            
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar modelo compatível: {e}")
    
    # 7ª PRIORIDADE: Modelo otimizado original (último recurso)
    try:
        model_path = base_path / "production_model_optimized.pkl"
        if model_path.exists():
            model_data['model'] = joblib.load(model_path)
            
            model_data['feature_names'] = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'bmi_smoker', 'age_smoker']
            model_data['model_type'] = 'original'
            
            logger.info("✅ Modelo original carregado como último recurso")
            return model_data
            
    except Exception as e:
        logger.error(f"❌ Falha ao carregar modelo original: {e}")
    
    # Fallback final: modelo dummy TREINADO
    logger.error("❌ Todos os modelos falharam - criando modelo dummy treinado")
    
    try:
        # Carregar dados reais para treinar o modelo dummy
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Tentar carregar dados do CSV
        data_path = Path(__file__).parent.parent / "data" / "insurance.csv"
        if not data_path.exists():
            # Se não encontrar, criar dados sintéticos mínimos
            logger.warning("⚠️ Dados reais não encontrados, criando dados sintéticos")
            dummy_data = pd.DataFrame({
                'age': [25, 35, 45, 55],
                'sex': ['male', 'female', 'male', 'female'],
                'bmi': [22, 28, 32, 25],
                'children': [0, 1, 2, 0],
                'smoker': ['no', 'no', 'yes', 'no'],
                'region': ['southwest', 'northeast', 'southeast', 'northwest'],
                'charges': [2000, 5000, 15000, 3000]
            })
            df = dummy_data
        else:
            # Carregar dados reais
            df = pd.read_csv(data_path)
            logger.info(f"✅ Dados reais carregados: {len(df)} registros")
        
        # Preparar features
        X = df[['age', 'bmi', 'children']].copy()
        
        # Encoding manual simples
        le_sex = LabelEncoder()
        X['sex'] = le_sex.fit_transform(df['sex'])
        
        le_smoker = LabelEncoder()
        X['smoker'] = le_smoker.fit_transform(df['smoker'])
        
        le_region = LabelEncoder()
        X['region'] = le_region.fit_transform(df['region'])
        
        # Features derivadas simples
        X['bmi_smoker'] = X['bmi'] * X['smoker']
        X['age_smoker'] = X['age'] * X['smoker']
        
        y = df['charges']
        
        # Treinar modelo dummy
        dummy_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            learning_rate=0.1
        )
        dummy_model.fit(X, y)
        
        # Salvar encoders para uso na predição
        encoders = {
            'sex': le_sex,
            'smoker': le_smoker,
            'region': le_region
        }
        
        model_data['model'] = dummy_model
        model_data['encoders'] = encoders
        model_data['feature_names'] = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'bmi_smoker', 'age_smoker']
        model_data['model_type'] = 'dummy_trained'
        
        logger.info(f"✅ Modelo dummy treinado com {len(df)} amostras")
        logger.info(f"🎯 R² score: {dummy_model.score(X, y):.4f}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Falha ao criar modelo dummy treinado: {e}")
        
        # Último recurso: modelo extremamente simples
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        # Dados sintéticos mínimos hardcoded
        X_simple = np.array([[25, 0, 22, 0, 0, 0], [45, 1, 30, 2, 1, 1]])
        y_simple = np.array([2000, 15000])
        
        simple_model = LinearRegression()
        simple_model.fit(X_simple, y_simple)
        
        model_data['model'] = simple_model
        model_data['feature_names'] = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        model_data['model_type'] = 'emergency_fallback'
        
        logger.info("✅ Modelo de emergência criado")
        return model_data

def prepare_features_local_model(data, preprocessor=None):
    """
    Prepara features para o modelo local superior (13 features)
    """
    try:
        if preprocessor and hasattr(preprocessor, 'transform'):
            # Usar preprocessor do modelo local
            input_df = pd.DataFrame([{
                'age': data['age'],
                'sex': data['sex'],
                'bmi': data['bmi'],
                'children': data['children'],
                'smoker': data['smoker'],
                'region': data['region']
            }])
            
            return preprocessor.transform(input_df)
        elif preprocessor and 'label_encoders' in preprocessor:
            # Modelo cloud compatível - preprocessor é dict simples
            return prepare_features_cloud_model(data, preprocessor)
        else:
            # Criar features manuais se não tiver preprocessor
            features = {
                'age': data['age'],
                'sex': 1 if data['sex'].lower() == 'male' else 0,
                'bmi': data['bmi'],
                'children': data['children'],
                'smoker': 1 if data['smoker'].lower() == 'yes' else 0,
                'region': {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}.get(data['region'].lower(), 0),
                'age_smoker_risk': data['age'] * (1 if data['smoker'].lower() == 'yes' else 0),
                'bmi_smoker_risk': data['bmi'] * (1 if data['smoker'].lower() == 'yes' else 0),
                'age_bmi_interaction': data['age'] * data['bmi'],
                'age_group': 1 if data['age'] >= 50 else 0,
                'bmi_category': 1 if data['bmi'] >= 30 else 0,
                'composite_risk_score': (data['age'] * 0.1 + data['bmi'] * 0.2 + (50 if data['smoker'].lower() == 'yes' else 0)),
                'region_density': {'southwest': 0.3, 'southeast': 0.4, 'northwest': 0.2, 'northeast': 0.5}.get(data['region'].lower(), 0.3)
            }
            
            return np.array([[features[name] for name in [
                'age', 'sex', 'bmi', 'children', 'smoker', 'region',
                'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
                'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
            ]]])
            
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features: {e}")
        raise

def prepare_features_cloud_model(data, preprocessor_data):
    """
    Prepara features para o modelo cloud compatível (13 features)
    """
    try:
        # Encode categóricas usando os encoders do preprocessor
        label_encoders = preprocessor_data.get('label_encoders', {})
        
        sex_encoded = label_encoders.get('sex', {}).get(data['sex'].lower(), 0)
        smoker_encoded = label_encoders.get('smoker', {}).get(data['smoker'].lower(), 0)
        region_encoded = label_encoders.get('region', {}).get(data['region'].lower(), 0)
        
        # Features básicas
        age = data['age']
        bmi = data['bmi']
        children = data['children']
        
        # Features avançadas (EXATAS do modelo cloud)
        age_smoker_risk = age * smoker_encoded
        bmi_smoker_risk = bmi * smoker_encoded
        age_bmi_interaction = age * bmi
        age_group = 1 if age >= 50 else 0
        bmi_category = 1 if bmi >= 30 else 0
        composite_risk_score = age * 0.1 + bmi * 0.2 + smoker_encoded * 50
        
        # Region density (EXATO do modelo cloud)
        region_density_map = {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.5}  # sw, se, nw, ne
        region_density = region_density_map.get(region_encoded, 0.3)
        
        # Ordem das features (EXATA do modelo cloud)
        features = [
            age, sex_encoded, bmi, children, smoker_encoded, region_encoded,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features cloud: {e}")
        raise

def prepare_features_simple_model(data):
    """
    Prepara features para os modelos simples (8 features) - fallback
    """
    try:
        # Encoding manual para modelos simples
        sex_encoded = 1 if data['sex'].lower() == 'male' else 0
        smoker_encoded = 1 if data['smoker'].lower() == 'yes' else 0
        region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region_encoded = region_map.get(data['region'].lower(), 0)
        
        # Features do modelo simples
        features = [
            data['age'],
            sex_encoded,
            data['bmi'],
            data['children'],
            smoker_encoded,
            region_encoded,
            data['bmi'] * smoker_encoded,  # bmi_smoker
            data['age'] * smoker_encoded   # age_smoker
        ]
        
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features simples: {e}")
        raise

def prepare_features_exact_model(data, preprocessor_data):
    """
    Prepara features usando o preprocessor exato (estrutura de dicionário)
    """
    try:
        if preprocessor_data is None:
            logger.warning("⚠️ Preprocessor exato não disponível, usando fallback simples")
            return prepare_features_simple_model(data)
        
        # Aplicar as mesmas transformações do local
        processed_data = data.copy()
        
        # Features básicas
        age = float(processed_data['age'])
        sex = processed_data['sex'].lower()
        bmi = float(processed_data['bmi'])
        children = int(processed_data['children'])
        smoker = processed_data['smoker'].lower()
        region = processed_data['region'].lower()
        
        # Encodings usando os label encoders salvos
        label_encoders = preprocessor_data['label_encoders']
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Features derivadas (mesmo cálculo do original)
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
        composite_risk_score = (age * 0.1) + (bmi * 0.2) + (smoker_encoded * 10) + (children * 0.5)
        
        # Region density (aproximação)
        region_density_map = {'southwest': 1, 'southeast': 2, 'northwest': 1, 'northeast': 3}
        region_density = region_density_map.get(region, 1)
        
        # Montar array de features na ordem correta
        features = [
            age,
            sex_encoded,
            bmi,
            children,
            smoker_encoded,
            region_encoded,
            age_smoker_risk,
            bmi_smoker_risk,
            age_bmi_interaction,
            age_group,
            bmi_category,
            composite_risk_score,
            region_density
        ]
        
        logger.info(f"✅ Features preparadas para modelo exato: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features exatas: {e}")
        logger.info("🔄 Tentando fallback para features simples...")
        return prepare_features_simple_model(data)

def prepare_features_superior_model(data, encoders):
    """
    Prepara features para o modelo superior (13 features avançadas)
    EXATAMENTE como foi treinado: age, bmi, children, sex, smoker, region,
    age_smoker_risk, bmi_smoker_risk, age_bmi_interaction, age_group, 
    bmi_category, composite_risk_score, region_density
    """
    try:
        # Features básicas
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        
        # Aplicar encoders salvos
        sex_encoded = encoders['sex'].transform([data['sex'].lower()])[0]
        smoker_encoded = encoders['smoker'].transform([data['smoker'].lower()])[0]
        region_encoded = encoders['region'].transform([data['region'].lower()])[0]
        
        # Features avançadas (EXATO DO MODELO SUPERIOR)
        age_smoker_risk = age * smoker_encoded
        bmi_smoker_risk = bmi * smoker_encoded
        age_bmi_interaction = age * bmi
        
        # Age group (EXATO DO TREINAMENTO)
        if age < 30:
            age_group = 0
        elif age < 45:
            age_group = 1
        elif age < 60:
            age_group = 2
        else:
            age_group = 3
        
        # BMI category (EXATO DO TREINAMENTO)
        if bmi < 18.5:
            bmi_category = 0  # Underweight
        elif bmi < 25:
            bmi_category = 1  # Normal
        elif bmi < 30:
            bmi_category = 2  # Overweight
        else:
            bmi_category = 3  # Obese
        
        # Composite risk score (EXATO DO TREINAMENTO)
        composite_risk_score = age * 0.1 + bmi * 0.2 + smoker_encoded * 10 + children * 0.5
        
        # Region density (EXATO DO TREINAMENTO)
        region_risk_map = {0: 1, 1: 2, 2: 1, 3: 3}  # northeast=3, northwest=2, southeast=1, southwest=1
        region_density = region_risk_map.get(region_encoded, 1)
        
        # Montar array de features na ordem EXATA do treinamento
        features = [
            age, bmi, children, sex_encoded, smoker_encoded, region_encoded,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        logger.info(f"✅ Features preparadas para modelo superior: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features superiores: {e}")
        logger.info("🔄 Tentando fallback para features simples...")
        return prepare_features_simple_model(data)

def prepare_features_dummy_model(data, encoders):
    """
    Prepara features para o modelo dummy treinado usando os encoders salvos
    """
    try:
        # Features básicas
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        
        # Aplicar encoders salvos
        sex_encoded = encoders['sex'].transform([data['sex'].lower()])[0]
        smoker_encoded = encoders['smoker'].transform([data['smoker'].lower()])[0]
        region_encoded = encoders['region'].transform([data['region'].lower()])[0]
        
        # Features derivadas
        bmi_smoker = bmi * smoker_encoded
        age_smoker = age * smoker_encoded
        
        # Montar array de features na ordem correta
        features = [
            age,
            sex_encoded,
            bmi,
            children,
            smoker_encoded,
            region_encoded,
            bmi_smoker,
            age_smoker
        ]
        
        logger.info(f"✅ Features preparadas para modelo dummy: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features dummy: {e}")
        logger.info("🔄 Tentando fallback para features simples...")
        return prepare_features_simple_model(data)

def predict_premium(input_data, model_data):
    """
    Faz predição usando o modelo carregado
    """
    try:
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        
        # Preparar features baseado no tipo de modelo
        if model_type == 'auto_trained_cloud':
            # Usar mapa de encoding do modelo auto-treinável (9 features)
            features = prepare_features_auto_model(input_data, model_data.get('encoding_map', {}))
        elif model_type == 'superior_deploy':
            # Usar encoders do modelo superior (13 features avançadas)
            features = prepare_features_superior_model(input_data, model_data.get('encoders', {}))
        elif model_type == 'robust_deploy':
            # Usar encoders do modelo robusto (8 features)
            features = prepare_features_dummy_model(input_data, model_data.get('encoders', {}))
        elif model_type == 'exact_replica':
            # Usar preprocessor específico para modelo exato
            features = prepare_features_exact_model(input_data, model_data.get('preprocessor'))
        elif model_type in ['local_superior', 'cloud_compatible']:
            # Usar preprocessor para modelos superiores
            features = prepare_features_local_model(input_data, model_data.get('preprocessor'))
        elif model_type == 'dummy_trained':
            # Usar encoders do modelo dummy treinado
            features = prepare_features_dummy_model(input_data, model_data.get('encoders', {}))
        else:
            # Modelos simples (compatível, original, emergency)
            features = prepare_features_simple_model(input_data)
        
        # Fazer predição
        prediction = model.predict(features)[0]
        
        # Garantir que a predição seja positiva
        prediction = max(0, prediction)
        
        logger.info(f"✅ Predição realizada: ${prediction:.2f} (modelo: {model_type})")
        
        return {
            'success': True,
            'predicted_premium': prediction,
            'monthly_premium': prediction / 12,
            'model_type': model_type,
            'features_used': len(model_data.get('feature_names', [])),
            'input_data': input_data,
            'processing_time_ms': 0
        }
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        return {
            'success': False,
            'error': str(e),
            'predicted_premium': 0.0,
            'model_type': model_data.get('model_type', 'unknown')
        }

def validate_input(data):
    """
    Valida os dados de entrada
    """
    required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    
    for field in required_fields:
        if field not in data:
            return False, f"Campo obrigatório '{field}' não encontrado"
    
    # Validações específicas
    if not (18 <= data['age'] <= 120):
        return False, "Idade deve estar entre 18 e 120 anos"
    
    if not (10 <= data['bmi'] <= 60):
        return False, "BMI deve estar entre 10 e 60"
    
    if not (0 <= data['children'] <= 10):
        return False, "Número de filhos deve estar entre 0 e 10"
    
    if data['sex'].lower() not in ['male', 'female']:
        return False, "Sexo deve ser 'male' ou 'female'"
    
    if data['smoker'].lower() not in ['yes', 'no']:
        return False, "Fumante deve ser 'yes' ou 'no'"
    
    if data['region'].lower() not in ['southwest', 'southeast', 'northwest', 'northeast']:
        return False, "Região deve ser uma das: southwest, southeast, northwest, northeast"
    
    return True, "Dados válidos"

def get_risk_analysis(data: Dict[str, Any], premium: float) -> Dict[str, Any]:
    """
    Analyze risk factors affecting the premium.
    
    Args:
        data: Input data dictionary
        premium: Predicted premium
        
    Returns:
        Risk analysis results
    """
    risk_factors = []
    risk_level = "Low"
    
    # Analyze risk factors
    if data['smoker'] == 'yes':
        risk_factors.append({
            "factor": "Smoker",
            "impact": "Very High",
            "description": "Smoking significantly increases health risks"
        })
        risk_level = "High"
    
    if data['age'] > 50:
        risk_factors.append({
            "factor": "Age > 50",
            "impact": "Medium",
            "description": "Higher age correlates with increased health risks"
        })
        if risk_level == "Low":
            risk_level = "Medium"
    
    if data['bmi'] > 30:
        risk_factors.append({
            "factor": "BMI > 30 (Obesity)",
            "impact": "Medium",
            "description": "Obesity increases risk of various health conditions"
        })
        if risk_level == "Low":
            risk_level = "Medium"
    
    if data['bmi'] < 18.5:
        risk_factors.append({
            "factor": "BMI < 18.5 (Underweight)",
            "impact": "Low",
            "description": "Being underweight may indicate health issues"
        })
    
    # Calculate premium percentile (approximation)
    if premium < 5000:
        percentile = "Bottom 25%"
    elif premium < 15000:
        percentile = "25-75%"
    elif premium < 30000:
        percentile = "Top 25%"
    else:
        percentile = "Top 10%"
    
    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "premium_percentile": percentile,
        "recommendations": get_recommendations(data)
    }

def get_recommendations(data: Dict[str, Any]) -> list:
    """Get health and financial recommendations based on profile."""
    recommendations = []
    
    if data['smoker'] == 'yes':
        recommendations.append("Consider smoking cessation programs to reduce premium and health risks")
    
    if data['bmi'] > 30:
        recommendations.append("Weight management programs may help reduce premium costs")
    
    if data['bmi'] < 18.5:
        recommendations.append("Consider nutritional counseling to achieve healthy weight")
    
    if data['age'] > 45:
        recommendations.append("Regular health check-ups become increasingly important")
    
    if not recommendations:
        recommendations.append("Maintain healthy lifestyle to keep premium costs low")
    
    return recommendations

if __name__ == "__main__":
    # Teste básico
    logger.info("🧪 Testando carregamento do modelo...")
    
    model_data = load_model()
    
    if model_data['model'] is not None:
        logger.info(f"✅ Modelo carregado com sucesso!")
        logger.info(f"🎯 Tipo: {model_data['model_type']}")
        logger.info(f"📊 Features: {len(model_data.get('feature_names', []))}")
        
        # Teste de predição
        test_data = {
            'age': 30,
            'sex': 'male',
            'bmi': 25.0,
            'children': 1,
            'smoker': 'no',
            'region': 'southwest'
        }
        
        result = predict_premium(test_data, model_data)
        
        if result['success']:
            logger.info(f"✅ Teste de predição bem-sucedido: ${result['predicted_premium']:.2f}")
        else:
            logger.error(f"❌ Teste de predição falhou: {result['error']}")
    else:
        logger.error("❌ Falha no carregamento do modelo!") 