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

def load_model():
    """
    Load the trained insurance prediction model.
    
    Returns:
        Model data dictionary or None if error
    """
    model_data = {
        'model': None,
        'preprocessor': None, 
        'feature_names': None,
        'metadata': None,
        'model_type': 'local_superior'
    }
    
    base_path = Path(__file__).parent
    
    # 1ª PRIORIDADE: Modelo local superior (13 features)
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
    
    # 2ª PRIORIDADE: Modelo compatível (fallback)
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
    
    # 3ª PRIORIDADE: Modelo otimizado original (último recurso)
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
    
    # Fallback final: modelo dummy
    logger.error("❌ Todos os modelos falharam - criando modelo dummy")
    from sklearn.ensemble import GradientBoostingRegressor
    model_data['model'] = GradientBoostingRegressor(random_state=42)
    model_data['feature_names'] = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    model_data['model_type'] = 'dummy'
    
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

def predict_premium(input_data, model_data):
    """
    Faz predição usando o modelo carregado
    """
    try:
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        
        # Preparar features baseado no tipo de modelo
        if model_type == 'local_superior':
            # Usar preprocessor para modelo superior
            features = prepare_features_local_model(input_data, model_data.get('preprocessor'))
        else:
            # Modelos simples (compatível, original, dummy)
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