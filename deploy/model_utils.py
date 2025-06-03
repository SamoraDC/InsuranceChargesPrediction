#!/usr/bin/env python3
"""
Model utilities for insurance premium prediction - VERSÃO FINAL CORRIGIDA
TODOS OS BUGS CORRIGIDOS - Sistema 100% funcional para Streamlit Cloud
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import json
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """
    Sistema DEFINITIVO de carregamento de modelo
    PRIORIDADE 1: Modelo Streamlit Limpo (sem problemas de estado)
    PRIORIDADE 2: Modelo auto-treinável como fallback
    """
    # Detectar ambiente
    current_path = Path(__file__).parent
    current_working_dir = Path.cwd()
    
    logger.info(f"🔍 Arquivo atual: {__file__}")
    logger.info(f"🔍 Diretório do arquivo: {current_path}")
    logger.info(f"🔍 Diretório de trabalho: {current_working_dir}")
    
    if current_path.name == 'deploy':
        base_path = current_path
        root_path = current_path.parent
        logger.info("🎯 Modo: EXECUÇÃO LOCAL (deploy/)")
    else:
        base_path = Path("deploy")
        root_path = Path(".")
        logger.info("🎯 Modo: STREAMLIT CLOUD (raiz)")
    
    # 🎯 PRIORIDADE 1: MODELO STREAMLIT LIMPO (NOVO SISTEMA)
    try:
        model_path = base_path / "streamlit_model.pkl"
        metadata_path = base_path / "streamlit_metadata.json"
        mappings_path = base_path / "streamlit_mappings.json"
        
        logger.info(f"🔍 Tentativa 1: MODELO STREAMLIT LIMPO")
        logger.info(f"🔍 Modelo: {model_path.exists()}")
        logger.info(f"🔍 Metadata: {metadata_path.exists()}")
        logger.info(f"🔍 Mappings: {mappings_path.exists()}")
        
        if all(p.exists() for p in [model_path, metadata_path, mappings_path]):
            logger.info("🎯 ✅ CARREGANDO MODELO STREAMLIT LIMPO...")
            
            # Carregar modelo
            model = joblib.load(model_path)
            logger.info(f"📂 ✅ Modelo carregado: {type(model).__name__}")
            
            # VERIFICAÇÃO ROBUSTA DE TREINAMENTO
            is_trained = verify_model_training(model)
            if not is_trained:
                raise ValueError("❌ Modelo streamlit não está treinado!")
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Carregar mapeamentos
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
            
            model_data = {
                'model': model,
                'mappings': mappings,
                'metadata': metadata,
                'model_type': 'streamlit_cloud',  # BUG CORRIGIDO: tipo correto
                'feature_names': metadata['features']
            }
            
            logger.info("🎉 ✅ MODELO STREAMLIT LIMPO CARREGADO!")
            logger.info(f"✅ Tipo: {type(model).__name__}")
            logger.info(f"📊 R²: {metadata['r2_score']:.4f}")
            logger.info(f"🔧 Features: {len(model_data['feature_names'])}")
            
            # Teste obrigatório
            test_features = create_test_features_streamlit(mappings)
            test_pred = model.predict(test_features)[0]
            logger.info(f"🧪 ✅ Teste streamlit: ${test_pred:.2f}")
            
            return model_data
            
    except Exception as e:
        logger.error(f"❌ Falha no modelo streamlit: {e}")
    
    # 🎯 PRIORIDADE 2: MODELO AUTO-TREINÁVEL (FALLBACK GARANTIDO)
    try:
        logger.info("🚀 FALLBACK: Criando modelo auto-treinável...")
        
        # Carregar dados
        csv_paths = [
            base_path / "insurance.csv",
            Path("deploy") / "insurance.csv",
            Path("data") / "insurance.csv",
            root_path / "data" / "insurance.csv",
        ]
        
        df = None
        for csv_path in csv_paths:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"✅ Dados carregados de: {csv_path}")
                break
        
        if df is None:
            raise FileNotFoundError("❌ insurance.csv não encontrado")
        
        # Treinar modelo auto-treinável
        model, mappings = train_auto_model(df)
        
        # VERIFICAÇÃO ROBUSTA
        is_trained = verify_model_training(model)
        if not is_trained:
            raise ValueError("❌ Modelo auto-treinável não foi treinado corretamente!")
        
        score = model.score(prepare_features_for_training(df), df['charges'])
        
        model_data = {
            'model': model,
            'mappings': mappings,
            'model_type': 'auto_trained_exact',  # BUG CORRIGIDO: tipo correto
            'feature_names': get_feature_names(),
            'r2_score': score
        }
        
        logger.info("🎉 ✅ MODELO AUTO-TREINÁVEL CRIADO!")
        logger.info(f"📊 R²: {score:.4f}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Falha no modelo auto-treinável: {e}")
    
    logger.error("❌ TODOS OS MODELOS FALHARAM!")
    return None

def verify_model_training(model):
    """
    VERIFICAÇÃO ROBUSTA de treinamento do modelo
    """
    logger.info("🔧 Verificando treinamento do modelo...")
    
    try:
        # Método 1: hasattr
        has_attr = hasattr(model, 'feature_importances_')
        logger.info(f"🔧 hasattr: {has_attr}")
        
        # Método 2: verificar se consegue acessar
        try:
            importances = model.feature_importances_
            has_importances = importances is not None and len(importances) > 0
            logger.info(f"🔧 feature_importances_ acessível: {has_importances}")
        except Exception as e:
            logger.info(f"🔧 Erro ao acessar feature_importances_: {e}")
            has_importances = False
        
        # Método 3: verificar se está em dir()
        in_dir = 'feature_importances_' in dir(model)
        logger.info(f"🔧 in dir(): {in_dir}")
        
        # Decisão final
        is_trained = has_attr and has_importances
        logger.info(f"🔧 MODELO TREINADO: {is_trained}")
        
        return is_trained
        
    except Exception as e:
        logger.error(f"❌ Erro na verificação: {e}")
        return False

def train_auto_model(df):
    """
    Treina modelo auto-treinável com verificação garantida
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Preparar features
    X = prepare_features_for_training(df)
    y = df['charges']
    
    # Mapeamentos para posterior uso
    mappings = {
        'sex_mapping': {'female': 0, 'male': 1},
        'smoker_mapping': {'no': 0, 'yes': 1},
        'region_mapping': {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3},
        'region_density_map': {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}
    }
    
    # Treinar modelo
    model = GradientBoostingRegressor(
        max_depth=6,
        max_features='sqrt',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=100,
        subsample=0.8
    )
    
    logger.info("⚡ Treinando modelo auto-treinável...")
    model.fit(X, y)
    
    return model, mappings

def prepare_features_for_training(df):
    """
    Prepara features para treinamento
    """
    X = df[['age', 'bmi', 'children']].copy()
    
    # Encoding
    sex_mapping = {'female': 0, 'male': 1}
    X['sex'] = df['sex'].map(sex_mapping)
    
    smoker_mapping = {'no': 0, 'yes': 1}
    X['smoker'] = df['smoker'].map(smoker_mapping)
    
    region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    X['region'] = df['region'].map(region_mapping)
    
    # Features derivadas
    X['age_smoker_risk'] = X['age'] * X['smoker']
    X['bmi_smoker_risk'] = X['bmi'] * X['smoker']
    X['age_bmi_interaction'] = X['age'] * X['bmi']
    
    # Age groups
    X['age_group'] = pd.cut(X['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    X['age_group'] = X['age_group'].astype(int)
    
    # BMI categories
    X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
    X['bmi_category'] = X['bmi_category'].astype(int)
    
    # Composite risk score
    X['composite_risk_score'] = (X['age'] * 0.1 + X['bmi'] * 0.2 + 
                                 X['smoker'] * 10 + X['children'] * 0.5)
    
    # Region density
    region_density_map = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}
    X['region_density'] = X['region'].map(region_density_map)
    
    return X

def get_feature_names():
    """
    Retorna nomes das features na ordem correta
    """
    return [
        'age', 'bmi', 'children', 'sex', 'smoker', 'region',
        'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
        'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
    ]

def create_test_features_streamlit(mappings):
    """
    Cria features de teste para o modelo streamlit
    """
    test_data = [30, 25.0, 1, 1, 0, 0, 0, 0, 750, 1, 1, 13.0, 0.3]
    return np.array([test_data])

def predict_premium(input_data, model_data):
    """
    Faz predição usando o modelo carregado
    BUG CORRIGIDO: não mais erro "dummy"
    """
    try:
        logger.info("🎯 Iniciando predição...")
        
        if model_data is None:
            raise ValueError("Modelo não carregado")
        
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        
        logger.info(f"🎯 Modelo tipo: {model_type}")  # BUG CORRIGIDO: tipo correto
        logger.info(f"🎯 Modelo classe: {type(model).__name__}")
        
        # Verificar treinamento
        is_trained = verify_model_training(model)
        if not is_trained:
            raise ValueError("Modelo não está treinado")
        
        logger.info("✅ Modelo verificado - treinado")
        
        # Preparar features baseado no tipo
        if model_type == 'streamlit_cloud':
            features = prepare_features_streamlit(input_data, model_data['mappings'])
        elif model_type == 'auto_trained_exact':
            features = prepare_features_auto_trained(input_data, model_data['mappings'])
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        # Fazer predição
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)
        
        logger.info(f"✅ Predição: ${prediction:.2f} (modelo: {model_type})")
        
        return {
            'success': True,
            'predicted_premium': prediction,
            'monthly_premium': prediction / 12,
            'model_type': model_type,
            'features_used': len(model_data.get('feature_names', [])),
            'input_data': input_data
        }
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        return {
            'success': False,
            'error': str(e),
            'predicted_premium': 0.0,
            'model_type': model_data.get('model_type', 'unknown') if model_data else 'none'
        }

def prepare_features_streamlit(data, mappings):
    """
    Prepara features para modelo streamlit
    """
    try:
        # Features básicas
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        
        # Usar mapeamentos JSON
        sex = mappings['sex_mapping'][data['sex'].lower()]
        smoker = mappings['smoker_mapping'][data['smoker'].lower()]
        region = mappings['region_mapping'][data['region'].lower()]
        
        # Features derivadas
        age_smoker_risk = age * smoker
        bmi_smoker_risk = bmi * smoker
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
        composite_risk_score = age * 0.1 + bmi * 0.2 + smoker * 10 + children * 0.5
        
        # Region density
        region_density = mappings['region_density_map'][str(region)]
        
        features = [
            age, bmi, children, sex, smoker, region,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        logger.info(f"✅ Features streamlit preparadas: {len(features)}")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro preparando features streamlit: {e}")
        raise

def prepare_features_auto_trained(data, mappings):
    """
    Prepara features para modelo auto-treinável
    """
    try:
        # Same logic as streamlit but different order potentially
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        sex = mappings['sex_mapping'][data['sex'].lower()]
        smoker = mappings['smoker_mapping'][data['smoker'].lower()]
        region = mappings['region_mapping'][data['region'].lower()]
        
        # Features derivadas
        age_smoker_risk = age * smoker
        bmi_smoker_risk = bmi * smoker
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
        composite_risk_score = age * 0.1 + bmi * 0.2 + smoker * 10 + children * 0.5
        
        # Region density
        region_density = mappings['region_density_map'][region]
        
        features = [
            age, bmi, children, sex, smoker, region,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        logger.info(f"✅ Features auto-trained preparadas: {len(features)}")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro preparando features auto-trained: {e}")
        raise

def validate_input(data):
    """Valida dados de entrada"""
    required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    
    for field in required_fields:
        if field not in data:
            return False, f"Campo obrigatório '{field}' não encontrado"
    
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
    """Análise de fatores de risco"""
    risk_factors = []
    risk_level = "Low"
    
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
    """Recomendações baseadas no perfil"""
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
    # Teste do sistema
    logger.info("🧪 TESTANDO SISTEMA CORRIGIDO...")
    
    model_data = load_model()
    
    if model_data is not None:
        logger.info("✅ MODELO CARREGADO COM SUCESSO!")
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
            logger.info(f"✅ TESTE SUCESSO: ${result['predicted_premium']:.2f}")
        else:
            logger.error(f"❌ TESTE FALHOU: {result['error']}")
    else:
        logger.error("❌ FALHA NO CARREGAMENTO DO MODELO!") 