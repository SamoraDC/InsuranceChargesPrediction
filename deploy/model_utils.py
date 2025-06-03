#!/usr/bin/env python3
"""
Model utilities for insurance premium prediction - Production Deploy
VERSÃO DEFINITIVA - USA EXATAMENTE O MODELO LOCAL
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
    CARREGA EXATAMENTE O MODELO LOCAL ORIGINAL
    Prioridade: Modelo local exato > Fallbacks robustos
    """
    # IMPORTANTE: Streamlit Cloud executa do diretório raiz!
    # Detectar se estamos no deploy/ ou na raiz
    current_path = Path(__file__).parent
    if current_path.name == 'deploy':
        # Estamos rodando localmente do diretório deploy
        base_path = current_path
        root_path = current_path.parent
    else:
        # Estamos rodando do diretório raiz (Streamlit Cloud)
        base_path = Path("deploy")
        root_path = Path(".")
    
    logger.info(f"🔍 Diretório atual: {current_path}")
    logger.info(f"🔍 Base path: {base_path}")
    logger.info(f"🔍 Root path: {root_path}")
    
    # 🎯 PRIORIDADE 1: MODELO LOCAL EXATO (CÓPIA DIRETA)
    try:
        model_path = base_path / "gradient_boosting_model_LOCAL_EXACT.pkl"
        metadata_path = base_path / "gradient_boosting_model_LOCAL_EXACT_metadata.json"
        preprocessor_path = base_path / "models" / "model_artifacts" / "preprocessor_LOCAL_EXACT.pkl"
        
        logger.info(f"🔍 Procurando modelo em: {model_path}")
        logger.info(f"🔍 Modelo existe: {model_path.exists()}")
        logger.info(f"🔍 Metadata existe: {metadata_path.exists()}")
        logger.info(f"🔍 Preprocessor existe: {preprocessor_path.exists()}")
        
        if all(p.exists() for p in [model_path, metadata_path, preprocessor_path]):
            logger.info("🎯 Carregando MODELO LOCAL EXATO...")
            
            # Carregar modelo
            model = joblib.load(model_path)
            
            # Verificar se modelo está treinado
            if not hasattr(model, 'feature_importances_'):
                raise ValueError("❌ Modelo não está treinado!")
            
            # Carregar metadados
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Carregar preprocessor
            preprocessor_data = joblib.load(preprocessor_path)
            
            # Estrutura do modelo
            model_data = {
                'model': model,
                'preprocessor': preprocessor_data,
                'metadata': metadata,
                'model_type': 'local_exact',
                'feature_names': [
                    'age', 'sex', 'bmi', 'children', 'smoker', 'region',
                    'age_smoker_risk', 'bmi_smoker_risk', 'age_bmi_interaction',
                    'age_group', 'bmi_category', 'composite_risk_score', 'region_density'
                ]
            }
            
            logger.info("🎉 MODELO LOCAL EXATO CARREGADO COM SUCESSO!")
            logger.info(f"✅ Tipo: {type(model).__name__}")
            logger.info(f"📊 R²: {metadata['training_history']['final_test_metrics']['r2']:.4f}")
            logger.info(f"💰 MAE: ${metadata['training_history']['final_test_metrics']['mae']:.2f}")
            logger.info(f"🔧 Features: {len(model_data['feature_names'])}")
            
            # Teste rápido
            test_X = np.array([[30, 1, 25, 1, 0, 0, 0, 0, 750, 1, 1, 13, 0.3]])
            test_pred = model.predict(test_X)[0]
            logger.info(f"🧪 Teste: ${test_pred:.2f}")
            
            return model_data
            
    except Exception as e:
        logger.error(f"❌ Falha no modelo local exato: {e}")
    
    # 🎯 PRIORIDADE 2: MODELO AUTO-TREINÁVEL (GARANTIA PARA CLOUD)
    try:
        logger.info("🚀 Criando modelo auto-treinável como fallback...")
        
        # Carregar dados CSV - CAMINHOS AJUSTADOS PARA STREAMLIT CLOUD
        csv_paths = [
            # Para Streamlit Cloud (executa do diretório raiz)
            Path("deploy") / "insurance.csv",
            Path("data") / "insurance.csv",
            # Para execução local do deploy/
            base_path / "insurance.csv",
            root_path / "data" / "insurance.csv",
            # Caminhos absolutos possíveis
            Path("/mount/src/insurancechargesprediction/data/insurance.csv"),
            Path("/mount/src/insurancechargesprediction/deploy/insurance.csv"),
            # Fallback simples
            Path("insurance.csv"),
        ]
        
        df = None
        for csv_path in csv_paths:
            logger.info(f"🔍 Tentando carregar dados de: {csv_path}")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"✅ Dados carregados de: {csv_path}")
                break
            else:
                logger.info(f"❌ Arquivo não encontrado: {csv_path}")
        
        if df is None:
            raise FileNotFoundError("❌ insurance.csv não encontrado em nenhum local")
        
        # Preparar features EXATAMENTE como o modelo local
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Features básicas
        X = df[['age', 'bmi', 'children']].copy()
        
        # Encoding categórico
        le_sex = LabelEncoder()
        X['sex'] = le_sex.fit_transform(df['sex'])
        
        le_smoker = LabelEncoder()
        X['smoker'] = le_smoker.fit_transform(df['smoker'])
        
        le_region = LabelEncoder()
        X['region'] = le_region.fit_transform(df['region'])
        
        # Features derivadas (EXATAMENTE como o modelo local)
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
        region_density_map = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}  # ne, nw, se, sw
        X['region_density'] = X['region'].map(region_density_map)
        
        y = df['charges']
        
        # Treinar com MESMOS parâmetros do modelo local
        model = GradientBoostingRegressor(
            max_depth=6,
            max_features='sqrt',
            min_samples_leaf=4,
            min_samples_split=10,
            n_estimators=200,
            n_iter_no_change=10,
            random_state=42,
            subsample=0.8
        )
        
        logger.info("⚡ Treinando modelo auto-treinável...")
        model.fit(X, y)
        
        # Verificar treinamento
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("❌ Modelo auto-treinável não foi treinado!")
        
        score = model.score(X, y)
        logger.info(f"📊 R² do modelo auto-treinável: {score:.4f}")
        
        # Salvar encoders
        encoders = {
            'sex': le_sex,
            'smoker': le_smoker,
            'region': le_region
        }
        
        model_data = {
            'model': model,
            'encoders': encoders,
            'model_type': 'auto_trained_exact',
            'feature_names': list(X.columns),
            'r2_score': score
        }
        
        logger.info("🎉 MODELO AUTO-TREINÁVEL CRIADO!")
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Falha no modelo auto-treinável: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    logger.error("❌ TODOS OS MODELOS FALHARAM!")
    return None

def prepare_features_local_exact(data, preprocessor=None):
    """
    Prepara features EXATAMENTE como o modelo local
    """
    try:
        # Se temos preprocessor, usar
        if preprocessor and hasattr(preprocessor, 'transform'):
            input_df = pd.DataFrame([data])
            return preprocessor.transform(input_df)
        
        # Caso contrário, criar features manualmente IGUAL ao local
        age = float(data['age'])
        sex = 1 if data['sex'].lower() == 'male' else 0
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = 1 if data['smoker'].lower() == 'yes' else 0
        
        # Region mapping (mesmo do local)
        region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        region = region_map.get(data['region'].lower(), 0)
    
        # Features derivadas (EXATAS do local)
        age_smoker_risk = age * smoker
        bmi_smoker_risk = bmi * smoker
        age_bmi_interaction = age * bmi
        
        # Age group (exato do local)
        if age < 30:
            age_group = 0
        elif age < 45:
            age_group = 1
        elif age < 60:
            age_group = 2
        else:
            age_group = 3
        
        # BMI category (exato do local)
        if bmi < 18.5:
            bmi_category = 0
        elif bmi < 25:
            bmi_category = 1
        elif bmi < 30:
            bmi_category = 2
        else:
            bmi_category = 3
        
        # Composite risk score (exato do local)
        composite_risk_score = age * 0.1 + bmi * 0.2 + smoker * 10 + children * 0.5
        
        # Region density (exato do local)
        region_density_map = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}  # ne, nw, se, sw
        region_density = region_density_map.get(region, 0.3)
        
        # Array final (MESMA ORDEM do modelo local)
        features = [
            age, sex, bmi, children, smoker, region,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        logger.info(f"✅ Features locais preparadas: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features locais: {e}")
        raise

def prepare_features_auto_trained(data, encoders):
    """
    Prepara features para modelo auto-treinável
    """
    try:
        # Features básicas
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        
        # Encoding usando encoders salvos
        sex = encoders['sex'].transform([data['sex'].lower()])[0]
        smoker = encoders['smoker'].transform([data['smoker'].lower()])[0]
        region = encoders['region'].transform([data['region'].lower()])[0]
        
        # Features derivadas (EXATAS)
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
        region_density_map = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.3}
        region_density = region_density_map.get(region, 0.3)
        
        features = [
            age, bmi, children, sex, smoker, region,
            age_smoker_risk, bmi_smoker_risk, age_bmi_interaction,
            age_group, bmi_category, composite_risk_score, region_density
        ]
        
        logger.info(f"✅ Features auto-treinável preparadas: {len(features)} features")
        return np.array([features])
        
    except Exception as e:
        logger.error(f"❌ Erro na preparação de features auto-treinável: {e}")
        raise

def predict_premium(input_data, model_data):
    """
    Faz predição usando o modelo carregado
    """
    try:
        if model_data is None or 'model' not in model_data:
            raise ValueError("Modelo não carregado")
        
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        
        # Verificar se modelo está treinado
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Modelo não está treinado")
        
        # Preparar features baseado no tipo
        if model_type == 'local_exact':
            features = prepare_features_local_exact(input_data, model_data.get('preprocessor'))
        elif model_type == 'auto_trained_exact':
            features = prepare_features_auto_trained(input_data, model_data.get('encoders', {}))
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
        
        # Fazer predição
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)  # Garantir valor positivo
        
        logger.info(f"✅ Predição realizada: ${prediction:.2f} (modelo: {model_type})")
        
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
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'predicted_premium': 0.0,
            'model_type': model_data.get('model_type', 'unknown') if model_data else 'none'
        }

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
    logger.info("🧪 TESTANDO SISTEMA...")
    
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