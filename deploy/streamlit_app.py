#!/usr/bin/env python3
"""
🏥 Insurance Charges Predictor - Streamlit App
🏥 Preditor de Preço de Convênio Médico - Aplicativo Streamlit

Bilingual Insurance Charges Prediction App for Streamlit Cloud Deploy
Aplicativo Bilíngue de Predição de Preço de Convênio Médico para Deploy no Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import joblib
from pathlib import Path
import sys

# Add path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# VERIFICAÇÃO CRÍTICA: Garantir que está usando o arquivo correto
model_utils_path = current_dir / "model_utils.py"
print(f"🔍 VERIFICAÇÃO CRÍTICA:")
print(f"🔍 Arquivo atual: {__file__}")
print(f"🔍 Diretório: {current_dir}")
print(f"🔍 model_utils.py existe: {model_utils_path.exists()}")
print(f"🔍 model_utils.py path: {model_utils_path.absolute()}")

try:
    # IMPORTAÇÃO EXPLÍCITA E VERIFICADA
    print("🔍 Tentando importar deploy/model_utils.py...")
    from model_utils import load_model, predict_premium, get_risk_analysis
    
    # VERIFICAÇÃO ADICIONAL: Testar se o modelo está correto
    print("🔍 Testando carregamento do modelo...")
    test_model = load_model()
    if test_model:
        model_type = test_model.get('model_type', 'unknown')
        print(f"✅ Modelo carregado com sucesso! Tipo: {model_type}")
        if model_type == 'dummy' or 'dummy' in str(model_type).lower():
            print("❌ ERRO CRÍTICO: Modelo dummy detectado!")
            raise ImportError("Modelo dummy sendo usado - arquivo errado!")
        else:
            print(f"✅ Modelo verificado: {model_type}")
    else:
        print("❌ Falha no carregamento do modelo")
    
    USE_LOCAL_MODEL = False
    print("✅ Using deploy model_utils (VERIFICADO - sem dummy)")
except ImportError as e:
    print(f"❌ Falha ao importar deploy model_utils: {e}")
    try:
        # Fallback to local system
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from insurance_prediction.models.predictor import load_production_model, predict_insurance_premium
        from insurance_prediction.config.settings import Config
        USE_LOCAL_MODEL = True
        print("✅ Using local model system (fallback)")
    except ImportError as e:
        print(f"❌ Failed to import any model system: {e}")
        USE_LOCAL_MODEL = None

# =============================================================================
# TRANSLATIONS / TRADUÇÕES
# =============================================================================

TRANSLATIONS = {
    "en": {
        # App Config
        "page_title": "🏥 Insurance Charges Predictor",
        "main_header": "🏥 Insurance Charges Predictor",
        "sub_header": "AI-powered system using Gradient Boosting algorithm",
        
        # Language Toggle
        "language": "🌍 Language",
        "lang_en": "🇺🇸 English",
        "lang_pt": "🇧🇷 Português",
        
        # Navigation
        "tab_individual": "🎯 Individual Prediction",
        "tab_batch": "📊 Batch Analysis", 
        "tab_about": "ℹ️ About",
        
        # Sidebar
        "sidebar_title": "🔧 System Information",
        "model_loaded": "✅ Model loaded!",
        "model_not_loaded": "❌ Model not loaded",
        "model_details": "📊 Model Details",
        "algorithm": "Algorithm",
        "version": "Version", 
        "performance": "Performance",
        "mae": "MAE",
        "quick_guide": "📖 Quick Guide",
        "how_to_use": "How to use:",
        "step1": "1. Fill in insured person's data",
        "step2": "2. Click 'Calculate Insurance'",
        "step3": "3. View prediction and analysis",
        "important_vars": "Important variables:",
        "smoker_impact": "🚬 **Smoker**: Highest impact on insurance",
        "age_impact": "👤 **Age**: Second highest impact",
        "bmi_impact": "⚖️ **BMI**: Third highest impact",
        
        # Input Form
        "insured_data": "📝 Insured Person Data",
        "age": "👤 Age",
        "age_help": "Age of insured person (18-64 years)",
        "gender": "👥 Gender", 
        "male": "👨 Male",
        "female": "👩 Female",
        "gender_help": "Gender of insured person",
        "smoker": "🚬 Smoker",
        "non_smoker": "🚭 Non-smoker",
        "smoker_yes": "🚬 Smoker",
        "smoker_help": "Smoking status (highest impact on insurance)",
        "bmi": "⚖️ BMI (Body Mass Index)",
        "bmi_help": "Body Mass Index (15.0-55.0)",
        "children": "👶 Number of Children",
        "children_help": "Number of dependent children (0-5)",
        "region": "📍 Region",
        "region_help": "Geographic region",
        "northeast": "🏢 Northeast",
        "northwest": "🏔️ Northwest", 
        "southeast": "🏖️ Southeast",
        "southwest": "🌵 Southwest",
        
        # BMI Categories
        "bmi_category": "BMI Category",
        "underweight": "Underweight",
        "normal_weight": "Normal weight",
        "overweight": "Overweight", 
        "obesity": "Obesity",
        
        # Prediction Button
        "calculate_btn": "🔮 Calculate Insurance",
        "calculating": "Calculating prediction...",
        
        # Results
        "prediction_result": "🔮 Prediction Result",
        "estimated_insurance": "💰 Estimated Insurance",
        "annual_insurance": "Annual health insurance value",
        "monthly": "💳 Monthly",
        "monthly_approx": "Approximate monthly value",
        "processing": "⚡ Processing",
        "processing_time": "Processing time",
        "model": "🤖 Model",
        "algorithm_used": "Algorithm used",
        
        # Risk Analysis
        "risk_analysis": "📊 Risk Analysis",
        "factors_increase": "⚠️ **Factors that increase insurance:**",
        "low_risk_profile": "✅ **Low risk profile** - Few factors that increase insurance",
        "comparison_title": "📈 Comparison with Similar Profiles",
        "your_profile": "Your Profile",
        "non_smokers": "Non-smokers",
        "smokers": "Smokers", 
        "general_average": "General Average",
        "comparison_chart_title": "Insurance Comparison by Category",
        "annual_insurance": "Annual Insurance ($)",
        
        # Risk Factors
        "high_risk_smoker": "🚬 Smoker - HIGH RISK",
        "advanced_age": "👴 Advanced age",
        "high_bmi": "⚖️ High BMI (obesity)",
        "low_bmi": "⚖️ Low BMI (underweight)",
        
        # About Section
        "about_project": "ℹ️ About the Project",
        "objective": "🎯 Objective",
        "objective_text": "Health insurance charges prediction system using advanced Machine Learning techniques.",
        "technology": "🤖 Technology",
        "tech_algorithm": "**Algorithm:** Gradient Boosting (sklearn)",
        "tech_performance": "**Performance:** R² > 0.87, MAE < $2,700", 
        "tech_architecture": "**Architecture:** Advanced feature engineering with 13 features",
        "important_features": "📊 Important Features",
        "feature1": "**Smoker** - Highest impact on insurance",
        "feature2": "**Age** - Second highest impact",
        "feature3": "**BMI** - Third highest impact", 
        "feature4": "**Advanced features** - age_smoker_risk, bmi_smoker_risk, composite_risk_score",
        
        # Error Messages
        "model_unavailable": "❌ Model not available. Please check configuration.",
        "prediction_error": "Error in prediction:",
        "validation_error": "Validation error:",
    },
    
    "pt": {
        # App Config  
        "page_title": "🏥 Preditor de Preço de Convênio Médico",
        "main_header": "🏥 Preditor de Preço de Convênio Médico", 
        "sub_header": "Sistema inteligente usando algoritmo Gradient Boosting",
        
        # Language Toggle
        "language": "🌍 Idioma",
        "lang_en": "🇺🇸 English",
        "lang_pt": "🇧🇷 Português",
        
        # Navigation
        "tab_individual": "🎯 Predição Individual",
        "tab_batch": "📊 Análise em Lote",
        "tab_about": "ℹ️ Sobre",
        
        # Sidebar
        "sidebar_title": "🔧 Informações do Sistema",
        "model_loaded": "✅ Modelo carregado!",
        "model_not_loaded": "❌ Modelo não carregado",
        "model_details": "📊 Detalhes do Modelo",
        "algorithm": "Algoritmo",
        "version": "Versão",
        "performance": "Performance", 
        "mae": "MAE",
        "quick_guide": "📖 Guia Rápido",
        "how_to_use": "Como usar:",
        "step1": "1. Preencha os dados do segurado",
        "step2": "2. Clique em 'Calcular Convênio'",
        "step3": "3. Veja a predição e análise",
        "important_vars": "Variáveis importantes:",
        "smoker_impact": "🚬 **Fumante**: Maior impacto no convênio",
        "age_impact": "👤 **Idade**: Segundo maior impacto", 
        "bmi_impact": "⚖️ **BMI**: Terceiro maior impacto",
        
        # Input Form
        "insured_data": "📝 Dados do Segurado",
        "age": "👤 Idade",
        "age_help": "Idade do segurado (18-64 anos)",
        "gender": "👥 Gênero",
        "male": "👨 Masculino", 
        "female": "👩 Feminino",
        "gender_help": "Gênero do segurado",
        "smoker": "🚬 Fumante",
        "non_smoker": "🚭 Não Fumante",
        "smoker_yes": "🚬 Fumante",
        "smoker_help": "Status de fumante (maior impacto no convênio)",
        "bmi": "⚖️ BMI (Índice de Massa Corporal)",
        "bmi_help": "Índice de Massa Corporal (15.0-55.0)",
        "children": "👶 Número de Filhos",
        "children_help": "Número de filhos dependentes (0-5)",
        "region": "📍 Região",
        "region_help": "Região geográfica",
        "northeast": "🏢 Nordeste",
        "northwest": "🏔️ Noroeste",
        "southeast": "🏖️ Sudeste", 
        "southwest": "🌵 Sudoeste",
        
        # BMI Categories
        "bmi_category": "Categoria BMI",
        "underweight": "Abaixo do peso",
        "normal_weight": "Peso normal",
        "overweight": "Sobrepeso",
        "obesity": "Obesidade",
        
        # Prediction Button
        "calculate_btn": "🔮 Calcular Convênio",
        "calculating": "Calculando predição...",
        
        # Results
        "prediction_result": "🔮 Resultado da Predição",
        "estimated_insurance": "💰 Convênio Estimado",
        "annual_insurance": "Convênio Anual ($)",
        "monthly": "💳 Mensal",
        "monthly_approx": "Valor mensal aproximado",
        "processing": "⚡ Processamento",
        "processing_time": "Tempo de processamento",
        "model": "🤖 Modelo",
        "algorithm_used": "Algoritmo usado",
        
        # Risk Analysis
        "risk_analysis": "📊 Análise de Risco",
        "factors_increase": "⚠️ **Fatores que aumentam o convênio:**",
        "low_risk_profile": "✅ **Perfil de baixo risco** - Poucos fatores que aumentam o convênio",
        "comparison_title": "📈 Comparação com Perfis Similares",
        "your_profile": "Seu Perfil",
        "non_smokers": "Não Fumantes",
        "smokers": "Fumantes", 
        "general_average": "Média Geral",
        "comparison_chart_title": "Comparação de Convênio por Categoria",
        "annual_insurance": "Convênio Anual ($)",
        
        # Risk Factors
        "high_risk_smoker": "🚬 Fumante - RISCO ALTO",
        "advanced_age": "👴 Idade avançada",
        "high_bmi": "⚖️ BMI alto (obesidade)",
        "low_bmi": "⚖️ BMI baixo (abaixo do peso)",
        
        # About Section  
        "about_project": "ℹ️ Sobre o Projeto",
        "objective": "🎯 Objetivo",
        "objective_text": "Sistema de predição de preços de convênio médico usando técnicas avançadas de Machine Learning.",
        "technology": "🤖 Tecnologia",
        "tech_algorithm": "**Algoritmo:** Gradient Boosting (sklearn)",
        "tech_performance": "**Performance:** R² > 0.87, MAE < $2,700",
        "tech_architecture": "**Arquitetura:** Feature engineering avançado com 13 features",
        "important_features": "📊 Features Importantes",
        "feature1": "**Fumante** - Maior impacto no convênio",
        "feature2": "**Idade** - Segundo maior impacto",
        "feature3": "**BMI** - Terceiro maior impacto",
        "feature4": "**Features avançadas** - age_smoker_risk, bmi_smoker_risk, composite_risk_score",
        
        # Error Messages
        "model_unavailable": "❌ Modelo não disponível. Verifique a configuração.",
        "prediction_error": "Erro na predição:",
        "validation_error": "Erro de validação:",
    }
}

def t(key: str, lang: str = "en") -> str:
    """Get translation for key in specified language."""
    return TRANSLATIONS.get(lang, {}).get(key, key)

@st.cache_resource(show_spinner=False, ttl=60)  # TTL curto para forçar refresh
def cached_load_model():
    """Load model with caching - Cache curto para evitar modelos antigos."""
    if USE_LOCAL_MODEL is None:
        return None
    elif USE_LOCAL_MODEL:
        try:
            predictor = load_production_model()
            return {"type": "local", "predictor": predictor}
        except Exception as e:
            st.error(f"Error loading local model: {e}")
            return None
    else:
        try:
            model_data = load_model()
            # VERIFICAR SE MODELO VÁLIDO
            if model_data and model_data.get('model_type') != 'dummy':
                return {"type": "deployment", "model_data": model_data}
            else:
                st.error("❌ Modelo inválido detectado - forçando reload")
                st.cache_resource.clear()  # Limpar cache
                return None
        except Exception as e:
            st.error(f"Error loading deploy model: {e}")
            st.cache_resource.clear()  # Limpar cache em caso de erro
            return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="🏥 Insurance Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Language selection in sidebar
    st.sidebar.title("🌍 Language / Idioma")
    lang = st.sidebar.selectbox(
        "Select language / Selecione o idioma",
        options=["pt", "en"],
        format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English"
    )

    # Main header
    st.markdown(f'<h1 class="main-header">{t("main_header", lang)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2rem;">{t("sub_header", lang)}</p>', unsafe_allow_html=True)

    # Load model
    model_info = cached_load_model()
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.title(t("sidebar_title", lang))
    
    if model_info:
        st.sidebar.success(t("model_loaded", lang))
        
        with st.sidebar.expander(t("model_details", lang)):
            if not USE_LOCAL_MODEL and model_info["type"] == "deployment":
                st.write(f"**{t('algorithm', lang)}:** Gradient Boosting (Advanced)")
                st.write(f"**{t('version', lang)}:** 2.0.0 (Deploy - Same as Local)")
                st.write(f"**Features:** 13 (with feature engineering)")
                st.write(f"**Model Type:** {model_info['model_data'].get('model_type', 'Advanced')}")
            elif USE_LOCAL_MODEL and model_info["type"] == "local":
                st.write(f"**{t('algorithm', lang)}:** Gradient Boosting (Advanced)")
                st.write(f"**{t('version', lang)}:** 2.0.0 (Local System)")
                st.write(f"**Features:** 13 (with feature engineering)")
            else:
                st.write(f"**{t('algorithm', lang)}:** Unknown")
                st.write(f"**{t('version', lang)}:** N/A")
    else:
        st.sidebar.error(t("model_not_loaded", lang))

    # Navigation tabs
    tab1, tab2 = st.tabs([t("tab_individual", lang), t("tab_about", lang)])
    
    with tab1:
        individual_prediction_tab(lang, model_info)
    
    with tab2:
        about_tab(lang)

def individual_prediction_tab(lang: str, model_data):
    """Individual prediction tab."""
    
    # Input form
    st.subheader(t("insured_data", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            t("age", lang),
            min_value=18,
            max_value=64,
            value=25,
            step=1,
            help=t("age_help", lang)
        )
        
        sex = st.selectbox(
            t("gender", lang),
            options=["male", "female"],
            format_func=lambda x: t("male", lang) if x == "male" else t("female", lang),
            help=t("gender_help", lang)
        )
        
        smoker = st.selectbox(
            t("smoker", lang),
            options=["no", "yes"],
            format_func=lambda x: t("non_smoker", lang) if x == "no" else t("smoker_yes", lang),
            help=t("smoker_help", lang)
        )
    
    with col2:
        bmi = st.number_input(
            t("bmi", lang),
            min_value=15.0,
            max_value=55.0,
            value=22.6,
            step=0.1,
            format="%.1f",
            help=t("bmi_help", lang)
        )
        
        children = st.number_input(
            t("children", lang),
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help=t("children_help", lang)
        )
        
        region = st.selectbox(
            t("region", lang),
            options=["northeast", "northwest", "southeast", "southwest"],
            format_func=lambda x: {
                "northeast": t("northeast", lang),
                "northwest": t("northwest", lang),
                "southeast": t("southeast", lang),
                "southwest": t("southwest", lang)
            }[x],
            help=t("region_help", lang)
        )

    # BMI category display
    bmi_category = get_bmi_category(bmi)
    st.info(f"**{t('bmi_category', lang)}:** {bmi_category}")

    # Prediction button
    if st.button(t("calculate_btn", lang), type="primary", use_container_width=True):
        with st.spinner(t("calculating", lang)):
            input_data = {
                'age': int(age),
                'sex': sex,
                'bmi': float(bmi),
                'children': int(children),
                'smoker': smoker,
                'region': region
            }
            
            if model_data:
                if not USE_LOCAL_MODEL and model_data["type"] == "deployment":
                    # Use deployment model system (preferred)
                    try:
                        result = predict_premium(input_data, model_data["model_data"])
                        if result["success"]:
                            show_prediction_results(result, input_data, lang)
                        else:
                            st.error(f"{t('prediction_error', lang)} {result['error']}")
                    except Exception as e:
                        st.error(f"{t('prediction_error', lang)} {e}")
                elif USE_LOCAL_MODEL and model_data["type"] == "local":
                    # Use local model system (fallback)
                    try:
                        result = predict_insurance_premium(**input_data)
                        show_prediction_results_local(result, input_data, lang)
                    except Exception as e:
                        st.error(f"{t('prediction_error', lang)} {e}")
                else:
                    st.error(t("model_unavailable", lang))
            else:
                st.error(t("model_unavailable", lang))

def show_prediction_results_local(result, input_data, lang):
    """Show prediction results using local model system."""
    st.subheader(t("prediction_result", lang))
    
    premium = result['predicted_premium']
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 {t("estimated_insurance", lang)}</h2>
        <h1>${premium:,.2f}</h1>
        <p>{t("annual_insurance", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"💳 {t('monthly', lang)}",
            f"${premium/12:,.2f}",
            help=t("monthly_approx", lang)
        )
    
    with col2:
        processing_time = result.get('processing_time_ms', 0)
        st.metric(
            f"⚡ {t('processing', lang)}",
            f"{processing_time:.1f}ms",
            help=t("processing_time", lang)
        )
    
    with col3:
        st.metric(
            f"🤖 {t('model', lang)}",
            result.get('model_type', 'Gradient Boosting'),
            help=t("algorithm_used", lang)
        )
    
    # Risk analysis
    show_risk_analysis(input_data, premium, lang)

def show_prediction_results(result, input_data, lang):
    """Show prediction results using deployment model."""
    st.subheader(t("prediction_result", lang))
    
    premium = result['predicted_premium']
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 {t("estimated_insurance", lang)}</h2>
        <h1>${premium:,.2f}</h1>
        <p>{t("annual_insurance", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"💳 {t('monthly', lang)}",
            f"${result['monthly_premium']:,.2f}",
            help=t("monthly_approx", lang)
        )
    
    with col2:
        processing_time = result.get('processing_time_ms', 0)
        st.metric(
            f"⚡ {t('processing', lang)}",
            f"{processing_time:.1f}ms",
            help=t("processing_time", lang)
        )
    
    with col3:
        st.metric(
            f"🤖 {t('model', lang)}",
            "Gradient Boosting",
            help=t("algorithm_used", lang)
        )
    
    # Risk analysis
    show_risk_analysis(input_data, premium, lang)

def show_risk_analysis(input_data, premium, lang):
    """Show risk analysis."""
    st.subheader(t("risk_analysis", lang))
    
    # Risk factors
    risk_factors = []
    
    if input_data['smoker'] == 'yes':
        risk_factors.append(t("high_risk_smoker", lang))
    
    if input_data['age'] > 50:
        risk_factors.append(t("advanced_age", lang))
    
    if input_data['bmi'] > 30:
        risk_factors.append(t("high_bmi", lang))
    
    if input_data['bmi'] < 18.5:
        risk_factors.append(t("low_bmi", lang))
    
    if risk_factors:
        st.markdown(f"**{t('factors_increase', lang)}**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.success(t("low_risk_profile", lang))

def about_tab(lang):
    """About tab."""
    st.subheader(t("about_project", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### {t("objective", lang)}
        {t("objective_text", lang)}
        
        ### {t("technology", lang)}
        - {t("tech_algorithm", lang)}
        - {t("tech_performance", lang)}
        - {t("tech_architecture", lang)}
        """)
    
    with col2:
        st.markdown(f"""
        ### {t("important_features", lang)}
        - {t("feature1", lang)}
        - {t("feature2", lang)}
        - {t("feature3", lang)}
        - {t("feature4", lang)}
        """)

def get_bmi_category(bmi):
    """Get BMI category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

def map_region_to_english(region_display, lang):
    """Map region display to English values."""
    region_map = {
        "🏢 Nordeste": "northeast",
        "🏔️ Noroeste": "northwest", 
        "🏖️ Sudeste": "southeast",
        "🌵 Sudoeste": "southwest",
        "🏢 Northeast": "northeast",
        "🏔️ Northwest": "northwest",
        "🏖️ Southeast": "southeast", 
        "🌵 Southwest": "southwest"
    }
    return region_map.get(region_display, region_display)

if __name__ == "__main__":
    main() 