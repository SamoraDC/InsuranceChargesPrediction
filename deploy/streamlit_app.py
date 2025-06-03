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
from model_utils import load_model, predict_premium, get_risk_analysis

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
        "tech_architecture": "**Architecture:** Independent and optimized for cloud deployment",
        "important_features": "📊 Important Features",
        "feature1": "**Smoker** - Highest impact on insurance",
        "feature2": "**Age** - Second highest impact",
        "feature3": "**BMI** - Third highest impact", 
        "feature4": "**Interactions** - age_smoker, bmi_smoker",
        
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
        "annual_insurance": "Valor anual do convênio médico",
        "monthly": "💳 Mensal",
        "monthly_approx": "Valor mensal aproximado",
        "processing": "⚡ Processamento",
        "processing_time": "Tempo de processamento",
        "model": "🤖 Modelo",
        "algorithm_used": "Algoritmo utilizado",
        
        # Risk Analysis  
        "risk_analysis": "📊 Análise de Risco",
        "factors_increase": "⚠️ **Fatores que elevam o convênio:**",
        "low_risk_profile": "✅ **Perfil de baixo risco** - Poucos fatores que elevam o convênio",
        "comparison_title": "📈 Comparação com Perfis Similares",
        "your_profile": "Seu Perfil",
        "non_smokers": "Não Fumantes", 
        "smokers": "Fumantes",
        "general_average": "Média Geral",
        "comparison_chart_title": "Comparação de Convênios por Categoria",
        "annual_insurance": "Convênio Anual ($)",
        
        # Risk Factors
        "high_risk_smoker": "🚬 Fumante - ALTO RISCO",
        "advanced_age": "👴 Idade avançada",
        "high_bmi": "⚖️ BMI elevado (obesidade)",
        "low_bmi": "⚖️ BMI baixo (abaixo do peso)",
        
        # About Section
        "about_project": "ℹ️ Sobre o Projeto",
        "objective": "🎯 Objetivo",
        "objective_text": "Sistema de predição de Preço de convênios médicos usando técnicas avançadas de Machine Learning.",
        "technology": "🤖 Tecnologia",
        "tech_algorithm": "**Algoritmo:** Gradient Boosting (sklearn)",
        "tech_performance": "**Performance:** R² > 0.87, MAE < $2,700",
        "tech_architecture": "**Arquitetura:** Independente e otimizada para deploy em nuvem",
        "important_features": "📊 Features Importantes",
        "feature1": "**Fumante** - Maior impacto no convênio",
        "feature2": "**Idade** - Segundo maior impacto",
        "feature3": "**BMI** - Terceiro maior impacto",
        "feature4": "**Interações** - age_smoker, bmi_smoker",
        
        # Error Messages
        "model_unavailable": "❌ Modelo não disponível. Verifique a configuração.",
        "prediction_error": "Erro na predição:",
        "validation_error": "Erro de validação:",
    }
}

def t(key: str, lang: str = "en") -> str:
    """Get translation for given key and language."""
    return TRANSLATIONS.get(lang, {}).get(key, key)

# =============================================================================
# MAIN APPLICATION / APLICAÇÃO PRINCIPAL
# =============================================================================

@st.cache_resource
def cached_load_model():
    """Cache model loading for better performance."""
    return load_model()

def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="🏥 Insurance Charges Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for language
    if 'language' not in st.session_state:
        st.session_state.language = 'pt'  # Default to Portuguese
    
    # Language toggle in sidebar
    with st.sidebar:
        st.subheader(t("language", st.session_state.language))
        lang_option = st.radio(
            "Select Language",
            ["🇧🇷 Português", "🇺🇸 English"],
            index=0 if st.session_state.language == 'pt' else 1,
            key="lang_radio",
            label_visibility="collapsed"
        )
        
        # Update language
        if "Português" in lang_option:
            st.session_state.language = 'pt'
        else:
            st.session_state.language = 'en'
    
    lang = st.session_state.language
    
    # Main header
    st.title(t("main_header", lang))
    st.markdown(f"**{t('sub_header', lang)}**")
    st.markdown("---")
    
    # Load model
    model_data = cached_load_model()
    
    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.subheader(t("sidebar_title", lang))
        
        if model_data:
            st.success(t("model_loaded", lang))
            
            # Model details
            with st.expander(t("model_details", lang)):
                metrics = model_data.get('metrics', {})
                st.write(f"**{t('algorithm', lang)}:** Gradient Boosting")
                st.write(f"**{t('version', lang)}:** 1.0.0")
                st.write(f"**R²:** {metrics.get('r2', 0.88):.3f}")
                st.write(f"**{t('mae', lang)}:** ${metrics.get('mae', 2650):.0f}")
        else:
            st.error(t("model_not_loaded", lang))
        
        # Quick guide
        st.markdown("---")
        with st.expander(t("quick_guide", lang)):
            st.markdown(f"**{t('how_to_use', lang)}**")
            st.markdown(f"- {t('step1', lang)}")
            st.markdown(f"- {t('step2', lang)}")
            st.markdown(f"- {t('step3', lang)}")
            
            st.markdown(f"**{t('important_vars', lang)}**")
            st.markdown(f"- {t('smoker_impact', lang)}")
            st.markdown(f"- {t('age_impact', lang)}")
            st.markdown(f"- {t('bmi_impact', lang)}")
    
    # Main tabs
    tab1, tab2 = st.tabs([
        t("tab_individual", lang),
        t("tab_about", lang)
    ])
    
    # Tab 1: Individual Prediction
    with tab1:
        individual_prediction_tab(lang, model_data)
    
    # Tab 2: About
    with tab2:
        about_tab(lang)

def individual_prediction_tab(lang: str, model_data):
    """Individual prediction tab content."""
    
    if not model_data:
        st.error(t("model_unavailable", lang))
        return
    
    # Input form
    with st.container():
        st.subheader(t("insured_data", lang))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input(
                t("age", lang),
                min_value=18, max_value=64, value=35,
                help=t("age_help", lang)
            )
            
            gender = st.selectbox(
                t("gender", lang),
                [t("male", lang), t("female", lang)],
                help=t("gender_help", lang)
            )
            
        with col2:
            smoker_status = st.selectbox(
                t("smoker", lang),
                [t("non_smoker", lang), t("smoker_yes", lang)],
                help=t("smoker_help", lang)
            )
            
            bmi = st.number_input(
                t("bmi", lang),
                min_value=15.0, max_value=55.0, value=25.0, step=0.1,
                help=t("bmi_help", lang)
            )
            
        with col3:
            children = st.number_input(
                t("children", lang),
                min_value=0, max_value=5, value=0,
                help=t("children_help", lang)
            )
            
            region = st.selectbox(
                t("region", lang),
                [t("northeast", lang), t("northwest", lang), 
                 t("southeast", lang), t("southwest", lang)],
                help=t("region_help", lang)
            )
        
        # BMI category
        bmi_category = get_bmi_category(bmi)
        bmi_label = t(bmi_category, lang)
        st.info(f"**{t('bmi_category', lang)}:** {bmi_label}")
        
        # Prediction button
        st.markdown("---")
        
        if st.button(t("calculate_btn", lang), type="primary", use_container_width=True):
            
            # Prepare data
            input_data = {
                "age": age,
                "sex": "male" if t("male", lang) in gender else "female",
                "bmi": bmi,
                "children": children,
                "smoker": "yes" if t("smoker_yes", lang) in smoker_status else "no",
                "region": map_region_to_english(region, lang)
            }
            
            # Show loading
            with st.spinner(t("calculating", lang)):
                time.sleep(0.5)  # Brief delay for UX
                
                # Make prediction
                result = predict_premium(input_data, model_data)
                
                if result["success"]:
                    show_prediction_results(result, input_data, lang)
                else:
                    st.error(f"{t('prediction_error', lang)} {result.get('error', 'Unknown error')}")

def show_prediction_results(result, input_data, lang):
    """Display prediction results with charts."""
    
    premium = result["predicted_premium"]
    monthly = result["monthly_premium"]
    processing_time = result["processing_time_ms"]
    
    # Results header
    st.markdown("---")
    st.subheader(t("prediction_result", lang))
    
    # Main metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label=f"💰 {t('estimated_insurance', lang)}",
            value=f"${premium:,.2f}",
            help=t("annual_insurance", lang)
        )
    
    with col2:
        st.metric(
            label=f"💳 {t('monthly', lang)}",
            value=f"${monthly:,.2f}",
            help=t("monthly_approx", lang)
        )
    
    # Performance metrics
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric(
            label=f"⚡ {t('processing', lang)}",
            value=f"{processing_time:.0f}ms",
            help=t("processing_time", lang)
        )
    
    with col4:
        st.metric(
            label=f"🤖 {t('model', lang)}",
            value="Gradient Boosting",
            help=t("algorithm_used", lang)
        )
    
    # Risk analysis
    show_risk_analysis(input_data, premium, lang)

def show_risk_analysis(input_data, premium, lang):
    """Show risk analysis and comparison charts."""
    
    st.markdown("---")
    st.subheader(t("risk_analysis", lang))
    
    # Risk factors
    risk_factors = []
    
    if input_data["smoker"] == "yes":
        risk_factors.append(t("high_risk_smoker", lang))
    
    if input_data["age"] >= 50:
        risk_factors.append(t("advanced_age", lang))
    
    if input_data["bmi"] >= 30:
        risk_factors.append(t("high_bmi", lang))
    elif input_data["bmi"] < 18.5:
        risk_factors.append(t("low_bmi", lang))
    
    if risk_factors:
        st.markdown(t("factors_increase", lang))
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.success(t("low_risk_profile", lang))
    
    # Comparison chart
    st.markdown("---")
    st.subheader(t("comparison_title", lang))
    
    # Sample comparison data (simplified)
    comparison_data = {
        t("your_profile", lang): premium,
        t("non_smokers", lang): premium * 0.6 if input_data["smoker"] == "yes" else premium,
        t("smokers", lang): premium * 1.8 if input_data["smoker"] == "no" else premium,
        t("general_average", lang): 13270  # Approximate average
    }
    
    # Create comparison chart
    fig = px.bar(
        x=list(comparison_data.keys()),
        y=list(comparison_data.values()),
        title=t("comparison_chart_title", lang),
        labels={'x': '', 'y': t("annual_insurance", lang)},
        color=list(comparison_data.values()),
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def about_tab(lang):
    """About tab content."""
    
    st.subheader(t("about_project", lang))
    
    # Objective
    st.markdown(f"### {t('objective', lang)}")
    st.markdown(t("objective_text", lang))
    
    # Technology
    st.markdown(f"### {t('technology', lang)}")
    st.markdown(f"- {t('tech_algorithm', lang)}")
    st.markdown(f"- {t('tech_performance', lang)}")
    st.markdown(f"- {t('tech_architecture', lang)}")
    
    # Important features
    st.markdown(f"### {t('important_features', lang)}")
    st.markdown(f"1. {t('feature1', lang)}")
    st.markdown(f"2. {t('feature2', lang)}")
    st.markdown(f"3. {t('feature3', lang)}")
    st.markdown(f"4. {t('feature4', lang)}")

def get_bmi_category(bmi):
    """Get BMI category."""
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal_weight"
    elif bmi < 30:
        return "overweight"
    else:
        return "obesity"

def map_region_to_english(region_display, lang):
    """Map displayed region to english value."""
    if lang == "pt":
        region_map = {
            "🏢 Nordeste": "northeast",
            "🏔️ Noroeste": "northwest", 
            "🏖️ Sudeste": "southeast",
            "🌵 Sudoeste": "southwest"
        }
    else:
        region_map = {
            "🏢 Northeast": "northeast",
            "🏔️ Northwest": "northwest",
            "🏖️ Southeast": "southeast", 
            "🌵 Southwest": "southwest"
        }
    
    return region_map.get(region_display, "northeast")

if __name__ == "__main__":
    main() 