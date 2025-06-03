#!/usr/bin/env python3
"""
🏥 Insurance Charges Predictor - Streamlit App
🏥 Preditor de Preço de Convênio Médico - Aplicativo Streamlit

Bilingual Insurance Charges Prediction App
Aplicativo Bilíngue de Predição de Preço de Convênio Médico
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent if '__file__' in globals() else Path('.')
sys.path.insert(0, str(project_root / "src"))

# Imports da nova arquitetura
from insurance_prediction.models.predictor import predict_insurance_premium, load_production_model
from insurance_prediction.config.settings import Config
from insurance_prediction.utils.logging import setup_logging, get_logger

# =============================================================================
# TRANSLATIONS / TRADUÇÕES
# =============================================================================

TRANSLATIONS = {
    "en": {
        # App Config
        "page_title": "🏥 Insurance Charges Predictor",
        "main_header": "🏥 Insurance Charges Predictor",
        "sub_header": "AI-powered system using Gradient Boosting algorithm",
        
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
        "calculate_btn": "🔮 Calculate insurance",
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
        
        # Batch Analysis
        "batch_analysis": "📊 Batch Analysis",
        "batch_info": "💡 Upload a CSV file with multiple insured persons for mass analysis.",
        "download_template": "📥 Download CSV Template",
        "choose_csv": "Choose a CSV file",
        "csv_help": "File must contain columns: age, sex, bmi, children, smoker, region",
        "data_loaded": "📋 Loaded Data",
        "process_batch": "🔮 Process Batch Predictions",
        "processing_predictions": "Processing predictions...",
        "results": "📊 Results",
        "total_records": "📊 Total Records",
        "average_insurance": "💰 Average Insurance",
        "total_revenue": "💵 Total Revenue",
        "download_results": "📥 Download Results",
        
        # About Section
        "about_project": "ℹ️ About the Project",
        "objective": "🎯 Objective",
        "objective_text": "Health insurance charges prediction system using advanced Machine Learning techniques.",
        "technology": "🤖 Technology",
        "tech_algorithm": "**Algorithm:** Gradient Boosting (sklearn)",
        "tech_performance": "**Performance:** R² > 0.85, RMSE < 4000", 
        "tech_architecture": "**Architecture:** Modular following best practices",
        "important_features": "📊 Important Features",
        "feature1": "**Smoker** - Highest impact on insurance",
        "feature2": "**Age** - Second highest impact",
        "feature3": "**BMI** - Third highest impact", 
        "feature4": "**Interactions** - age_smoker_risk, bmi_smoker_risk",
        "how_to_use_about": "🔧 How to Use",
        "usage1": "1. Fill in the insured person's data",
        "usage2": "2. Run the application", 
        "usage3": "3. Fill in data and get predictions",
        "quality_metrics": "📈 Quality Metrics",
        "high_precision": "- High precision in predictions",
        "fast_response": "- Response time < 50ms",
        "robust_validation": "- Robust input validation",
        
        # Error Messages
        "model_unavailable": "❌ Model not available. Please check configuration.",
        "prediction_error": "Error in prediction:",
        "file_error": "Error processing file:",
        "validation_error": "Validation error:",
        
        # File Processing
        "error_row": "Error in row",
        "filename_template": "template_insured.csv",
        "filename_results": "insurance_predictions.csv"
    },
    
    "pt": {
        # App Config  
        "page_title": "🏥 Preditor de Preço de Convênio Médico",
        "main_header": "🏥 Preditor de Preço de Convênio Médico", 
        "sub_header": "Sistema inteligente usando algoritmo Gradient Boosting",
        
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
        
        # Batch Analysis
        "batch_analysis": "📊 Análise em Lote",
        "batch_info": "💡 Faça upload de um arquivo CSV com múltiplos segurados para análise em massa.",
        "download_template": "📥 Baixar Template CSV",
        "choose_csv": "Escolha um arquivo CSV",
        "csv_help": "Arquivo deve conter colunas: age, sex, bmi, children, smoker, region",
        "data_loaded": "📋 Dados Carregados",
        "process_batch": "🔮 Processar Predições em Lote",
        "processing_predictions": "Processando predições...",
        "results": "📊 Resultados",
        "total_records": "📊 Total de Registros",
        "average_insurance": "💰 Convênio Médio", 
        "total_revenue": "💵 Receita Total",
        "download_results": "📥 Baixar Resultados",
        
        # About Section
        "about_project": "ℹ️ Sobre o Projeto",
        "objective": "🎯 Objetivo",
        "objective_text": "Sistema de predição de Preço de convênios médicos usando técnicas avançadas de Machine Learning.",
        "technology": "🤖 Tecnologia",
        "tech_algorithm": "**Algoritmo:** Gradient Boosting (sklearn)",
        "tech_performance": "**Performance:** R² > 0.85, RMSE < 4000",
        "tech_architecture": "**Arquitetura:** Modular seguindo melhores práticas",
        "important_features": "📊 Features Importantes",
        "feature1": "**Fumante** - Maior impacto no convênio",
        "feature2": "**Idade** - Segundo maior impacto",
        "feature3": "**BMI** - Terceiro maior impacto",
        "feature4": "**Interações** - age_smoker_risk, bmi_smoker_risk",
        "how_to_use_about": "🔧 Como Usar", 
        "usage1": "1. Preencha os dados do segurado",
        "usage2": "2. Execute a aplicação",
        "usage3": "3. Preencha dados e obtenha predições",
        "quality_metrics": "📈 Métricas de Qualidade",
        "high_precision": "- Precisão alta em predições",
        "fast_response": "- Tempo de resposta < 50ms",
        "robust_validation": "- Validação robusta de entrada",
        
        # Error Messages
        "model_unavailable": "❌ Modelo não disponível. Execute o treinamento primeiro.",
        "prediction_error": "Erro na predição:",
        "file_error": "Erro ao processar arquivo:",
        "validation_error": "Erro de validação:",
        
        # File Processing
        "error_row": "Erro na linha",
        "filename_template": "template_segurados.csv",
        "filename_results": "predicoes_seguro.csv"
    }
}

def t(key: str, lang: str = "en") -> str:
    """Get translation for given key and language."""
    return TRANSLATIONS.get(lang, {}).get(key, key)

# Configurar logging
setup_logging("INFO")
logger = get_logger(__name__)

# Configuração da página
st.set_page_config(
    page_title="🏥 Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE AND LANGUAGE SETUP / CONFIGURAÇÃO DE ESTADO E IDIOMA
# =============================================================================

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'pt'  # Default to Portuguese

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-box {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF3CD;
        border: 1px solid #FFECB5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Carrega o modelo treinado."""
    try:
        predictor = load_production_model()
        logger.info("Modelo carregado com sucesso!")
        return predictor
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        st.error(f"Erro ao carregar modelo: {e}")
        return None

def render_sidebar(lang):
    """Renderiza a sidebar com informações do modelo."""
    st.sidebar.title(t("sidebar_title", lang))
    
    # Status do modelo
    if 'predictor' in st.session_state and st.session_state.predictor:
        st.sidebar.success(t("model_loaded", lang))
        
        # Informações do modelo
        with st.sidebar.expander(t("model_details", lang)):
            st.write(f"**{t('algorithm', lang)}:** Gradient Boosting")
            st.write(f"**{t('version', lang)}:** 1.0.0")
            st.write(f"**{t('performance', lang)}:** R² > 0.85")
    else:
        st.sidebar.error(t("model_not_loaded", lang))
    
    st.sidebar.markdown("---")
    
    # Guia rápido
    with st.sidebar.expander(t("quick_guide", lang)):
        st.markdown(f"""
        **{t('how_to_use', lang)}**
        - {t('step1', lang)}
        - {t('step2', lang)}
        - {t('step3', lang)}
        
        **{t('important_vars', lang)}**
        - {t('smoker_impact', lang)}
        - {t('age_impact', lang)}
        - {t('bmi_impact', lang)}
        """)

def render_input_form(lang):
    """Renderiza o formulário de entrada."""
    st.subheader(t("insured_data", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            t("age", lang),
            min_value=18,
            max_value=64,
            value=35,
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
            value=25.0,
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
    
    return {
        'age': int(age),
        'sex': sex,
        'bmi': float(bmi),
        'children': int(children),
        'smoker': smoker,
        'region': region
    }

def render_bmi_info(bmi, lang):
    """Renderiza informações sobre BMI."""
    if bmi < 18.5:
        category = "Abaixo do peso"
        color = "#3498db"
    elif bmi < 25:
        category = "Peso normal"
        color = "#2ecc71"
    elif bmi < 30:
        category = "Sobrepeso"
        color = "#f39c12"
    else:
        category = "Obesidade"
        color = "#e74c3c"
    
    st.markdown(f"""
    <div class="metric-box">
        <strong>BMI: {bmi:.1f}</strong><br>
        <span style="color: {color};">📊 {category}</span>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_result(result, lang):
    """Renderiza o resultado da predição."""
    st.subheader("🔮 Resultado da Predição")
    
    premium = result['predicted_premium']
    
    # Card principal com o resultado
    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 Convênio Estimado</h2>
        <h1>${premium:,.2f}</h1>
        <p>Valor anual do convênio médico</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas adicionais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💳 Mensal",
            f"${premium/12:,.2f}",
            help="Valor mensal aproximado"
        )
    
    with col2:
        processing_time = result.get('processing_time_ms', 0)
        st.metric(
            "⚡ Processamento",
            f"{processing_time:.1f}ms",
            help="Tempo de processamento"
        )
    
    with col3:
        st.metric(
            "🤖 Modelo",
            result.get('model_type', 'N/A'),
            help="Algoritmo utilizado"
        )
    
    # Análise de risco
    render_risk_analysis(result['input_data'], premium, lang)

def render_risk_analysis(input_data, premium, lang):
    """Renderiza análise de risco."""
    st.subheader("📊 Análise de Risco")
    
    # Fatores de risco
    risk_factors = []
    
    if input_data['smoker'] == 'yes':
        risk_factors.append("🚬 Fumante - ALTO RISCO")
    
    if input_data['age'] > 50:
        risk_factors.append("👴 Idade avançada")
    
    if input_data['bmi'] > 30:
        risk_factors.append("⚖️ BMI elevado (obesidade)")
    
    if input_data['bmi'] < 18.5:
        risk_factors.append("⚖️ BMI baixo (abaixo do peso)")
    
    if risk_factors:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("⚠️ **Fatores que elevam o convênio:**")
        for factor in risk_factors:
            st.write(f"- {factor}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("✅ **Perfil de baixo risco** - Poucos fatores que elevam o convênio")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparação com médias
    render_comparison_chart(input_data, premium, lang)

def render_comparison_chart(input_data, premium, lang):
    """Renderiza gráfico de comparação."""
    st.subheader("📈 Comparação com Perfis Similares")
    
    # Criar dados de comparação (simulados baseados no perfil)
    categories = ['Seu Perfil', 'Não Fumantes', 'Fumantes', 'Média Geral']
    
    # Estimativas baseadas no conhecimento do modelo
    if input_data['smoker'] == 'yes':
        values = [premium, premium * 0.3, premium * 0.95, premium * 0.65]
    else:
        values = [premium, premium * 1.05, premium * 3.2, premium * 1.8]
    
    colors = ['#FF6B6B', '#3498db', '#e74c3c', '#95a5a6']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Comparação de Convênios por Categoria",
        yaxis_title="Convênio Anual ($)",
        yaxis=dict(tickformat='$,.0f'),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_batch_analysis(lang):
    """Renderiza análise em lote."""
    st.header("📊 Análise em Lote")
    
    st.info("💡 Faça upload de um arquivo CSV com múltiplos segurados para análise em massa.")
    
    # Template para download
    if st.button("📥 Baixar Template CSV"):
        template_data = {
            'age': [35, 45, 28],
            'sex': ['male', 'female', 'male'],
            'bmi': [25.0, 30.5, 22.1],
            'children': [2, 1, 0],
            'smoker': ['no', 'yes', 'no'],
            'region': ['northeast', 'southeast', 'northwest']
        }
        
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        
        st.download_button(
            label="Baixar template.csv",
            data=csv,
            file_name="template_segurados.csv",
            mime="text/csv"
        )
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type=['csv'],
        help="Arquivo deve conter colunas: age, sex, bmi, children, smoker, region"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Dados Carregados")
            st.dataframe(df.head())
            
            if st.button("🔮 Processar Predições em Lote"):
                with st.spinner("Processando predições..." if lang == 'pt' else "Processing predictions..."):
                    # Processar predições para cada linha
                    predictions = []
                    
                    for _, row in df.iterrows():
                        try:
                            result = predict_insurance_premium(
                                age=int(row['age']),
                                sex=row['sex'],
                                bmi=float(row['bmi']),
                                children=int(row['children']),
                                smoker=row['smoker'],
                                region=row['region']
                            )
                            predictions.append(result['predicted_premium'])
                        except Exception as e:
                            st.error(f"{t('error_row', lang)} {_}: {e}")
                            predictions.append(None)
                    
                    # Adicionar predições ao DataFrame
                    df['predicted_premium'] = predictions
                    
                    # Mostrar resultados
                    st.subheader("📊 Resultados")
                    st.dataframe(df)
                    
                    # Estatísticas
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 Total de Registros", len(df))
                    
                    with col2:
                        avg_premium = df['predicted_premium'].mean()
                        st.metric("💰 Convênio Médio", f"${avg_premium:,.2f}")
                    
                    with col3:
                        total_revenue = df['predicted_premium'].sum()
                        st.metric("💵 Receita Total", f"${total_revenue:,.2f}")
                    
                    # Download dos resultados
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Resultados",
                        data=csv_result,
                        file_name="predicoes_seguro.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"{t('file_error', lang)} {e}")

def render_about_tab(lang):
    """Renderiza a seção sobre o projeto."""
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Sistema de predição de Preço de convênios médicos usando técnicas avançadas de Machine Learning.
    
    ### 🤖 Tecnologia
    - **Algoritmo:** Gradient Boosting (sklearn)
    - **Performance:** R² > 0.85, RMSE < 4000
    - **Arquitetura:** Modular e seguindo melhores práticas
    
    ### 📊 Features Importantes
    1. **Fumante** - Maior impacto no convênio
    2. **Idade** - Segundo maior impacto
    3. **BMI** - Terceiro maior impacto
    4. **Interações** - age_smoker_risk, bmi_smoker_risk
    
    ### 🔧 Como Usar
    1. Treine o modelo: `python scripts/train_model.py`
    2. Execute a aplicação: `streamlit run app_new.py`
    3. Preencha os dados e obtenha predições
    
    ### 📈 Métricas de Qualidade
    - Precisão alta em predições
    - Tempo de resposta < 50ms
    - Validação robusta de entrada
    """)

def main():
    """Função principal da aplicação."""
    
    # Language toggle in sidebar
    with st.sidebar:
        st.subheader("🌍 Idioma / Language")
        lang_option = st.radio(
            "",
            ["🇧🇷 Português", "🇺🇸 English"],
            index=0 if st.session_state.language == 'pt' else 1,
            key="lang_radio"
        )
        
        # Update language
        if "Português" in lang_option:
            st.session_state.language = 'pt'
        else:
            st.session_state.language = 'en'
    
    lang = st.session_state.language
    
    # Header
    st.markdown(f'<h1 class="main-header">{t("main_header", lang)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t("sub_header", lang)}</p>', unsafe_allow_html=True)
    
    # Carregar modelo
    if 'predictor' not in st.session_state:
        with st.spinner("Carregando modelo..." if lang == 'pt' else "Loading model..."):
            st.session_state.predictor = load_model()
    
    # Renderizar sidebar
    render_sidebar(lang)
    
    # Menu principal
    tab1, tab2, tab3 = st.tabs([
        t("tab_individual", lang),
        t("tab_batch", lang), 
        t("tab_about", lang)
    ])
    
    with tab1:
        if st.session_state.predictor is None:
            st.error(t("model_unavailable", lang))
            return
        
        # Formulário de entrada
        user_data = render_input_form(lang)
        
        # Mostrar BMI info
        render_bmi_info(user_data['bmi'], lang)
        
        # Botão de predição
        if st.button(t("calculate_btn", lang), type="primary"):
            with st.spinner(t("calculating", lang)):
                try:
                    # Fazer predição
                    result = predict_insurance_premium(**user_data)
                    
                    # Mostrar resultado
                    render_prediction_result(result, lang)
                    
                except Exception as e:
                    st.error(f"{t('prediction_error', lang)} {e}")
                    logger.error(f"Erro na predição: {e}")
    
    with tab2:
        render_batch_analysis(lang)
    
    with tab3:
        render_about_tab(lang)

if __name__ == "__main__":
    main() 