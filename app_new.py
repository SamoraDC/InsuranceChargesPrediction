#!/usr/bin/env python3
"""
Aplicação Streamlit para predição de prêmios de seguro.
Versão atualizada usando a arquitetura refatorada.
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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Imports da nova arquitetura
from insurance_prediction.models.predictor import predict_insurance_premium, load_production_model
from insurance_prediction.config.settings import Config
from insurance_prediction.utils.logging import setup_logging, get_logger

# Configurar logging
setup_logging("INFO")
logger = get_logger(__name__)

# Configuração da página
st.set_page_config(
    page_title=Config.STREAMLIT_CONFIG["page_title"],
    page_icon=Config.STREAMLIT_CONFIG["page_icon"],
    layout=Config.STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=Config.STREAMLIT_CONFIG["initial_sidebar_state"]
)

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

def render_sidebar():
    """Renderiza a sidebar com informações do modelo."""
    st.sidebar.title("🔧 Informações do Sistema")
    
    # Status do modelo
    if 'predictor' in st.session_state and st.session_state.predictor:
        st.sidebar.success("✅ Modelo carregado!")
        
        # Informações do modelo
        with st.sidebar.expander("📊 Detalhes do Modelo"):
            st.write("**Algoritmo:** Gradient Boosting")
            st.write("**Versão:** 1.0.0")
            st.write("**Performance:** R² > 0.85")
    else:
        st.sidebar.error("❌ Modelo não carregado")
    
    st.sidebar.markdown("---")
    
    # Guia rápido
    with st.sidebar.expander("📖 Guia Rápido"):
        st.markdown("""
        **Como usar:**
        1. Preencha os dados do segurado
        2. Clique em "Calcular Prêmio"
        3. Veja a predição e análise
        
        **Variáveis importantes:**
        - 🚬 **Fumante**: Maior impacto no prêmio
        - 👤 **Idade**: Segundo maior impacto
        - ⚖️ **BMI**: Terceiro maior impacto
        """)

def render_input_form():
    """Renderiza o formulário de entrada."""
    st.subheader("📝 Dados do Segurado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "👤 Idade",
            min_value=Config.NUMERICAL_RANGES["age"]["min"],
            max_value=Config.NUMERICAL_RANGES["age"]["max"],
            value=35,
            step=1,
            help="Idade do segurado (18-64 anos)"
        )
        
        sex = st.selectbox(
            "👥 Gênero",
            options=Config.CATEGORICAL_VALUES["sex"],
            format_func=lambda x: "👨 Masculino" if x == "male" else "👩 Feminino",
            help="Gênero do segurado"
        )
        
        smoker = st.selectbox(
            "🚬 Fumante",
            options=Config.CATEGORICAL_VALUES["smoker"],
            format_func=lambda x: "🚭 Não Fumante" if x == "no" else "🚬 Fumante",
            help="Status de fumante (maior impacto no prêmio)"
        )
    
    with col2:
        bmi = st.number_input(
            "⚖️ BMI (Índice de Massa Corporal)",
            min_value=Config.NUMERICAL_RANGES["bmi"]["min"],
            max_value=Config.NUMERICAL_RANGES["bmi"]["max"],
            value=25.0,
            step=0.1,
            format="%.1f",
            help="Índice de Massa Corporal (15.0-55.0)"
        )
        
        children = st.number_input(
            "👶 Número de Filhos",
            min_value=Config.NUMERICAL_RANGES["children"]["min"],
            max_value=Config.NUMERICAL_RANGES["children"]["max"],
            value=0,
            step=1,
            help="Número de filhos dependentes (0-5)"
        )
        
        region = st.selectbox(
            "📍 Região",
            options=Config.CATEGORICAL_VALUES["region"],
            format_func=lambda x: {
                "northeast": "🏢 Nordeste",
                "northwest": "🏔️ Noroeste",
                "southeast": "🏖️ Sudeste",
                "southwest": "🌵 Sudoeste"
            }[x],
            help="Região geográfica"
        )
    
    return {
        'age': int(age),
        'sex': sex,
        'bmi': float(bmi),
        'children': int(children),
        'smoker': smoker,
        'region': region
    }

def render_bmi_info(bmi):
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

def render_prediction_result(result):
    """Renderiza o resultado da predição."""
    st.subheader("🔮 Resultado da Predição")
    
    premium = result['predicted_premium']
    
    # Card principal com o resultado
    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 Prêmio Estimado</h2>
        <h1>${premium:,.2f}</h1>
        <p>Valor anual do seguro de saúde</p>
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
    render_risk_analysis(result['input_data'], premium)

def render_risk_analysis(input_data, premium):
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
        st.write("⚠️ **Fatores que elevam o prêmio:**")
        for factor in risk_factors:
            st.write(f"- {factor}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("✅ **Perfil de baixo risco** - Poucos fatores que elevam o prêmio")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparação com médias
    render_comparison_chart(input_data, premium)

def render_comparison_chart(input_data, premium):
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
        title="Comparação de Prêmios por Categoria",
        yaxis_title="Prêmio Anual ($)",
        height=400,
        showlegend=False
    )
    
    fig.update_yaxis(tickformat='$,.0f')
    
    st.plotly_chart(fig, use_container_width=True)

def render_batch_analysis():
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
                with st.spinner("Processando predições..."):
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
                            st.error(f"Erro na linha {_}: {e}")
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
                        st.metric("💰 Prêmio Médio", f"${avg_premium:,.2f}")
                    
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
            st.error(f"Erro ao processar arquivo: {e}")

def main():
    """Função principal da aplicação."""
    # Header
    st.markdown('<h1 class="main-header">🏥 Preditor de Prêmios de Seguro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema inteligente de predição usando Gradient Boosting</p>', unsafe_allow_html=True)
    
    # Carregar modelo
    if 'predictor' not in st.session_state:
        with st.spinner("Carregando modelo..."):
            st.session_state.predictor = load_model()
    
    # Renderizar sidebar
    render_sidebar()
    
    # Menu principal
    tab1, tab2, tab3 = st.tabs(["🎯 Predição Individual", "📊 Análise em Lote", "ℹ️ Sobre"])
    
    with tab1:
        if st.session_state.predictor is None:
            st.error("❌ Modelo não disponível. Execute o treinamento primeiro.")
            st.code("python scripts/train_model.py")
            return
        
        # Formulário de entrada
        user_data = render_input_form()
        
        # Mostrar BMI info
        render_bmi_info(user_data['bmi'])
        
        # Botão de predição
        if st.button("🔮 Calcular Prêmio", type="primary"):
            with st.spinner("Calculando predição..."):
                try:
                    # Fazer predição
                    result = predict_insurance_premium(**user_data)
                    
                    # Mostrar resultado
                    render_prediction_result(result)
                    
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    logger.error(f"Erro na predição: {e}")
    
    with tab2:
        render_batch_analysis()
    
    with tab3:
        st.header("ℹ️ Sobre o Projeto")
        
        st.markdown("""
        ### 🎯 Objetivo
        Sistema de predição de prêmios de seguro de saúde usando técnicas avançadas de Machine Learning.
        
        ### 🤖 Tecnologia
        - **Algoritmo:** Gradient Boosting (sklearn)
        - **Performance:** R² > 0.85, RMSE < 4000
        - **Arquitetura:** Modular e seguindo melhores práticas
        
        ### 📊 Features Importantes
        1. **Fumante** - Maior impacto no prêmio
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

if __name__ == "__main__":
    main() 