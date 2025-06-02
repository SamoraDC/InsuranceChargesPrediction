import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.predict import InsurancePredictor
from src.config import PROJECT_CONFIG
from app.components.input_form import render_input_form
from app.components.results_display import render_results
from app.components.charts import render_charts
from app.utils.helpers import load_sample_data, format_currency

# Configuração da página
st.set_page_config(
    page_title="🏥 Preditor de Prêmios de Seguro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🏥 Preditor de Prêmios de Seguro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema inteligente para predição de custos de seguro de saúde baseado em características pessoais</p>', unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("🔧 Painel de Controle")
    
    # Seleção da página
    page = st.sidebar.selectbox(
        "Escolha uma funcionalidade:",
        ["🎯 Predição Individual", "📊 Análise em Lote", "📈 Dashboard Analytics", "ℹ️ Sobre o Projeto"]
    )
    
    # Inicializar o preditor
    @st.cache_resource
    def load_predictor():
        try:
            return InsurancePredictor()
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {str(e)}")
            return None
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Não foi possível carregar o modelo. Verifique se os arquivos estão no local correto.")
        return
    
    # Roteamento das páginas
    if page == "🎯 Predição Individual":
        render_individual_prediction(predictor)
    elif page == "📊 Análise em Lote":
        render_batch_analysis(predictor)
    elif page == "📈 Dashboard Analytics":
        render_analytics_dashboard(predictor)
    else:
        render_about_page()

def render_individual_prediction(predictor):
    """Renderiza a página de predição individual"""
    st.header("🎯 Predição Individual de Prêmio")
    
    # Duas colunas: formulário e resultados
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Dados do Segurado")
        
        # Formulário simples sem submit interno
        age = st.number_input(
            "👤 Idade",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Idade do segurado em anos"
        )
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            sex = st.selectbox(
                "👥 Gênero",
                options=["male", "female"],
                format_func=lambda x: "Masculino" if x == "male" else "Feminino",
                help="Gênero do segurado"
            )
            
            smoker = st.selectbox(
                "🚬 Fumante",
                options=["no", "yes"],
                format_func=lambda x: "Não" if x == "no" else "Sim",
                help="Indica se o segurado é fumante"
            )
        
        with col1_2:
            bmi = st.number_input(
                "⚖️ BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1,
                format="%.1f",
                help="Índice de Massa Corporal"
            )
            
            children = st.number_input(
                "👶 Filhos",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Número de filhos dependentes"
            )
        
        region = st.selectbox(
            "📍 Região",
            options=["northeast", "northwest", "southeast", "southwest"],
            format_func=lambda x: {
                "northeast": "Nordeste",
                "northwest": "Noroeste", 
                "southeast": "Sudeste",
                "southwest": "Sudoeste"
            }[x],
            help="Região geográfica do segurado"
        )
        
        # Criar dados do usuário
        user_data = {
            'age': int(age),
            'sex': sex,
            'bmi': float(bmi),
            'children': int(children),
            'smoker': smoker,
            'region': region
        }
        
        # Botão de predição
        if st.button("🔮 Calcular Prêmio", key="predict_btn", type="primary"):
            with st.spinner("Calculando predição..."):
                try:
                    # Fazer predição
                    prediction_result = predictor.predict_single(user_data)
                    
                    # Salvar resultado na sessão
                    st.session_state.prediction_result = prediction_result
                    st.session_state.user_data = user_data
                    
                except Exception as e:
                    st.error(f"Erro na predição: {str(e)}")
    
    with col2:
        st.subheader("📊 Resultado da Predição")
        
        # Mostrar resultados se disponíveis
        if hasattr(st.session_state, 'prediction_result'):
            render_results(st.session_state.prediction_result, st.session_state.user_data)
        else:
            st.info("💡 Preencha os dados à esquerda e clique em 'Calcular Prêmio' para ver os resultados.")
            
            # Mostrar exemplo
            st.markdown("### 📋 Exemplo de Uso")
            example_data = {
                'age': 35,
                'sex': 'male',
                'bmi': 25.5,
                'children': 2,
                'smoker': 'no',
                'region': 'northeast'
            }
            
            example_result = predictor.predict_single(example_data)
            st.markdown("**Dados de exemplo:**")
            for key, value in example_data.items():
                st.write(f"- **{key.title()}**: {value}")
            
            st.markdown(f"**Prêmio estimado**: {format_currency(example_result['prediction'])}")

def render_batch_analysis(predictor):
    """Renderiza a página de análise em lote"""
    st.header("📊 Análise em Lote")
    
    st.markdown("""
    Upload um arquivo CSV com múltiplos segurados para análise em lote.
    
    **Formato esperado do CSV:**
    - age (int): Idade do segurado
    - sex (str): Gênero ('male' ou 'female')
    - bmi (float): Índice de massa corporal
    - children (int): Número de filhos
    - smoker (str): Fumante ('yes' ou 'no')
    - region (str): Região ('northeast', 'northwest', 'southeast', 'southwest')
    """)
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "📁 Escolha um arquivo CSV",
        type=['csv'],
        help="Upload um arquivo CSV com os dados dos segurados"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
            
            # Mostrar prévia dos dados
            st.subheader("👀 Prévia dos Dados")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Botão para processar
            if st.button("🚀 Processar Predições em Lote", type="primary"):
                with st.spinner("Processando predições..."):
                    try:
                        # Converter DataFrame para lista de dicionários
                        data_list = df.to_dict('records')
                        
                        # Fazer predições
                        results = predictor.predict_batch(data_list)
                        
                        # Adicionar predições ao DataFrame original
                        df['predicted_charges'] = [r['prediction'] for r in results]
                        
                        # Mostrar estatísticas
                        st.subheader("📈 Estatísticas das Predições")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 Total de Predições", len(df))
                        
                        with col2:
                            avg_charge = df['predicted_charges'].mean()
                            st.metric("💰 Prêmio Médio", format_currency(avg_charge))
                        
                        with col3:
                            min_charge = df['predicted_charges'].min()
                            st.metric("📉 Menor Prêmio", format_currency(min_charge))
                        
                        with col4:
                            max_charge = df['predicted_charges'].max()
                            st.metric("📈 Maior Prêmio", format_currency(max_charge))
                        
                        # Gráficos de análise
                        render_batch_charts(df)
                        
                        # Tabela de resultados
                        st.subheader("📋 Resultados Detalhados")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download dos resultados
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download dos Resultados (CSV)",
                            data=csv,
                            file_name="predicoes_seguro.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Erro no processamento: {str(e)}")
                        
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
    
    # Mostrar dados de exemplo
    st.subheader("📋 Exemplo de Formato de Dados")
    example_df = pd.DataFrame({
        'age': [25, 35, 45, 55],
        'sex': ['male', 'female', 'male', 'female'],
        'bmi': [22.5, 28.0, 25.5, 30.2],
        'children': [0, 2, 1, 3],
        'smoker': ['no', 'no', 'yes', 'no'],
        'region': ['northeast', 'southwest', 'southeast', 'northwest']
    })
    st.dataframe(example_df, use_container_width=True)

def render_batch_charts(df):
    """Renderiza gráficos para análise em lote"""
    st.subheader("📊 Análise Visual dos Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma das predições
        fig_hist = px.histogram(
            df, 
            x='predicted_charges',
            title="Distribuição dos Prêmios Preditos",
            labels={'predicted_charges': 'Prêmio Predito ($)', 'count': 'Frequência'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot por região
        fig_box = px.box(
            df,
            x='region',
            y='predicted_charges',
            title="Prêmios por Região",
            labels={'predicted_charges': 'Prêmio Predito ($)', 'region': 'Região'},
            color='region'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Scatter plot: Idade vs Prêmio
    fig_scatter = px.scatter(
        df,
        x='age',
        y='predicted_charges',
        color='smoker',
        size='bmi',
        title="Relação: Idade vs Prêmio (tamanho = BMI, cor = fumante)",
        labels={'predicted_charges': 'Prêmio Predito ($)', 'age': 'Idade'},
        hover_data=['children', 'region']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def render_analytics_dashboard(predictor):
    """Renderiza dashboard de analytics"""
    st.header("📈 Dashboard Analytics")
    
    st.markdown("""
    Explore padrões e insights sobre predições de seguros com base em diferentes características.
    """)
    
    # Simulação de dados para análise
    st.subheader("🎛️ Simulador de Cenários")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Configure os parâmetros para análise:**")
        
        # Controles para simulação
        age_range = st.slider("Faixa Etária", 18, 70, (25, 55))
        bmi_range = st.slider("Faixa de BMI", 15.0, 50.0, (20.0, 35.0))
        include_smokers = st.checkbox("Incluir Fumantes", value=True)
        selected_regions = st.multiselect(
            "Regiões",
            ['northeast', 'northwest', 'southeast', 'southwest'],
            default=['northeast', 'southwest']
        )
        
        if st.button("🔄 Gerar Análise", type="primary"):
            # Gerar dados sintéticos para análise
            with st.spinner("Gerando análise..."):
                analysis_data = generate_analysis_data(
                    predictor, age_range, bmi_range, include_smokers, selected_regions
                )
                st.session_state.analysis_data = analysis_data
    
    with col2:
        if hasattr(st.session_state, 'analysis_data'):
            render_analytics_charts(st.session_state.analysis_data)
        else:
            st.info("👈 Configure os parâmetros à esquerda e clique em 'Gerar Análise'")

def generate_analysis_data(predictor, age_range, bmi_range, include_smokers, regions):
    """Gera dados sintéticos para análise"""
    import random
    
    data = []
    n_samples = 200
    
    for _ in range(n_samples):
        age = random.randint(age_range[0], age_range[1])
        bmi = random.uniform(bmi_range[0], bmi_range[1])
        sex = random.choice(['male', 'female'])
        children = random.randint(0, 5)
        smoker = random.choice(['yes', 'no']) if include_smokers else 'no'
        region = random.choice(regions) if regions else 'northeast'
        
        # Fazer predição
        sample_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        
        result = predictor.predict_single(sample_data)
        sample_data['predicted_charges'] = result['prediction']
        
        data.append(sample_data)
    
    return pd.DataFrame(data)

def render_analytics_charts(df):
    """Renderiza gráficos de analytics"""
    st.subheader("📊 Insights dos Dados")
    
    # Métricas resumo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_charge = df['predicted_charges'].mean()
        st.metric("💰 Prêmio Médio", format_currency(avg_charge))
    
    with col2:
        smoker_impact = df.groupby('smoker')['predicted_charges'].mean()
        if 'yes' in smoker_impact.index and 'no' in smoker_impact.index:
            impact = smoker_impact['yes'] - smoker_impact['no']
            st.metric("🚬 Impacto do Tabagismo", format_currency(impact))
    
    with col3:
        age_corr = df[['age', 'predicted_charges']].corr().iloc[0,1]
        st.metric("👴 Correlação Idade", f"{age_corr:.3f}")
    
    with col4:
        bmi_corr = df[['bmi', 'predicted_charges']].corr().iloc[0,1]
        st.metric("⚖️ Correlação BMI", f"{bmi_corr:.3f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap de correlação
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            title="Matriz de Correlação",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Violin plot por gênero e fumante
        fig_violin = px.violin(
            df,
            x='sex',
            y='predicted_charges',
            color='smoker',
            title="Distribuição de Prêmios por Gênero e Tabagismo",
            box=True
        )
        fig_violin.update_layout(height=400)
        st.plotly_chart(fig_violin, use_container_width=True)

def render_about_page():
    """Renderiza a página sobre o projeto"""
    st.header("ℹ️ Sobre o Projeto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Objetivo
        
        Este sistema utiliza **Machine Learning** para predizer prêmios de seguro de saúde com base em características pessoais dos segurados.
        
        ## 🔬 Metodologia
        
        ### Dados
        - **Dataset**: Insurance charges dataset
        - **Registros**: 1.338 segurados
        - **Features**: Idade, sexo, BMI, filhos, tabagismo, região
        
        ### Modelos Utilizados
        - **Ridge Regression** (Modelo Principal)
        - Random Forest
        - Linear Regression
        - Gradient Boosting
        
        ### Performance do Modelo
        - **R² Score**: 0.8856 (88.56% de variância explicada)
        - **MAE**: ~$2,700
        - **RMSE**: ~$6,100
        
        ## 🎨 Funcionalidades
        
        - **Predição Individual**: Calcule o prêmio para um segurado
        - **Análise em Lote**: Processe múltiplos segurados de uma vez
        - **Dashboard Analytics**: Explore padrões e insights
        - **Intervalos de Confiança**: Estimativas com incerteza
        - **Feature Importance**: Entenda quais fatores mais influenciam
        
        ## 🏗️ Arquitetura
        
        ```
        📊 Streamlit App
        ├── 🤖 ML Pipeline (Ridge + Random Forest)
        ├── 🔧 Feature Engineering
        ├── 📈 Visualizações Interativas
        └── 💾 Artefatos Salvos
        ```
        
        ## 📈 Principais Insights
        
        1. **Tabagismo** é o fator mais importante (+$20,000 em média)
        2. **Idade** tem forte correlação positiva com prêmios
        3. **BMI elevado** aumenta significativamente os custos
        4. **Interações** entre fatores são importantes (ex: idade × BMI)
        
        ## 🔧 Tecnologias
        
        - **Python**: Linguagem principal
        - **Streamlit**: Interface web
        - **Scikit-learn**: Machine Learning
        - **Plotly**: Visualizações interativas
        - **Pandas/NumPy**: Manipulação de dados
        - **MLflow**: Tracking de experimentos
        
        """)
    
    with col2:
        st.markdown("### 📊 Estatísticas do Projeto")
        
        # Métricas do projeto
        metrics_data = {
            "Linhas de Código": "~3,500",
            "Módulos Python": "8",
            "Testes Implementados": "6",
            "Modelos Treinados": "8",
            "Features Criadas": "15",
            "Gráficos Gerados": "12"
        }
        
        for metric, value in metrics_data.items():
            st.markdown(f"**{metric}**: {value}")
        
        st.markdown("---")
        
        st.markdown("### 🎓 Desenvolvido para")
        st.markdown("**FIAP - Tech Challenge 01**")
        st.markdown("*Pós-graduação em Data Science*")
        
        st.markdown("---")
        
        st.markdown("### 📧 Contato")
        st.markdown("Para dúvidas ou sugestões sobre o projeto.")
        
        # Botões de ação
        if st.button("📁 Ver Código no GitHub", key="github"):
            st.info("🔗 Link do repositório seria disponibilizado aqui")
        
        if st.button("📊 Download do Dataset", key="dataset"):
            st.info("📥 Download do dataset seria iniciado aqui")

if __name__ == "__main__":
    main() 