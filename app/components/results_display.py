import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app.utils.helpers import format_currency

def render_results(prediction_result, user_data):
    """
    Renderiza os resultados da predição de forma visual
    
    Args:
        prediction_result (dict): Resultado da predição
        user_data (dict): Dados do usuário
    """
    
    # Card principal com a predição
    prediction_value = prediction_result['prediction']
    
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🎯 Prêmio Estimado</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">{format_currency(prediction_value)}</h1>
        <p style="font-size: 1.1rem;">Por ano</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas adicionais
    if 'confidence_interval' in prediction_result:
        ci_lower = prediction_result['confidence_interval']['lower']
        ci_upper = prediction_result['confidence_interval']['upper']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📉 Valor Mínimo",
                format_currency(ci_lower),
                delta=format_currency(ci_lower - prediction_value),
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "🎯 Predição",
                format_currency(prediction_value),
                help="Valor predito pelo modelo"
            )
        
        with col3:
            st.metric(
                "📈 Valor Máximo", 
                format_currency(ci_upper),
                delta=format_currency(ci_upper - prediction_value),
                delta_color="normal"
            )
        
        # Gráfico de intervalo de confiança
        render_confidence_interval_chart(prediction_result)
    
    # Feature importance se disponível
    if 'feature_importance' in prediction_result:
        render_feature_importance(prediction_result['feature_importance'])
    
    # Comparações e insights
    render_insights(prediction_result, user_data)
    
    # Breakdown do valor
    render_value_breakdown(prediction_result, user_data)

def render_confidence_interval_chart(prediction_result):
    """
    Renderiza gráfico do intervalo de confiança
    
    Args:
        prediction_result (dict): Resultado da predição
    """
    st.subheader("📊 Intervalo de Confiança (95%)")
    
    prediction = prediction_result['prediction']
    ci_lower = prediction_result['confidence_interval']['lower']
    ci_upper = prediction_result['confidence_interval']['upper']
    
    # Criar gráfico de gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prêmio Estimado ($)"},
        delta={'reference': (ci_lower + ci_upper) / 2},
        gauge={
            'axis': {'range': [None, ci_upper * 1.1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, ci_lower], 'color': "lightgray"},
                {'range': [ci_lower, ci_upper], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"🎯 **Interpretação**: Existe 95% de confiança de que o prêmio real estará entre {format_currency(ci_lower)} e {format_currency(ci_upper)}.")

def render_feature_importance(feature_importance):
    """
    Renderiza gráfico de importância das features
    
    Args:
        feature_importance (dict): Importância das features
    """
    st.subheader("🔍 Fatores mais Importantes")
    
    if feature_importance:
        # Converter para DataFrame e ordenar
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        # Criar gráfico de barras horizontal
        fig = px.bar(
            importance_df.tail(10),  # Top 10 features
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Fatores que Influenciam o Prêmio",
            labels={'Importance': 'Importância Relativa', 'Feature': 'Fator'},
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicação textual
        top_feature = importance_df.iloc[-1]
        st.info(f"💡 **Insight**: O fator mais importante para seu prêmio é **{top_feature['Feature']}** com importância de {top_feature['Importance']:.3f}.")

def render_insights(prediction_result, user_data):
    """
    Renderiza insights personalizados baseados nos dados do usuário
    
    Args:
        prediction_result (dict): Resultado da predição
        user_data (dict): Dados do usuário
    """
    st.subheader("💡 Insights Personalizados")
    
    insights = []
    prediction = prediction_result['prediction']
    
    # Insights baseados em fumante
    if user_data['smoker'] == 'yes':
        insights.append({
            'type': 'warning',
            'title': '🚬 Impacto do Tabagismo',
            'message': 'Ser fumante aumenta significativamente o prêmio do seguro. Parar de fumar pode reduzir seus custos em até $20,000 por ano.'
        })
    else:
        insights.append({
            'type': 'success',
            'title': '✅ Benefício de Não Fumar',
            'message': 'Por não ser fumante, você está economizando significativamente no prêmio do seguro!'
        })
    
    # Insights baseados no BMI
    bmi = user_data['bmi']
    if bmi >= 30:
        insights.append({
            'type': 'warning',
            'title': '⚖️ IMC Elevado',
            'message': f'Seu BMI de {bmi:.1f} está na faixa de obesidade, o que pode aumentar o prêmio. Manter um peso saudável pode reduzir custos.'
        })
    elif bmi >= 25:
        insights.append({
            'type': 'info',
            'title': '⚖️ IMC no Limite',
            'message': f'Seu BMI de {bmi:.1f} está na faixa de sobrepeso. Manter um estilo de vida saudável pode ajudar a controlar custos.'
        })
    else:
        insights.append({
            'type': 'success',
            'title': '✅ IMC Saudável',
            'message': f'Seu BMI de {bmi:.1f} está na faixa saudável, o que contribui para um prêmio mais baixo!'
        })
    
    # Insights baseados na idade
    age = user_data['age']
    if age >= 50:
        insights.append({
            'type': 'info',
            'title': '👴 Fator Idade',
            'message': f'Com {age} anos, a idade é um fator que influencia o prêmio. Considere opções de seguro com cobertura preventiva.'
        })
    elif age <= 25:
        insights.append({
            'type': 'success',
            'title': '👶 Vantagem da Juventude',
            'message': f'Sua idade de {age} anos contribui para um prêmio mais baixo. É um bom momento para contratar um seguro!'
        })
    
    # Exibir insights
    for insight in insights:
        if insight['type'] == 'success':
            st.success(f"**{insight['title']}**: {insight['message']}")
        elif insight['type'] == 'warning':
            st.warning(f"**{insight['title']}**: {insight['message']}")
        else:
            st.info(f"**{insight['title']}**: {insight['message']}")

def render_value_breakdown(prediction_result, user_data):
    """
    Renderiza breakdown aproximado do valor do prêmio
    
    Args:
        prediction_result (dict): Resultado da predição
        user_data (dict): Dados do usuário
    """
    st.subheader("📋 Breakdown Estimado do Prêmio")
    
    prediction = prediction_result['prediction']
    
    # Estimativas aproximadas baseadas nos dados (simplificado)
    base_cost = 5000  # Custo base estimado
    
    # Ajustes baseados nos fatores
    age_factor = max(0, (user_data['age'] - 18) * 100)
    bmi_factor = max(0, (user_data['bmi'] - 25) * 200) if user_data['bmi'] > 25 else 0
    smoker_factor = 15000 if user_data['smoker'] == 'yes' else 0
    children_factor = user_data['children'] * 500
    
    # Ajustar para somar ao valor predito (aproximação)
    total_factors = age_factor + bmi_factor + smoker_factor + children_factor
    remaining = max(0, prediction - base_cost - total_factors)
    
    breakdown_data = {
        'Componente': [
            'Custo Base',
            'Fator Idade',
            'Fator BMI',
            'Fator Tabagismo',
            'Fator Filhos',
            'Outros Fatores'
        ],
        'Valor': [
            base_cost,
            age_factor,
            bmi_factor,
            smoker_factor,
            children_factor,
            remaining
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    breakdown_df = breakdown_df[breakdown_df['Valor'] > 0]  # Remover valores zero
    
    # Gráfico de pizza
    fig = px.pie(
        breakdown_df,
        values='Valor',
        names='Componente',
        title="Composição Estimada do Prêmio",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("**Detalhamento:**")
    for _, row in breakdown_df.iterrows():
        percentage = (row['Valor'] / prediction) * 100
        st.write(f"• **{row['Componente']}**: {format_currency(row['Valor'])} ({percentage:.1f}%)")
    
    st.markdown(f"**Total**: {format_currency(prediction)}")

def render_comparison_chart(prediction, user_data):
    """
    Renderiza gráfico de comparação com perfis similares
    
    Args:
        prediction (float): Valor da predição
        user_data (dict): Dados do usuário
    """
    st.subheader("📊 Comparação com Perfis Similares")
    
    # Dados simulados para comparação (em um caso real, viria do dataset)
    similar_profiles = {
        'Seu Perfil': prediction,
        'Média Geral': 13270,  # Valor médio do dataset
        'Não Fumantes': 8434,   # Média de não fumantes
        'Fumantes': 32050,      # Média de fumantes
        f'Idade {user_data["age"]}-{user_data["age"]+5}': prediction * 0.9  # Estimativa para faixa etária
    }
    
    # Filtrar comparações relevantes
    if user_data['smoker'] == 'yes':
        del similar_profiles['Não Fumantes']
    else:
        del similar_profiles['Fumantes']
    
    # Criar gráfico de barras
    profiles_df = pd.DataFrame(
        list(similar_profiles.items()),
        columns=['Perfil', 'Prêmio']
    )
    
    fig = px.bar(
        profiles_df,
        x='Perfil',
        y='Prêmio',
        title="Comparação de Prêmios por Perfil",
        color='Prêmio',
        color_continuous_scale='RdYlBu_r'
    )
    
    # Destacar o perfil do usuário
    fig.update_traces(
        marker_color=['#FF6B6B' if x == 'Seu Perfil' else '#4ECDC4' for x in profiles_df['Perfil']]
    )
    
    st.plotly_chart(fig, use_container_width=True) 