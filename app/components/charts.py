import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def render_charts():
    """
    Renderiza gráficos diversos para análise
    """
    pass

def create_prediction_gauge(prediction, min_val=0, max_val=50000):
    """
    Cria gráfico gauge para mostrar predição
    
    Args:
        prediction (float): Valor da predição
        min_val (float): Valor mínimo do gauge
        max_val (float): Valor máximo do gauge
        
    Returns:
        plotly.graph_objects.Figure: Gráfico gauge
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prêmio Estimado ($)"},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.3], 'color': "lightgreen"},
                {'range': [max_val*0.3, max_val*0.7], 'color': "yellow"},
                {'range': [max_val*0.7, max_val], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    return fig

def create_feature_importance_chart(feature_importance, top_n=10):
    """
    Cria gráfico de barras para importância das features
    
    Args:
        feature_importance (dict): Importância das features
        top_n (int): Número de features mais importantes a mostrar
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de barras
    """
    if not feature_importance:
        return None
    
    # Converter para DataFrame e ordenar
    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n} Fatores Mais Importantes",
        labels={'Importance': 'Importância', 'Feature': 'Fator'},
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    return fig

def create_comparison_radar_chart(user_data, avg_data=None):
    """
    Cria gráfico radar para comparar perfil do usuário com média
    
    Args:
        user_data (dict): Dados do usuário
        avg_data (dict): Dados médios para comparação
        
    Returns:
        plotly.graph_objects.Figure: Gráfico radar
    """
    if avg_data is None:
        # Valores médios aproximados do dataset
        avg_data = {
            'age': 39,
            'bmi': 30.7,
            'children': 1.1
        }
    
    categories = ['Idade', 'BMI', 'Filhos']
    
    # Normalizar valores para escala 0-100
    user_values = [
        (user_data['age'] / 100) * 100,  # Idade normalizada
        (user_data['bmi'] / 50) * 100,   # BMI normalizado
        (user_data['children'] / 5) * 100  # Filhos normalizado
    ]
    
    avg_values = [
        (avg_data['age'] / 100) * 100,
        (avg_data['bmi'] / 50) * 100,
        (avg_data['children'] / 5) * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Seu Perfil',
        line_color='rgb(255, 100, 100)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Média Geral',
        line_color='rgb(100, 100, 255)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Comparação do Seu Perfil vs Média Geral"
    )
    
    return fig

def create_distribution_histogram(data, column, title=None, bins=30):
    """
    Cria histograma de distribuição
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        column (str): Nome da coluna para plotar
        title (str): Título do gráfico
        bins (int): Número de bins
        
    Returns:
        plotly.graph_objects.Figure: Histograma
    """
    if title is None:
        title = f"Distribuição de {column}"
    
    fig = px.histogram(
        data,
        x=column,
        nbins=bins,
        title=title,
        labels={column: column.title()},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title=column.title(),
        yaxis_title="Frequência"
    )
    
    return fig

def create_correlation_heatmap(data, title="Matriz de Correlação"):
    """
    Cria heatmap de correlação
    
    Args:
        data (pd.DataFrame): DataFrame com dados numéricos
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
    # Selecionar apenas colunas numéricas
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title=title,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )
    
    fig.update_layout(
        width=600,
        height=500
    )
    
    return fig

def create_box_plot_by_category(data, numeric_col, category_col, title=None):
    """
    Cria box plot agrupado por categoria
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        numeric_col (str): Coluna numérica
        category_col (str): Coluna categórica
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Box plot
    """
    if title is None:
        title = f"{numeric_col.title()} por {category_col.title()}"
    
    fig = px.box(
        data,
        x=category_col,
        y=numeric_col,
        title=title,
        color=category_col
    )
    
    fig.update_layout(
        xaxis_title=category_col.title(),
        yaxis_title=numeric_col.title()
    )
    
    return fig

def create_scatter_plot_3d(data, x_col, y_col, z_col, color_col=None, size_col=None, title=None):
    """
    Cria scatter plot 3D
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        x_col (str): Coluna para eixo X
        y_col (str): Coluna para eixo Y
        z_col (str): Coluna para eixo Z
        color_col (str): Coluna para cor
        size_col (str): Coluna para tamanho
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot 3D
    """
    if title is None:
        title = f"Relação 3D: {x_col} vs {y_col} vs {z_col}"
    
    fig = px.scatter_3d(
        data,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=data.columns
    )
    
    fig.update_layout(height=600)
    return fig

def create_violin_plot(data, x_col, y_col, title=None):
    """
    Cria violin plot
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        x_col (str): Coluna categórica (eixo X)
        y_col (str): Coluna numérica (eixo Y)
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Violin plot
    """
    if title is None:
        title = f"Distribuição de {y_col} por {x_col}"
    
    fig = px.violin(
        data,
        x=x_col,
        y=y_col,
        title=title,
        box=True,
        color=x_col
    )
    
    return fig

def create_sunburst_chart(data, path_cols, values_col, title=None):
    """
    Cria gráfico sunburst (hierárquico)
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        path_cols (list): Lista de colunas para hierarquia
        values_col (str): Coluna com valores
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Sunburst chart
    """
    if title is None:
        title = "Análise Hierárquica"
    
    fig = px.sunburst(
        data,
        path=path_cols,
        values=values_col,
        title=title
    )
    
    return fig

def create_animated_scatter(data, x_col, y_col, animation_frame, color_col=None, size_col=None, title=None):
    """
    Cria scatter plot animado
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        x_col (str): Coluna para eixo X
        y_col (str): Coluna para eixo Y
        animation_frame (str): Coluna para animação
        color_col (str): Coluna para cor
        size_col (str): Coluna para tamanho
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot animado
    """
    if title is None:
        title = f"Evolução: {x_col} vs {y_col}"
    
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        animation_frame=animation_frame,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=data.columns
    )
    
    return fig

def create_parallel_coordinates(data, color_col, title=None):
    """
    Cria gráfico de coordenadas paralelas
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        color_col (str): Coluna para colorir
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de coordenadas paralelas
    """
    if title is None:
        title = "Análise de Coordenadas Paralelas"
    
    # Selecionar apenas colunas numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    fig = px.parallel_coordinates(
        data,
        color=color_col,
        dimensions=numeric_cols,
        title=title
    )
    
    return fig

def create_treemap(data, path_cols, values_col, title=None):
    """
    Cria treemap
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        path_cols (list): Lista de colunas para hierarquia
        values_col (str): Coluna com valores
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Treemap
    """
    if title is None:
        title = "Análise Hierárquica (TreeMap)"
    
    fig = px.treemap(
        data,
        path=path_cols,
        values=values_col,
        title=title
    )
    
    return fig 