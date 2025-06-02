import pandas as pd
import numpy as np
import streamlit as st
import pickle
import json
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def format_currency(value, currency="$"):
    """
    Formata valor como moeda
    
    Args:
        value (float): Valor a ser formatado
        currency (str): Símbolo da moeda
        
    Returns:
        str: Valor formatado como moeda
    """
    if value is None or np.isnan(value):
        return f"{currency}0"
    
    return f"{currency}{value:,.2f}"

def format_percentage(value, decimals=1):
    """
    Formata valor como porcentagem
    
    Args:
        value (float): Valor a ser formatado (ex: 0.85 para 85%)
        decimals (int): Número de casas decimais
        
    Returns:
        str: Valor formatado como porcentagem
    """
    if value is None or np.isnan(value):
        return "0%"
    
    return f"{value*100:.{decimals}f}%"

def load_sample_data():
    """
    Carrega dados de exemplo para demonstração
    
    Returns:
        pd.DataFrame: DataFrame com dados de exemplo
    """
    sample_data = {
        'age': [25, 35, 45, 55, 30, 40, 50, 60],
        'sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'bmi': [22.5, 28.0, 25.5, 30.2, 24.1, 26.8, 29.3, 31.5],
        'children': [0, 2, 1, 3, 1, 2, 0, 4],
        'smoker': ['no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
        'region': ['northeast', 'southwest', 'southeast', 'northwest', 
                  'northeast', 'southwest', 'southeast', 'northwest']
    }
    
    return pd.DataFrame(sample_data)

def get_model_info():
    """
    Retorna informações sobre o modelo carregado
    
    Returns:
        dict: Informações do modelo
    """
    try:
        model_info_path = Path(__file__).parent.parent.parent / "models" / "model_info.pkl"
        
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        return model_info
    except Exception:
        # Informações padrão se não conseguir carregar
        return {
            'best_model_name': 'Ridge Regression',
            'best_score': 0.8856,
            'feature_count': 15,
            'training_size': 1069
        }

def validate_input_data(data):
    """
    Valida dados de entrada do usuário
    
    Args:
        data (dict): Dados do usuário
        
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Validar idade
    if 'age' in data:
        if not isinstance(data['age'], (int, float)) or data['age'] < 18 or data['age'] > 100:
            errors.append("Idade deve ser um número entre 18 e 100")
    
    # Validar BMI
    if 'bmi' in data:
        if not isinstance(data['bmi'], (int, float)) or data['bmi'] < 10 or data['bmi'] > 60:
            errors.append("BMI deve ser um número entre 10 e 60")
    
    # Validar children
    if 'children' in data:
        if not isinstance(data['children'], int) or data['children'] < 0 or data['children'] > 10:
            errors.append("Número de filhos deve ser um inteiro entre 0 e 10")
    
    # Validar sex
    if 'sex' in data:
        if data['sex'] not in ['male', 'female']:
            errors.append("Sexo deve ser 'male' ou 'female'")
    
    # Validar smoker
    if 'smoker' in data:
        if data['smoker'] not in ['yes', 'no']:
            errors.append("Fumante deve ser 'yes' ou 'no'")
    
    # Validar region
    if 'region' in data:
        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if data['region'] not in valid_regions:
            errors.append(f"Região deve ser uma das: {', '.join(valid_regions)}")
    
    return len(errors) == 0, errors

def create_download_link(data, filename, file_format='csv'):
    """
    Cria link de download para dados
    
    Args:
        data (pd.DataFrame): Dados para download
        filename (str): Nome do arquivo
        file_format (str): Formato do arquivo ('csv', 'json', 'excel')
        
    Returns:
        bytes: Dados para download
    """
    if file_format.lower() == 'csv':
        return data.to_csv(index=False).encode('utf-8')
    elif file_format.lower() == 'json':
        return data.to_json(orient='records', indent=2).encode('utf-8')
    elif file_format.lower() == 'excel':
        # Para Excel, precisaríamos de openpyxl
        return data.to_csv(index=False).encode('utf-8')
    else:
        return data.to_csv(index=False).encode('utf-8')

def get_risk_category(prediction, thresholds=None):
    """
    Categoriza o risco baseado na predição
    
    Args:
        prediction (float): Valor predito
        thresholds (dict): Limites customizados
        
    Returns:
        tuple: (categoria, cor, emoji)
    """
    if thresholds is None:
        thresholds = {
            'low': 8000,
            'medium': 20000,
            'high': 35000
        }
    
    if prediction <= thresholds['low']:
        return ("Baixo Risco", "#28a745", "🟢")
    elif prediction <= thresholds['medium']:
        return ("Risco Médio", "#ffc107", "🟡")
    elif prediction <= thresholds['high']:
        return ("Alto Risco", "#fd7e14", "🟠")
    else:
        return ("Risco Muito Alto", "#dc3545", "🔴")

def calculate_savings_potential(user_data, prediction):
    """
    Calcula potencial de economia baseado nas características do usuário
    
    Args:
        user_data (dict): Dados do usuário
        prediction (float): Predição atual
        
    Returns:
        dict: Informações sobre economia potencial
    """
    savings = {}
    
    # Economia por parar de fumar
    if user_data.get('smoker') == 'yes':
        savings['quit_smoking'] = {
            'amount': 20000,  # Estimativa baseada em dados históricos
            'description': "Parar de fumar pode reduzir significativamente seus prêmios"
        }
    
    # Economia por reduzir BMI (se aplicável)
    bmi = user_data.get('bmi', 25)
    if bmi >= 30:  # Obesidade
        potential_reduction = (bmi - 29) * 500  # Estimativa
        savings['weight_loss'] = {
            'amount': potential_reduction,
            'description': f"Reduzir BMI de {bmi:.1f} para 29 pode economizar"
        }
    elif bmi >= 25:  # Sobrepeso
        potential_reduction = (bmi - 24.9) * 300  # Estimativa menor
        savings['weight_management'] = {
            'amount': potential_reduction,
            'description': f"Manter BMI saudável (< 25) pode economizar"
        }
    
    return savings

def get_statistical_summary(data, column):
    """
    Calcula resumo estatístico de uma coluna
    
    Args:
        data (pd.DataFrame): DataFrame com dados
        column (str): Nome da coluna
        
    Returns:
        dict: Resumo estatístico
    """
    if column not in data.columns:
        return {}
    
    series = data[column]
    
    if pd.api.types.is_numeric_dtype(series):
        return {
            'count': series.count(),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'q25': series.quantile(0.25),
            'median': series.median(),
            'q75': series.quantile(0.75),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
    else:
        return {
            'count': series.count(),
            'unique': series.nunique(),
            'top': series.mode().iloc[0] if not series.mode().empty else None,
            'freq': series.value_counts().iloc[0] if not series.empty else 0
        }

def create_comparison_data(user_data, prediction):
    """
    Cria dados de comparação com perfis similares
    
    Args:
        user_data (dict): Dados do usuário
        prediction (float): Predição do usuário
        
    Returns:
        pd.DataFrame: DataFrame com comparações
    """
    # Dados de comparação simulados (em caso real, viria de análise do dataset)
    comparisons = []
    
    # Perfil do usuário
    comparisons.append({
        'Perfil': 'Seu Perfil',
        'Prêmio': prediction,
        'Categoria': 'Usuário'
    })
    
    # Média geral
    comparisons.append({
        'Perfil': 'Média Geral',
        'Prêmio': 13270,  # Valor médio aproximado do dataset
        'Categoria': 'Geral'
    })
    
    # Por status de fumante
    if user_data.get('smoker') == 'yes':
        comparisons.append({
            'Perfil': 'Fumantes (Média)',
            'Prêmio': 32050,
            'Categoria': 'Fumante'
        })
    else:
        comparisons.append({
            'Perfil': 'Não Fumantes (Média)',
            'Prêmio': 8434,
            'Categoria': 'Não Fumante'
        })
    
    # Por faixa etária
    age = user_data.get('age', 35)
    if age < 30:
        age_avg = 8500
        age_label = "Jovens (18-29)"
    elif age < 40:
        age_avg = 11000
        age_label = "Adultos Jovens (30-39)"
    elif age < 50:
        age_avg = 15000
        age_label = "Adultos (40-49)"
    elif age < 60:
        age_avg = 20000
        age_label = "Adultos Maduros (50-59)"
    else:
        age_avg = 25000
        age_label = "Idosos (60+)"
    
    comparisons.append({
        'Perfil': age_label,
        'Prêmio': age_avg,
        'Categoria': 'Faixa Etária'
    })
    
    return pd.DataFrame(comparisons)

def load_app_config():
    """
    Carrega configurações da aplicação
    
    Returns:
        dict: Configurações da aplicação
    """
    default_config = {
        'app_title': "Preditor de Prêmios de Seguro",
        'app_version': "1.0.0",
        'max_prediction_value': 100000,
        'confidence_level': 0.95,
        'currency_symbol': "$",
        'supported_regions': ['northeast', 'northwest', 'southeast', 'southwest'],
        'theme_colors': {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D'
        }
    }
    
    # Tentar carregar configuração customizada
    try:
        config_path = Path(__file__).parent.parent.parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
    except Exception:
        pass  # Usar configuração padrão
    
    return default_config

def generate_insights_text(user_data, prediction_result):
    """
    Gera insights textuais baseados nos dados e predição
    
    Args:
        user_data (dict): Dados do usuário
        prediction_result (dict): Resultado da predição
        
    Returns:
        list: Lista de insights
    """
    insights = []
    prediction = prediction_result['prediction']
    
    # Insight sobre valor total
    risk_category, _, risk_emoji = get_risk_category(prediction)
    insights.append(f"{risk_emoji} Seu prêmio estimado de {format_currency(prediction)} está na categoria de {risk_category.lower()}.")
    
    # Insight sobre idade
    age = user_data.get('age', 0)
    if age < 25:
        insights.append("🎯 Sua idade jovem contribui para um prêmio mais baixo. É um ótimo momento para contratar um seguro!")
    elif age > 50:
        insights.append("📈 Com o avanço da idade, os prêmios tendem a aumentar. Considere planos preventivos.")
    
    # Insight sobre BMI
    bmi = user_data.get('bmi', 25)
    if bmi < 18.5:
        insights.append("⚖️ Seu BMI está abaixo do peso ideal. Consulte um médico para orientações nutricionais.")
    elif bmi > 30:
        insights.append("🏃‍♂️ Manter um peso saudável pode significar economia significativa nos prêmios.")
    elif 25 <= bmi <= 30:
        insights.append("💪 Você está na faixa de sobrepeso. Pequenas mudanças no estilo de vida podem trazer grandes benefícios.")
    else:
        insights.append("✅ Parabéns! Seu BMI está na faixa saudável, contribuindo para prêmios mais baixos.")
    
    # Insight sobre fumante
    if user_data.get('smoker') == 'yes':
        potential_savings = 20000  # Estimativa
        insights.append(f"🚭 Parar de fumar pode reduzir seu prêmio em até {format_currency(potential_savings)} por ano!")
    else:
        insights.append("🌟 Por não fumar, você já está economizando significativamente no seu seguro de saúde.")
    
    # Insight sobre filhos
    children = user_data.get('children', 0)
    if children > 2:
        insights.append("👨‍👩‍👧‍👦 Famílias maiores podem se beneficiar de planos familiares específicos.")
    elif children == 0:
        insights.append("💑 Sem dependentes, você pode considerar planos individuais mais econômicos.")
    
    return insights

@st.cache_data
def load_cached_data(file_path):
    """
    Carrega dados com cache do Streamlit
    
    Args:
        file_path (str): Caminho do arquivo
        
    Returns:
        pd.DataFrame: Dados carregados
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.pkl'):
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {file_path}")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def create_session_state_key(prefix, user_data):
    """
    Cria chave única para session state baseada nos dados do usuário
    
    Args:
        prefix (str): Prefixo da chave
        user_data (dict): Dados do usuário
        
    Returns:
        str: Chave única
    """
    # Criar hash dos dados do usuário para chave única
    data_str = json.dumps(user_data, sort_keys=True)
    hash_value = hash(data_str)
    return f"{prefix}_{hash_value}" 