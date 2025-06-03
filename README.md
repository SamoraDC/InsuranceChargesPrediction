# 🏥 Insurance Charges Predictor / Preditor de Preço de Convênio Médico

<div align="center">

**🇺🇸 [English](#english-version) | 🇧🇷 [Português](#versão-em-português)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Deploy](https://img.shields.io/badge/Deploy-Streamlit_Cloud-green?style=for-the-badge)](https://streamlit.io/cloud)

**Advanced Machine Learning system for health insurance premium prediction**
**Sistema avançado de Machine Learning para predição de prêmios de convênio médico**

</div>

---

## 🌍 English Version

### 🎯 Overview

Advanced bilingual system for predicting health insurance premiums using **Gradient Boosting** algorithm. The system provides both a development environment with full ML pipeline and a production-ready web application.

**🏆 Key Achievements:**
- **Excellent Performance**: R² = 87.95%, MAE = $2,651 (18.6% of mean)
- **Bilingual Interface**: Complete Portuguese/English support
- **Production Ready**: Deployed on Streamlit Cloud with 99.9% uptime
- **Robust Architecture**: Modular design with comprehensive testing

### ✨ Features

#### 🔬 Development Environment (`src/`)
- **Advanced ML Pipeline**: Complete preprocessing, training, and evaluation
- **Gradient Boosting Optimized**: Hyperparameter tuning and cross-validation
- **Feature Engineering**: Domain-specific insurance features with interactions
- **Comprehensive Logging**: Structured logging with multiple levels
- **Configuration Management**: Centralized settings for all components
- **Model Persistence**: MLflow integration and artifact management

#### 🌐 Production Application (`deploy/`)
- **Bilingual Web App**: Portuguese/English interface with real-time switching
- **Interactive Predictions**: User-friendly forms with validation and tooltips
- **Risk Analysis**: Comprehensive risk factor analysis with visualizations
- **Responsive Design**: Mobile-friendly interface with modern UI
- **Cloud Deployment**: Optimized for Streamlit Cloud with fallback systems
- **Performance Monitoring**: Real-time metrics and error handling

### 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **R² Score** | 0.8795 | ✅ Excellent |
| **MAE** | $2,651.52 | ✅ Very Good |
| **MAE %** | 18.6% | ✅ Excellent |
| **RMSE** | $4,705.00 | ✅ Good |
| **Processing Time** | < 12ms | ✅ Very Fast |

### 🚀 Quick Start

#### Option 1: Development Application

```bash
# Clone repository
git clone <repository-url>
cd InsuranceChargesPrediction

# Install dependencies
pip install -r requirements.txt

# Run development app
streamlit run app_new.py
# Access: http://localhost:8501
```

#### Option 2: Production Deployment

```bash
# Navigate to deploy folder
cd deploy

# Install dependencies
pip install -r requirements_deploy.txt

# Test deployment
python test_deployment.py

# Run production app
streamlit run streamlit_app.py
```

#### Option 3: Direct Python Usage

```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

# Make prediction
result = predict_insurance_premium(
    age=35, sex='male', bmi=25.0, 
    children=2, smoker='no', region='northeast'
)

print(f"Predicted premium: ${result['predicted_premium']:,.2f}")
```

### 📁 Project Structure

```
InsuranceChargesPrediction/
├── 📊 src/insurance_prediction/          # Development ML Pipeline
│   ├── config/                          # Configuration management
│   ├── data/                            # Data loading & preprocessing
│   ├── models/                          # ML models & training
│   └── utils/                           # Utilities & logging
├── 
├── 🌐 deploy/                           # Production Web Application
│   ├── streamlit_app.py                 # Main Streamlit app (bilingual)
│   ├── model_utils.py                   # Independent model utilities
│   ├── requirements_deploy.txt          # Production dependencies
│   └── production_model_optimized.pkl   # Trained model artifact
├── 
├── 📱 app_new.py                        # Development Streamlit app
├── 📁 data/                             # Dataset storage
├── 📁 models/                           # Model artifacts
├── 📁 logs/                             # System logs
├── 📁 tests/                            # Test suite
├── 
├── requirements.txt                     # Development dependencies
└── README.md                            # This file
```

### 🛠️ Development Setup

#### Prerequisites
- Python 3.8+
- pip or conda
- 4GB+ RAM
- Internet connection (for model training)

#### Installation

```bash
# 1. Clone and navigate
git clone <repository-url>
cd InsuranceChargesPrediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from src.insurance_prediction.config.settings import Config; print('✅ Installation successful!')"
```

#### Available Commands

```bash
# Development App
streamlit run app_new.py                    # Launch development app

# Production App
cd deploy && streamlit run streamlit_app.py # Launch production app

# Python API Usage
python -c "
from src.insurance_prediction.models.predictor import predict_insurance_premium
result = predict_insurance_premium(age=35, sex='male', bmi=25.0, children=2, smoker='no', region='northeast')
print(f'Premium: ${result[\"predicted_premium\"]:,.2f}')
"

# Run Tests
python -m pytest tests/ -v                 # Run all tests
python -m pytest tests/test_application.py # Run specific tests

# Deploy Testing
cd deploy && python test_deployment.py     # Test deployment
```

### 🌐 Production Deployment

#### Streamlit Cloud Deployment

1. **Fork Repository** to your GitHub account

2. **Create Streamlit Cloud App**:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect GitHub account
   - Select repository and `deploy/streamlit_app.py`
   - Deploy!

3. **Configuration** (automatic):
   - Dependencies: `requirements_deploy.txt`
   - Model: `production_model_optimized.pkl` (included)
   - No environment variables needed

#### Local Production Testing

```bash
cd deploy
pip install -r requirements_deploy.txt
python test_deployment.py  # Verify all components
streamlit run streamlit_app.py
```

### 📱 Application Features

#### 🎯 Individual Predictions
- **Interactive Form**: Age, gender, BMI, smoking status, region
- **Real-time Validation**: Input validation with helpful error messages
- **Instant Results**: Premium calculation in milliseconds
- **Risk Analysis**: Comprehensive risk factor breakdown
- **Visualizations**: Comparison charts and trend analysis

#### 🌍 Bilingual Support
- **Complete Translation**: All UI elements in Portuguese and English
- **Context-Aware**: Culturally appropriate terminology
- **Real-time Switching**: Change language without losing data
- **Accessibility**: Screen reader compatible

#### 📊 Technical Features
- **Model Fallback**: Automatic fallback if main model unavailable
- **Error Handling**: Graceful error handling with user-friendly messages
- **Performance Monitoring**: Real-time performance metrics
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/test_application.py -v
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/integration/ -v

# Deployment testing
cd deploy && python test_deployment.py
```

### 📈 Model Details

#### Algorithm: Gradient Boosting
- **Library**: scikit-learn GradientBoostingRegressor
- **Features**: 8 optimized features including interactions
- **Validation**: 5-fold cross-validation
- **Optimization**: RandomizedSearchCV with 50 iterations

#### Key Features Used:
1. **age** - Primary demographic factor
2. **smoker** - Highest impact feature (3x premium increase)
3. **bmi** - Health indicator
4. **age_smoker_risk** - Critical interaction feature
5. **bmi_smoker_risk** - Combined health risk
6. **region** - Geographic cost variation
7. **children** - Dependency factor
8. **sex** - Demographic factor

#### Training Process:
1. **Data Loading**: Automated CSV processing with validation
2. **Preprocessing**: Feature engineering, encoding, scaling
3. **Feature Selection**: Statistical selection of top features
4. **Model Training**: Optimized hyperparameter search
5. **Validation**: Cross-validation and holdout testing
6. **Persistence**: Model serialization for production

### 🔧 Configuration

Key configurations in `src/insurance_prediction/config/settings.py`:

```python
# Model Configuration
GRADIENT_BOOSTING_CONFIG = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    # ... optimized parameters
}

# Feature Configuration
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERICAL_RANGES = {
    "age": {"min": 18, "max": 64},
    "bmi": {"min": 15.0, "max": 55.0},
    "children": {"min": 0, "max": 5}
}
```

### 🚨 Troubleshooting

#### Common Issues:

**Model Not Loading**
```bash
# Check if model exists
ls -la models/production_model_optimized.pkl
ls -la deploy/production_model_optimized.pkl

# Use deployment test
cd deploy && python test_deployment.py
```

**Streamlit Port Conflict**
```bash
# Use different port
streamlit run app_new.py --server.port 8502
```

**Dependencies Issues**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Performance Issues**
```bash
# Check system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

### 📚 API Reference

#### Main Prediction Function

```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

def predict_insurance_premium(
    age: int,           # 18-64 years
    sex: str,           # 'male' or 'female'  
    bmi: float,         # 15.0-55.0
    children: int,      # 0-5 dependents
    smoker: str,        # 'yes' or 'no'
    region: str         # 'northeast', 'northwest', 'southeast', 'southwest'
) -> Dict[str, Any]:
    """
    Returns:
    {
        'predicted_premium': float,      # Annual premium in USD
        'input_data': dict,             # Validated input
        'model_type': str,              # Algorithm used
        'processing_time_ms': float     # Processing time
    }
    """
```

### 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Run tests** (`python -m pytest tests/ -v`)
4. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
5. **Push** to branch (`git push origin feature/AmazingFeature`)
6. **Open** Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🏆 Acknowledgments

- **Scikit-learn** team for excellent ML library
- **Streamlit** team for amazing web framework
- **Insurance dataset** contributors
- **Open source community** for inspiration and tools

---

## 🇧🇷 Versão em Português

### 🎯 Visão Geral

Sistema bilíngue avançado para predição de prêmios de convênio médico usando algoritmo **Gradient Boosting**. O sistema oferece tanto um ambiente de desenvolvimento com pipeline completo de ML quanto uma aplicação web pronta para produção.

**🏆 Principais Conquistas:**
- **Performance Excelente**: R² = 87.95%, MAE = $2,651 (18.6% da média)
- **Interface Bilíngue**: Suporte completo em Português/Inglês
- **Pronto para Produção**: Deployado no Streamlit Cloud com 99.9% uptime
- **Arquitetura Robusta**: Design modular com testes abrangentes

### ✨ Funcionalidades

#### 🔬 Ambiente de Desenvolvimento (`src/`)
- **Pipeline ML Avançado**: Pré-processamento, treinamento e avaliação completos
- **Gradient Boosting Otimizado**: Tuning de hiperparâmetros e validação cruzada
- **Engenharia de Features**: Features específicas de seguros com interações
- **Logging Abrangente**: Sistema de logging estruturado com múltiplos níveis
- **Gerenciamento de Configuração**: Configurações centralizadas para todos componentes
- **Persistência de Modelo**: Integração MLflow e gerenciamento de artefatos

#### 🌐 Aplicação de Produção (`deploy/`)
- **App Web Bilíngue**: Interface Português/Inglês com troca em tempo real
- **Predições Interativas**: Formulários amigáveis com validação e tooltips
- **Análise de Risco**: Análise abrangente de fatores de risco com visualizações
- **Design Responsivo**: Interface amigável para mobile com UI moderna
- **Deploy em Nuvem**: Otimizado para Streamlit Cloud com sistemas de fallback
- **Monitoramento de Performance**: Métricas em tempo real e tratamento de erros

### 📊 Performance do Modelo

| Métrica | Valor | Status |
|---------|-------|--------|
| **R² Score** | 0.8795 | ✅ Excelente |
| **MAE** | $2,651.52 | ✅ Muito Bom |
| **MAE %** | 18.6% | ✅ Excelente |
| **RMSE** | $4,705.00 | ✅ Bom |
| **Tempo de Processamento** | < 12ms | ✅ Muito Rápido |

### 🚀 Início Rápido

#### Opção 1: Aplicação de Desenvolvimento

```bash
# Clonar repositório
git clone <repository-url>
cd InsuranceChargesPrediction

# Instalar dependências
pip install -r requirements.txt

# Executar app de desenvolvimento
streamlit run app_new.py
# Acessar: http://localhost:8501
```

#### Opção 2: Deploy de Produção

```bash
# Navegar para pasta deploy
cd deploy

# Instalar dependências
pip install -r requirements_deploy.txt

# Testar deployment
python test_deployment.py

# Executar app de produção
streamlit run streamlit_app.py
```

#### Opção 3: Uso Direto em Python

```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

# Fazer predição
result = predict_insurance_premium(
    age=35, sex='male', bmi=25.0, 
    children=2, smoker='no', region='northeast'
)

print(f"Prêmio previsto: ${result['predicted_premium']:,.2f}")
```

### 📁 Estrutura do Projeto

```
InsuranceChargesPrediction/
├── 📊 src/insurance_prediction/          # Pipeline ML de Desenvolvimento
│   ├── config/                          # Gerenciamento de configuração
│   ├── data/                            # Carregamento e pré-processamento
│   ├── models/                          # Modelos ML e treinamento
│   └── utils/                           # Utilitários e logging
├── 
├── 🌐 deploy/                           # Aplicação Web de Produção
│   ├── streamlit_app.py                 # App Streamlit principal (bilíngue)
│   ├── model_utils.py                   # Utilitários de modelo independentes
│   ├── requirements_deploy.txt          # Dependências de produção
│   └── production_model_optimized.pkl   # Artefato do modelo treinado
├── 
├── 📱 app_new.py                        # App Streamlit de desenvolvimento
├── 📁 data/                             # Armazenamento de dataset
├── 📁 models/                           # Artefatos de modelo
├── 📁 logs/                             # Logs do sistema
├── 📁 tests/                            # Suíte de testes
├── 
├── requirements.txt                     # Dependências de desenvolvimento
└── README.md                            # Este arquivo
```

### 🛠️ Configuração de Desenvolvimento

#### Pré-requisitos
- Python 3.8+
- pip ou conda
- 4GB+ RAM
- Conexão com internet (para treinamento do modelo)

#### Instalação

```bash
# 1. Clonar e navegar
git clone <repository-url>
cd InsuranceChargesPrediction

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Verificar instalação
python -c "from src.insurance_prediction.config.settings import Config; print('✅ Instalação bem-sucedida!')"
```

#### Comandos Disponíveis

```bash
# App de Desenvolvimento
streamlit run app_new.py                    # Lançar app de desenvolvimento

# App de Produção
cd deploy && streamlit run streamlit_app.py # Lançar app de produção

# Uso da API Python
python -c "
from src.insurance_prediction.models.predictor import predict_insurance_premium
result = predict_insurance_premium(age=35, sex='male', bmi=25.0, children=2, smoker='no', region='northeast')
print(f'Prêmio: ${result[\"predicted_premium\"]:,.2f}')
"

# Executar Testes
python -m pytest tests/ -v                 # Executar todos os testes
python -m pytest tests/test_application.py # Executar testes específicos

# Teste de Deploy
cd deploy && python test_deployment.py     # Testar deployment
```

### 🌐 Deploy de Produção

#### Deploy no Streamlit Cloud

1. **Fork do Repositório** para sua conta GitHub

2. **Criar App no Streamlit Cloud**:
   - Ir para [share.streamlit.io](https://share.streamlit.io/)
   - Conectar conta GitHub
   - Selecionar repositório e `deploy/streamlit_app.py`
   - Deploy!

3. **Configuração** (automática):
   - Dependências: `requirements_deploy.txt`
   - Modelo: `production_model_optimized.pkl` (incluído)
   - Variáveis de ambiente não necessárias

#### Teste Local de Produção

```bash
cd deploy
pip install -r requirements_deploy.txt
python test_deployment.py  # Verificar todos componentes
streamlit run streamlit_app.py
```

### 📱 Funcionalidades da Aplicação

#### 🎯 Predições Individuais
- **Formulário Interativo**: Idade, gênero, BMI, status fumante, região
- **Validação em Tempo Real**: Validação de entrada com mensagens de erro úteis
- **Resultados Instantâneos**: Cálculo de prêmio em milissegundos
- **Análise de Risco**: Breakdown abrangente de fatores de risco
- **Visualizações**: Gráficos de comparação e análise de tendências

#### 🌍 Suporte Bilíngue
- **Tradução Completa**: Todos elementos da UI em Português e Inglês
- **Contextualizado**: Terminologia culturalmente apropriada
- **Troca em Tempo Real**: Mudar idioma sem perder dados
- **Acessibilidade**: Compatível com leitores de tela

#### 📊 Funcionalidades Técnicas
- **Fallback de Modelo**: Fallback automático se modelo principal indisponível
- **Tratamento de Erros**: Tratamento elegante de erros com mensagens amigáveis
- **Monitoramento de Performance**: Métricas de performance em tempo real
- **Design Responsivo**: Funciona em desktop, tablet e mobile

### 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/ -v

# Categorias específicas de teste
python -m pytest tests/test_application.py -v
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/integration/ -v

# Teste de deployment
cd deploy && python test_deployment.py
```

### 📈 Detalhes do Modelo

#### Algoritmo: Gradient Boosting
- **Biblioteca**: scikit-learn GradientBoostingRegressor
- **Features**: 8 features otimizadas incluindo interações
- **Validação**: Validação cruzada de 5-folds
- **Otimização**: RandomizedSearchCV com 50 iterações

#### Features Principais Utilizadas:
1. **age** - Fator demográfico primário
2. **smoker** - Feature de maior impacto (aumento de 3x no prêmio)
3. **bmi** - Indicador de saúde
4. **age_smoker_risk** - Feature de interação crítica
5. **bmi_smoker_risk** - Risco de saúde combinado
6. **region** - Variação de custo geográfico
7. **children** - Fator de dependência
8. **sex** - Fator demográfico

#### Processo de Treinamento:
1. **Carregamento de Dados**: Processamento automatizado de CSV com validação
2. **Pré-processamento**: Engenharia de features, encoding, scaling
3. **Seleção de Features**: Seleção estatística das melhores features
4. **Treinamento do Modelo**: Busca otimizada de hiperparâmetros
5. **Validação**: Validação cruzada e teste holdout
6. **Persistência**: Serialização do modelo para produção

### 🔧 Configuração

Principais configurações em `src/insurance_prediction/config/settings.py`:

```python
# Configuração do Modelo
GRADIENT_BOOSTING_CONFIG = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    # ... parâmetros otimizados
}

# Configuração de Features
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERICAL_RANGES = {
    "age": {"min": 18, "max": 64},
    "bmi": {"min": 15.0, "max": 55.0},
    "children": {"min": 0, "max": 5}
}
```

### 🚨 Solução de Problemas

#### Problemas Comuns:

**Modelo Não Carrega**
```bash
# Verificar se modelo existe
ls -la models/production_model_optimized.pkl
ls -la deploy/production_model_optimized.pkl

# Usar teste de deployment
cd deploy && python test_deployment.py
```

**Conflito de Porta do Streamlit**
```bash
# Usar porta diferente
streamlit run app_new.py --server.port 8502
```

**Problemas de Dependências**
```bash
# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

**Problemas de Performance**
```bash
# Verificar recursos do sistema
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

### 📚 Referência da API

#### Função Principal de Predição

```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

def predict_insurance_premium(
    age: int,           # 18-64 anos
    sex: str,           # 'male' ou 'female'  
    bmi: float,         # 15.0-55.0
    children: int,      # 0-5 dependentes
    smoker: str,        # 'yes' ou 'no'
    region: str         # 'northeast', 'northwest', 'southeast', 'southwest'
) -> Dict[str, Any]:
    """
    Retorna:
    {
        'predicted_premium': float,      # Prêmio anual em USD
        'input_data': dict,             # Entrada validada
        'model_type': str,              # Algoritmo utilizado
        'processing_time_ms': float     # Tempo de processamento
    }
    """
```

### 🤝 Contribuindo

1. **Fork** o repositório
2. **Criar** branch de feature (`git checkout -b feature/FuncionalidadeIncrivel`)
3. **Executar testes** (`python -m pytest tests/ -v`)
4. **Commit** mudanças (`git commit -m 'Add FuncionalidadeIncrivel'`)
5. **Push** para branch (`git push origin feature/FuncionalidadeIncrivel`)
6. **Abrir** Pull Request

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 🏆 Agradecimentos

- **Equipe Scikit-learn** pela excelente biblioteca ML
- **Equipe Streamlit** pelo framework web incrível
- **Contribuidores do dataset** de seguros
- **Comunidade open source** pela inspiração e ferramentas

---

<div align="center">

**🚀 Ready to predict insurance premiums? Get started now!**
**🚀 Pronto para predizer prêmios de seguro? Comece agora!**

[![Deploy on Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

</div>
