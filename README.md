# 🏥 Insurance Charges Predictor / Preditor de Preço de Convênio Médico

<div align="center">

**🇺🇸 [English](#-english) | 🇧🇷 [Português](#-português)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)

**Machine Learning system for health insurance premium prediction**  
**Sistema de Machine Learning para predição de prêmios de convênio médico**

</div>

---

## 🇺🇸 English

### 🎯 Overview

Bilingual health insurance premium prediction system using **Gradient Boosting** algorithm with development environment and production-ready web application.

**Performance:** R² = 87.95% | MAE = $2,651 | Processing < 12ms

### 🚀 Quick Start

#### 🌐 Production App (Recommended)
```bash
cd deploy
pip install -r requirements_deploy.txt
streamlit run streamlit_app.py
```

#### 🔬 Development Environment
```bash
pip install -r requirements.txt
streamlit run app_new.py
```

#### 🐍 Direct Python Usage
```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

result = predict_insurance_premium(
    age=35, sex='male', bmi=25.0, 
    children=2, smoker='no', region='northeast'
)
print(f"Premium: ${result['predicted_premium']:,.2f}")
```

### 📁 Project Structure

```
InsuranceChargesPrediction/
├── 🌐 deploy/                    # Production Web App
│   ├── streamlit_app.py          # Bilingual Streamlit app
│   ├── model_utils.py            # Auto-training model system
│   ├── insurance.csv             # Training data
│   └── requirements_deploy.txt   # Production dependencies
├── 
├── 🔬 src/insurance_prediction/  # Development ML Pipeline
│   ├── config/                   # Configuration management
│   ├── data/                     # Data processing
│   ├── models/                   # ML models & training
│   └── utils/                    # Utilities & logging
├── 
├── 📱 app_new.py                 # Development Streamlit app
├── 📁 data/                      # Raw datasets
└── requirements.txt              # Development dependencies
```

### ✨ Features

- **🌍 Bilingual Interface:** Portuguese/English real-time switching
- **⚡ Auto-Training:** No pre-saved models required, trains fresh
- **📊 Risk Analysis:** Comprehensive risk factor analysis
- **📱 Responsive Design:** Works on desktop and mobile
- **🔒 Production Ready:** Deployed on Streamlit Cloud

### 🛠️ Development

```bash
# Setup
git clone <repository>
cd InsuranceChargesPrediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run development app
streamlit run app_new.py

# Run production app
cd deploy && streamlit run streamlit_app.py

# Test deployment
cd deploy && python model_utils.py
```

### 🌐 Deployment

**Streamlit Cloud:**
1. Fork repository
2. Deploy from `deploy/streamlit_app.py`
3. Uses `requirements_deploy.txt` automatically

### 📊 Model Details

- **Algorithm:** Gradient Boosting (scikit-learn)
- **Features:** 13 engineered features including interactions
- **Performance:** R² = 87.95%, MAE = $2,651
- **Key Factors:** Smoker (highest impact), Age, BMI

---

## 🇧🇷 Português

### 🎯 Visão Geral

Sistema bilíngue de predição de prêmios de convênio médico usando algoritmo **Gradient Boosting** com ambiente de desenvolvimento e aplicação web pronta para produção.

**Performance:** R² = 87.95% | MAE = $2,651 | Processamento < 12ms

### 🚀 Início Rápido

#### 🌐 App de Produção (Recomendado)
```bash
cd deploy
pip install -r requirements_deploy.txt
streamlit run streamlit_app.py
```

#### 🔬 Ambiente de Desenvolvimento
```bash
pip install -r requirements.txt
streamlit run app_new.py
```

#### 🐍 Uso Direto em Python
```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

resultado = predict_insurance_premium(
    age=35, sex='male', bmi=25.0, 
    children=2, smoker='no', region='northeast'
)
print(f"Prêmio: ${resultado['predicted_premium']:,.2f}")
```

### 📁 Estrutura do Projeto

```
InsuranceChargesPrediction/
├── 🌐 deploy/                    # App Web de Produção
│   ├── streamlit_app.py          # App Streamlit bilíngue
│   ├── model_utils.py            # Sistema de modelo auto-treinável
│   ├── insurance.csv             # Dados de treinamento
│   └── requirements_deploy.txt   # Dependências de produção
├── 
├── 🔬 src/insurance_prediction/  # Pipeline ML de Desenvolvimento
│   ├── config/                   # Gerenciamento de configuração
│   ├── data/                     # Processamento de dados
│   ├── models/                   # Modelos ML e treinamento
│   └── utils/                    # Utilitários e logging
├── 
├── 📱 app_new.py                 # App Streamlit de desenvolvimento
├── 📁 data/                      # Datasets originais
└── requirements.txt              # Dependências de desenvolvimento
```

### ✨ Funcionalidades

- **🌍 Interface Bilíngue:** Troca Português/Inglês em tempo real
- **⚡ Auto-Treinamento:** Não requer modelos pré-salvos, treina automaticamente
- **📊 Análise de Risco:** Análise abrangente de fatores de risco
- **📱 Design Responsivo:** Funciona em desktop e mobile
- **🔒 Pronto para Produção:** Deployado no Streamlit Cloud

### 🛠️ Desenvolvimento

```bash
# Configuração
git clone <repositório>
cd InsuranceChargesPrediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Executar app de desenvolvimento
streamlit run app_new.py

# Executar app de produção
cd deploy && streamlit run streamlit_app.py

# Testar deployment
cd deploy && python model_utils.py
```

### 🌐 Deploy

**Streamlit Cloud:**
1. Fork do repositório
2. Deploy a partir de `deploy/streamlit_app.py`
3. Usa `requirements_deploy.txt` automaticamente

### 📊 Detalhes do Modelo

- **Algoritmo:** Gradient Boosting (scikit-learn)
- **Features:** 13 features engenheiradas incluindo interações
- **Performance:** R² = 87.95%, MAE = $2,651
- **Fatores Principais:** Fumante (maior impacto), Idade, BMI

---

<div align="center">

**🚀 Ready to predict insurance premiums?**  
**🚀 Pronto para predizer prêmios de seguro?**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

</div>
