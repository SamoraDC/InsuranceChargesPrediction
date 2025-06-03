# 🏥 Sistema de Predição de Prêmios de Seguro

Sistema de predição de prêmios de seguro usando **Gradient Boosting** otimizado, seguindo melhores práticas de engenharia de software e MLOps.

## 🎯 Visão Geral

Este projeto implementa um sistema completo de predição de prêmios de seguro com:

- **Performance Excelente**: MAE $2,651 (18.6%), R² 0.8795
- **Arquitetura Limpa**: Organização modular e bem estruturada
- **Pronto para Produção**: Scripts centralizados e facilidade de uso
- **Qualidade de Código**: Testes, logging, configuração centralizada

## 🏗️ Estrutura Otimizada do Projeto

```
InsuranceChargesPrediction/
├── main.py                           # 🚀 Script principal de execução
├── config.py                         # ⚙️ Configurações centralizadas
├── Makefile                          # 🛠️ Automação de tarefas
├── pytest.ini                       # 🧪 Configuração de testes
├── 
├── src/
│   └── insurance_prediction/         # Pacote principal do sistema
│       ├── config/                   # Configurações específicas
│       ├── data/                     # Módulos de dados
│       ├── models/                   # Módulos de ML
│       └── utils/                    # Utilitários
├── 
├── scripts/                          # Scripts organizados
│   ├── optimization/                 # Scripts de otimização
│   │   ├── quick_optimize.py        # Otimização rápida
│   │   ├── optimize_model.py        # Otimização completa
│   │   └── save_best_model.py       # Salvar modelo otimizado
│   ├── experiments/                  # Scripts experimentais
│   │   └── investigate_data_issues.py
│   └── legacy/                       # Arquivos legados organizados
├── 
├── tests/                            # Testes organizados
│   ├── integration/                  # Testes de integração
│   │   └── test_log_transform.py
│   ├── test_application.py          # Testes da aplicação
│   ├── test_data_loader.py          # Testes de dados
│   └── test_xgb_lgb_simple.py       # Testes de modelos
├── 
├── app/                              # Aplicação Streamlit
├── data/                             # Dados do projeto
├── models/                           # Modelos treinados
├── logs/                            # Logs do sistema
└── notebooks/                       # Jupyter notebooks
```

## 🚀 Execução Simplificada

### Comandos Principais

```bash
# Comando de ajuda
python main.py -h

# Treinar modelo otimizado
python main.py train

# Executar otimização completa
python main.py optimize

# Executar aplicação Streamlit
python main.py app

# Executar todos os testes
python main.py test

# Predição interativa
python main.py predict
```

### Usando Makefile (Recomendado)

```bash
# Ver todas as opções disponíveis
make help

# Configuração inicial
make setup
make install

# Execução
make train          # Treinar modelo
make optimize       # Otimizar modelo
make app           # Executar Streamlit
make test          # Executar testes
make predict       # Predição interativa

# Desenvolvimento
make lint          # Verificar código
make format        # Formatar código
make clean         # Limpar arquivos temporários
```

## 📊 Performance Atual (EXCELENTE)

✅ **Métricas Otimizadas:**
- **MAE: $2,651.52 (18.6%)** - MUITO BOM
- **MSE: 22,146,259** - OTIMIZADO  
- **R²: 0.8795** - EXCELENTE
- **Configuração: 8 features essenciais**

## ⚡ Quick Start

### 1. Instalação e Configuração

```bash
# Clonar repositório
git clone <repository-url>
cd InsuranceChargesPrediction

# Configuração inicial
make setup
make install

# Verificar instalação
python main.py -h
```

### 2. Treinar e Usar Modelo

```bash
# Treinar modelo otimizado
make train

# Executar aplicação
make app
# Acesse: http://localhost:8501

# Ou fazer predições via linha de comando
python main.py predict
```

### 3. Usar Modelo Programaticamente

```python
import joblib
from config import PRODUCTION_MODEL_PATH

# Carregar modelo otimizado
model_data = joblib.load(PRODUCTION_MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']

# Fazer predição
def predict_premium(age, sex, bmi, children, smoker, region):
    # Preparar dados (encoding + scaling)
    data = prepare_data(age, sex, bmi, children, smoker, region, encoders)
    data_scaled = scaler.transform(data)
    
    # Predição
    prediction = model.predict(data_scaled)[0]
    return prediction

# Exemplo
premium = predict_premium(35, 'male', 25.0, 2, 'no', 'northeast')
print(f"Prêmio: ${premium:,.2f}")
```

## 🔧 Funcionalidades Principais

### ✨ Sistema Otimizado
- **Preprocessing Inteligente**: Features essenciais (8 features)
- **Modelo Gradient Boosting**: Configuração otimizada
- **Performance Excelente**: MAE 18.6%, R² 0.8795
- **Arquitetura Limpa**: Código organizado e modular

### 🎯 Facilidade de Uso
- **Script Principal**: Comandos centralizados em `main.py`
- **Makefile**: Automação de tarefas comuns
- **Configuração Central**: `config.py` para todas as configurações
- **Testes Organizados**: Estrutura clara para testes

### 📊 Qualidade e Manutenção
- **Logging Estruturado**: Sistema de logging configurável
- **Testes Abrangentes**: Unit tests, integration tests
- **Documentação Completa**: Guias para desenvolvimento e produção
- **Código Limpo**: Estrutura seguindo melhores práticas

## 🧪 Testes

```bash
# Executar todos os testes
make test

# Testes específicos
python tests/test_application.py
python tests/test_data_loader.py
python tests/integration/test_log_transform.py

# Com pytest
pytest tests/ -v
```

## 📝 Desenvolvimento

### Estrutura de Comandos

```bash
# Desenvolvimento
make dev-install    # Instalar deps de desenvolvimento
make lint          # Verificar código
make format        # Formatar código
make clean         # Limpar temporários

# Análise
make notebook      # Jupyter notebooks
python scripts/experiments/investigate_data_issues.py

# Deploy
make deploy-check  # Verificar se pronto para deploy
make backup        # Backup de modelos e dados
```

### Configurações

Edite `config.py` para personalizar:

```python
# Configurações de performance
MAX_ACCEPTABLE_MAE = 3000
MIN_ACCEPTABLE_R2 = 0.85

# Configurações de treinamento
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Configurações da aplicação
STREAMLIT_PORT = 8501
```

## 📋 Próximos Passos

1. ✅ **Sistema Otimizado** - Concluído (MAE $2,651)
2. 🔄 **Deploy Automatizado** - Em desenvolvimento
3. 🔄 **Monitoramento** - Planejado
4. 🔄 **API REST** - Planejado

## 🏆 Resultados

**Sistema otimizado e pronto para produção:**
- Performance excelente (MAE 18.6%)
- Arquitetura limpa e bem organizada
- Facilidade de uso e manutenção
- Testes abrangentes e documentação completa

---

**🎯 Execute `make help` para ver todas as opções disponíveis!**
