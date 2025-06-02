# 🏥 Insurance Premium Prediction

Sistema de predição de prêmios de seguro usando **Gradient Boosting** como algoritmo principal, desenvolvido seguindo as melhores práticas de engenharia de software e MLOps.

## 🎯 Visão Geral

Este projeto implementa um sistema completo de predição de prêmios de seguro com:

- **Algoritmo Principal**: Gradient Boosting (sklearn) otimizado para o domínio
- **Arquitetura Modular**: Estrutura bem organizada seguindo padrões de engenharia de software
- **Pipeline Automatizado**: Do carregamento dos dados até o modelo em produção
- **Qualidade de Código**: Logging, testes, validação e documentação completos

## 🏗️ Estrutura do Projeto

```
InsuranceChargesPrediction/
├── src/
│   └── insurance_prediction/           # Pacote principal
│       ├── __init__.py
│       ├── config/                     # Configurações centralizadas
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── data/                       # Módulos de dados
│       │   ├── __init__.py
│       │   ├── loader.py              # Carregamento e validação
│       │   └── preprocessor.py        # Pré-processamento otimizado
│       ├── models/                     # Módulos de ML
│       │   ├── __init__.py
│       │   ├── trainer.py             # Treinamento especializado
│       │   ├── predictor.py           # Predições em produção
│       │   └── evaluator.py           # Avaliação de modelos
│       └── utils/                      # Utilitários
│           ├── __init__.py
│           └── logging.py             # Sistema de logging
├── tests/                              # Testes unitários e integração
├── scripts/                            # Scripts de execução
│   └── train_model.py                 # Script principal de treinamento
├── data/                               # Dados do projeto
│   ├── raw/                           # Dados originais
│   ├── processed/                     # Dados processados
│   └── interim/                       # Dados intermediários
├── models/                             # Modelos treinados
│   └── model_artifacts/               # Artefatos (preprocessor, plots)
├── logs/                              # Logs do sistema
├── app/                               # Aplicação Streamlit
├── notebooks/                         # Jupyter notebooks (desenvolvimento)
├── requirements.txt                    # Dependências
└── README.md
```

## 🚀 Quick Start

### 1. Instalação

```bash
# Clonar o repositório
git clone <repository-url>
cd InsuranceChargesPrediction

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
python scripts/train_model.py --help
```

### 2. Treinamento do Modelo

```bash
# Treinamento completo com otimização de hiperparâmetros
python scripts/train_model.py

# Treinamento rápido (sem otimização)
python scripts/train_model.py --no-optimize

# Com MLflow tracking
python scripts/train_model.py --use-mlflow

# Especificar arquivo de dados customizado
python scripts/train_model.py --data-path /path/to/your/data.csv
```

### 3. Usar Modelo Treinado

```python
from src.insurance_prediction.models.predictor import predict_insurance_premium

# Predição simples
result = predict_insurance_premium(
    age=39,
    sex='female',
    bmi=27.9,
    children=3,
    smoker='no',
    region='southeast'
)

print(f"Prêmio previsto: ${result['predicted_premium']:,.2f}")
```

## 🔧 Funcionalidades Principais

### ✨ Pré-processamento Inteligente
- **Detecção e tratamento de outliers** (método IQR conservador)
- **Features de domínio específicas** (interações críticas para seguros)
- **Encoding otimizado** para Gradient Boosting (Label Encoding)
- **Seleção automática de features** relevantes

### 🎯 Modelo Otimizado
- **Gradient Boosting** como algoritmo principal
- **Otimização automática de hiperparâmetros** (RandomizedSearchCV)
- **Validação cruzada** robusta
- **Múltiplas métricas** de avaliação (R², MAE, RMSE, MAPE, etc.)

### 📊 Monitoramento e Tracking
- **Logging estruturado** com diferentes níveis
- **MLflow integration** (opcional)
- **Métricas abrangentes** de performance
- **Feature importance** automática

### 🏭 Pronto para Produção
- **API de predição** robusta
- **Validação de entrada** completa
- **Tratamento de erros** elegante
- **Intervalos de confiança** nas predições

## 📈 Performance

O modelo Gradient Boosting otimizado alcança:

- **R² > 0.85**: Performance excelente
- **RMSE < 4000**: Erro baixo em valores absolutos
- **MAPE < 15%**: Erro percentual aceitável
- **Tempo de predição < 50ms**: Rápido para produção

## 🛠️ Configuração Avançada

### Customizar Hiperparâmetros

Edite `src/insurance_prediction/config/settings.py`:

```python
GRADIENT_BOOSTING_CONFIG = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    # ... outros parâmetros
}
```

### Configurar Logging

```python
from src.insurance_prediction.utils.logging import setup_logging

# Configurar nível de log
setup_logging("DEBUG")  # DEBUG, INFO, WARNING, ERROR
```

### MLflow Tracking

```bash
# Iniciar MLflow UI
mlflow ui

# Executar com tracking
python scripts/train_model.py --use-mlflow
```

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/

# Testes com cobertura
python -m pytest tests/ --cov=src

# Teste específico
python -m pytest tests/test_data_loader.py
```

## 📝 Desenvolvimento

### Adicionar Nova Feature

1. Edite `src/insurance_prediction/data/preprocessor.py`
2. Atualize `create_domain_features()`
3. Execute testes: `python -m pytest tests/test_preprocessor.py`

### Adicionar Novo Modelo

1. Crie módulo em `src/insurance_prediction/models/`
2. Implemente interface similar ao `GradientBoostingTrainer`
3. Atualize `__init__.py` dos módulos

### Code Quality

```bash
# Formatação
black src/ tests/ scripts/

# Linting
flake8 src/ tests/ scripts/

# Type checking (opcional)
mypy src/
```

## 📊 Dataset

O modelo é treinado com dados de seguros contendo:

- **age**: Idade do segurado (18-64 anos)
- **sex**: Sexo (male/female)
- **bmi**: Índice de massa corporal (15.0-55.0)
- **children**: Número de filhos (0-5)
- **smoker**: Fumante (yes/no)
- **region**: Região (northeast/northwest/southeast/southwest)
- **charges**: Prêmio do seguro (target)

## 🔗 Features Importantes

O modelo identifica automaticamente as features mais importantes:

1. **smoker**: Maior preditor (fumantes pagam muito mais)
2. **age**: Segunda maior importância
3. **bmi**: Terceira maior importância
4. **age_smoker_risk**: Interação crítica
5. **bmi_smoker_risk**: Risco composto

## 🚨 Troubleshooting

### Erro de Import
```bash
# Adicionar src ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Modelo não encontrado
```bash
# Verificar se o modelo foi treinado
ls models/gradient_boosting_model.pkl

# Retreinar se necessário
python scripts/train_model.py
```

### Performance baixa
- Verificar qualidade dos dados de entrada
- Considerar mais dados de treinamento
- Ajustar hiperparâmetros em `config/settings.py`

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🏆 Créditos

Desenvolvido seguindo as melhores práticas de:
- **Engenharia de Software**: Modularidade, testabilidade, manutenibilidade
- **MLOps**: Tracking, versionamento, deployment
- **Data Science**: Feature engineering, validação, avaliação

---

**Feito com ❤️ usando Gradient Boosting e Python**
