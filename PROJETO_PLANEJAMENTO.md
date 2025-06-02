# 📋 PLANEJAMENTO DO PROJETO - Insurance Charges Prediction

## 🎯 Objetivo
Modularizar o notebook `FIAP_Tech_Challenge_01.ipynb` em uma estrutura de projeto MLOps organizada, seguindo boas práticas e incluindo deployment no Streamlit Cloud.

## 🏗️ Estrutura do Projeto Final
```
insurance_project/
├── data/
│   ├── raw/
│   │   └── insurance.csv              # Dados originais
│   ├── processed/
│   │   ├── train_data.csv             # Dados de treino processados
│   │   └── test_data.csv              # Dados de teste processados
│   └── interim/
│       └── cleaned_data.csv           # Dados após limpeza inicial
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # Análise exploratória
│   ├── 02_feature_engineering.ipynb   # Engenharia de features
│   └── 03_model_comparison.ipynb      # Comparação de modelos
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configurações e constantes
│   ├── data_loader.py                 # Carregamento de dados
│   ├── preprocessing.py               # Pré-processamento e feature engineering
│   ├── model_training.py              # Treinamento e tuning de modelos
│   ├── evaluation.py                  # Avaliação de modelos e análise de erros
│   ├── predict.py                     # Funções para fazer predições
│   └── utils.py                       # Funções utilitárias
├── models/
│   ├── best_model.pkl                 # Modelo final treinado
│   ├── preprocessor.pkl               # Pipeline de pré-processamento
│   └── model_artifacts/               # Artefatos do MLflow
├── app/
│   ├── app.py                         # Aplicação Streamlit principal
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py                 # Componentes da sidebar
│   │   ├── prediction_form.py         # Formulário de predição
│   │   └── visualizations.py          # Visualizações
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                 # Funções auxiliares do app
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Testes para preprocessing
│   ├── test_model_training.py         # Testes para modelos
│   └── test_api.py                    # Testes para API/app
├── scripts/
│   ├── train_model.py                 # Script para treinar modelo
│   ├── evaluate_model.py              # Script para avaliar modelo
│   └── deploy_model.py                # Script para deployment
├── .streamlit/
│   └── config.toml                    # Configurações do Streamlit
├── requirements.txt                   # Dependências Python
├── packages.txt                       # Dependências sistema (se necessário)
├── main.py                           # Script principal para orquestrar pipeline
├── README.md                         # Documentação do projeto
└── .gitignore                        # Arquivos para ignorar no Git
```

## 📊 Análise do Notebook Atual

### Bibliotecas Identificadas
- **Análise de Dados**: pandas, numpy
- **Visualização**: matplotlib, seaborn
- **ML/Estatística**: scikit-learn, scipy
- **Pré-processamento**: StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline
- **Modelos**: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBoost
- **Feature Engineering**: PolynomialFeatures, category_encoders
- **Feature Selection**: SelectKBest, SelectFromModel, RFE
- **Validação**: cross_val_score, train_test_split

### Etapas Identificadas no Notebook
1. **Carregamento e Exploração dos Dados**
2. **Análise Exploratória de Dados (EDA)**
3. **Pré-processamento e Limpeza**
4. **Feature Engineering**
5. **Feature Selection**
6. **Treinamento de Múltiplos Modelos**
7. **Avaliação e Comparação de Modelos**
8. **Análise de Erros e Importância de Features**

## 🔄 Plano de Modularização

### Fase 1: Configuração Base
1. **Criar estrutura de diretórios**
2. **Configurar `config.py`** com todas as constantes e parâmetros
3. **Criar `requirements.txt`** com todas as dependências identificadas
4. **Configurar `.gitignore`** para evitar arquivos desnecessários

### Fase 2: Módulos de Dados
1. **`data_loader.py`**:
   - Função para carregar dados originais
   - Validação de integridade dos dados
   - Logging de informações do dataset

2. **`preprocessing.py`**:
   - Pipeline de limpeza de dados
   - Tratamento de valores ausentes
   - Detecção e tratamento de outliers
   - Transformações de variáveis (Box-Cox, Yeo-Johnson)
   - Normalização/Padronização
   - Encoding de variáveis categóricas

### Fase 3: Feature Engineering e Selection
1. **Expandir `preprocessing.py`**:
   - Feature engineering (polynomial features, interactions)
   - Feature selection (filter, wrapper, embedded methods)
   - Pipeline completo de transformação

### Fase 4: Modelagem
1. **`model_training.py`**:
   - Classe para cada tipo de modelo
   - Hyperparameter tuning
   - Cross-validation
   - Treinamento com MLflow tracking

2. **`evaluation.py`**:
   - Métricas de avaliação (MAE, MSE, R², etc.)
   - Gráficos de análise (predito vs real, resíduos)
   - Feature importance analysis
   - Testes estatísticos de validação

### Fase 5: Pipeline de Predição
1. **`predict.py`**:
   - Carregamento de modelo e preprocessor salvos
   - Função de predição para novos dados
   - Validação de entrada
   - Interpretabilidade do modelo

### Fase 6: Aplicação Streamlit
1. **`app/app.py`** - Interface principal:
   - Layout responsivo e moderno
   - Formulário de entrada interativo
   - Visualizações dinâmicas
   - Explicabilidade das predições

2. **Componentes modulares**:
   - `sidebar.py`: Navegação e configurações
   - `prediction_form.py`: Formulário de input
   - `visualizations.py`: Gráficos e charts

### Fase 7: Scripts de Execução
1. **`main.py`**: Orquestrador principal do pipeline
2. **`scripts/train_model.py`**: Script para re-treinar modelos
3. **`scripts/evaluate_model.py`**: Avaliação completa de modelos

## 🚀 Estratégia de Deployment no Streamlit Cloud

### Pré-requisitos
1. **Repository no GitHub**: Código organizado e versionado
2. **requirements.txt**: Dependências otimizadas para produção
3. **Configuração Streamlit**: Arquivo `.streamlit/config.toml`

### Configurações Específicas

#### `requirements.txt` Otimizado
```txt
streamlit>=1.28.0
pandas>=1.5.3
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
joblib>=1.3.0
```

#### `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

### Funcionalidades da Aplicação
1. **Interface Intuitiva**:
   - Formulário para inserir dados do segurado
   - Sliders e selectboxes para facilitar entrada
   - Validação em tempo real

2. **Predições Interativas**:
   - Predição em tempo real
   - Intervalos de confiança
   - Explicabilidade do resultado

3. **Visualizações**:
   - Gráficos de feature importance
   - Comparação com histórico
   - Distribuições de dados

4. **Modelo de Dados**:
   - Campos: age, sex, bmi, children, smoker, region
   - Validação de tipos e ranges
   - Feedback visual para o usuário

## 📝 Checklist de Implementação

### ✅ Estrutura Base
- [ ] Criar estrutura de diretórios
- [ ] Configurar `config.py`
- [ ] Criar `requirements.txt`
- [ ] Configurar `.gitignore`

### ✅ Módulos de Dados
- [ ] Implementar `data_loader.py`
- [ ] Implementar `preprocessing.py` (básico)
- [ ] Testar carregamento e limpeza básica

### ✅ Feature Engineering
- [ ] Expandir `preprocessing.py` com feature engineering
- [ ] Implementar feature selection
- [ ] Criar pipelines de transformação

### ✅ Modelagem
- [ ] Implementar `model_training.py`
- [ ] Implementar `evaluation.py`
- [ ] Configurar MLflow tracking
- [ ] Treinar e salvar melhor modelo

### ✅ Predição
- [ ] Implementar `predict.py`
- [ ] Testar pipeline completo de predição
- [ ] Validar entrada e saída

### ✅ Aplicação Streamlit
- [ ] Criar estrutura base do app
- [ ] Implementar formulário de entrada
- [ ] Adicionar visualizações
- [ ] Testar localmente

### ✅ Deployment
- [ ] Configurar arquivos para Streamlit Cloud
- [ ] Testar deployment local
- [ ] Deploy no Streamlit Cloud
- [ ] Validar funcionamento em produção

### ✅ Documentação
- [ ] Atualizar README.md
- [ ] Documentar APIs dos módulos
- [ ] Criar guia de uso da aplicação

## 🔧 Configurações Técnicas

### MLflow Integration
- Tracking de experimentos
- Registro de modelos
- Versionamento de artefatos
- Comparação de performance

### Testes e Validação
- Testes unitários para cada módulo
- Validação de dados de entrada
- Testes de integração
- CI/CD básico (futuro)

### Performance e Escalabilidade
- Otimização de carregamento de modelos
- Cache de predições
- Validação eficiente de entrada
- Logging estruturado

## 📈 Próximos Passos

1. **Começar pela estrutura base** e configuração
2. **Modularizar gradualmente** seguindo a ordem das fases
3. **Testar cada módulo** individualmente antes de integrar
4. **Desenvolver aplicação Streamlit** em paralelo aos módulos finais
5. **Realizar deployment** após validação local completa

Este planejamento garante uma migração organizada do notebook para uma estrutura de produção, mantendo todas as funcionalidades existentes enquanto adiciona robustez, testabilidade e facilidade de manutenção. 