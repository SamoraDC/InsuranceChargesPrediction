# 🏥 Preditor de Prêmios de Seguro - Aplicação Completa

Uma aplicação web inteligente para predição de prêmios de seguro de saúde usando Machine Learning, desenvolvida com Streamlit.

## 🎯 Visão Geral

Este projeto implementa um sistema completo de predição de prêmios de seguro de saúde que permite:

- **Predições individuais** com interface intuitiva
- **Análise em lote** para múltiplos segurados
- **Dashboard analytics** com insights visuais
- **Intervalos de confiança** para estimativas
- **Explicabilidade** do modelo com feature importance

## 🚀 Funcionalidades Principais

### 🎯 Predição Individual
- Formulário interativo para entrada de dados
- Predição em tempo real
- Visualizações personalizadas dos resultados
- Insights automáticos baseados no perfil
- Intervalos de confiança (95%)

### 📊 Análise em Lote
- Upload de arquivos CSV
- Processamento de múltiplos segurados
- Estatísticas e visualizações das predições
- Download dos resultados

### 📈 Dashboard Analytics
- Simulador de cenários
- Gráficos interativos (correlação, distribuição)
- Análise comparativa por características
- Métricas estatísticas em tempo real

### ℹ️ Informações do Projeto
- Documentação completa da metodologia
- Estatísticas do modelo
- Performance e métricas

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit** - Interface web
- **Scikit-learn** - Machine Learning
- **Plotly** - Visualizações interativas
- **Pandas/NumPy** - Manipulação de dados
- **MLflow** - Tracking de experimentos (opcional)

## 📁 Estrutura do Projeto

```
InsuranceChargesPrediction/
├── app.py                          # Aplicação principal Streamlit
├── app/
│   ├── components/
│   │   ├── input_form.py          # Formulário de entrada
│   │   ├── results_display.py     # Exibição de resultados
│   │   └── charts.py              # Gráficos e visualizações
│   └── utils/
│       └── helpers.py             # Funções auxiliares
├── src/
│   ├── config.py                  # Configurações do projeto
│   ├── data_loader.py             # Carregamento de dados
│   ├── preprocessing.py           # Pré-processamento
│   ├── model_training.py          # Treinamento de modelos
│   ├── evaluation.py              # Avaliação de modelos
│   └── predict.py                 # Módulo de predições
├── data/
│   ├── raw/insurance.csv          # Dataset original
│   └── processed/                 # Dados processados
├── models/
│   ├── best_model.pkl             # Melhor modelo treinado
│   └── model_artifacts/           # Artefatos do modelo
├── .streamlit/
│   └── config.toml               # Configuração do Streamlit
├── requirements.txt              # Dependências
└── README.md                     # Este arquivo
```

## 🚀 Como Executar

### 1. Instalação

```bash
# Clonar o repositório
git clone <repository-url>
cd InsuranceChargesPrediction

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar os Dados

```bash
# Executar pipeline de treinamento (se necessário)
python src/data_loader.py
python src/preprocessing.py
python src/model_training.py
```

### 3. Executar a Aplicação

```bash
# Iniciar aplicação Streamlit
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

## 📊 Performance do Modelo

### Métricas Principais
- **R² Score**: 0.8856 (88.56% de variância explicada)
- **MAE**: ~$2,700
- **RMSE**: ~$6,100
- **Modelo Principal**: Ridge Regression

### Features Mais Importantes
1. **bmi_smoker_interaction** - Interação BMI × Fumante
2. **age²** - Idade ao quadrado
3. **age** - Idade
4. **age_bmi** - Interação Idade × BMI
5. **smoker_yes** - Status de fumante

## 🎨 Interface da Aplicação

### Página Principal
- Layout responsivo com sidebar
- Navegação intuitiva entre funcionalidades
- Design moderno com gradientes e animações

### Formulário de Entrada
- Validação em tempo real
- Tooltips explicativos
- Referências de valores (ex: BMI)

### Resultados
- Card destacado com predição principal
- Métricas complementares
- Gráficos interativos (gauge, barras)
- Insights personalizados

### Visualizações
- Gráficos Plotly interativos
- Heatmaps de correlação
- Distribuições e box plots
- Análises 3D e radar charts

## 🔧 Configuração

### Streamlit Config (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
port = 8501
maxUploadSize = 200
```

### Variáveis de Ambiente
- `STREAMLIT_SERVER_PORT`: Porta da aplicação (padrão: 8501)
- `STREAMLIT_THEME_PRIMARY_COLOR`: Cor primária
- `MODEL_PATH`: Caminho customizado para modelos

## 📈 Casos de Uso

### Para Seguradoras
- Precificação automatizada de prêmios
- Análise de risco de portfólio
- Segmentação de clientes

### Para Corretores
- Cotações rápidas para clientes
- Comparação de perfis
- Demonstração de fatores de risco

### Para Consumidores
- Estimativa de custos
- Compreensão de fatores que influenciam prêmios
- Planejamento financeiro

## 🔮 Roadmap

### Versão 1.1
- [ ] Integração com APIs de seguradoras
- [ ] Modelo de classificação de risco
- [ ] Histórico de predições

### Versão 1.2
- [ ] Autenticação de usuários
- [ ] Dashboard administrativo
- [ ] Notificações e alertas

### Versão 2.0
- [ ] Modelos específicos por região
- [ ] Predições com série temporal
- [ ] Integração com dados externos

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Desenvolvido para

**FIAP - Tech Challenge 01**  
*Pós-graduação em Data Science*

## 📧 Contato

Para dúvidas, sugestões ou colaborações, entre em contato através dos issues do GitHub.

---

⭐ **Star este repositório se ele foi útil para você!**
