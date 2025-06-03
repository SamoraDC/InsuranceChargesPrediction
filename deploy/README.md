# 🚀 Insurance Charges Predictor - Deploy Package
# 🚀 Preditor de Preço de Convênio Médico - Pacote de Deploy

## 📝 Overview / Visão Geral

This deploy package contains a **standalone, bilingual version** of the Insurance Charges Predictor for **Streamlit Cloud deployment**.

Este pacote de deploy contém uma **versão standalone e bilíngue** do Preditor de Preço de Convênio Médico para **deploy no Streamlit Cloud**.

## 🎯 Phase 1 Implementation / Implementação da Fase 1

### ✅ What's Included / O que está incluído:

1. **🔧 Independent model utilities** / **Utilitários de modelo independentes**
   - `model_utils.py`: Standalone prediction functions / Funções de predição independentes
   - No dependencies on `src/` directory / Sem dependências do diretório `src/`
   - Uses `production_model_optimized.pkl` / Usa `production_model_optimized.pkl`
   - **NEW:** Robust model loading with fallback / **NOVO:** Carregamento robusto com fallback

2. **🌐 Bilingual Streamlit App** / **App Streamlit Bilíngue**
   - `streamlit_app.py`: Complete bilingual interface / Interface bilíngue completa
   - 🇺🇸 English + 🇧🇷 Portuguese support / Suporte a Inglês + Português
   - Language toggle button / Botão de alternância de idioma

3. **📦 Minimal Dependencies** / **Dependências Mínimas**
   - `requirements_deploy.txt`: Only 6 essential packages / Apenas 6 pacotes essenciais
   - Optimized for Streamlit Cloud / Otimizado para Streamlit Cloud

4. **⚙️ Streamlit Configuration** / **Configuração do Streamlit**
   - `.streamlit/config.toml`: Theme and server settings / Configurações de tema e servidor

5. **🤖 Included Model** / **Modelo Incluído**
   - `production_model_optimized.pkl`: Pre-trained model / Modelo pré-treinado
   - **NEW:** Multiple path detection / **NOVO:** Detecção de múltiplos caminhos
   - **NEW:** Automatic fallback system / **NOVO:** Sistema de fallback automático

## 🔄 Model Performance / Performance do Modelo

- **Algorithm / Algoritmo:** Gradient Boosting
- **R² Score:** 0.8795 (87.95%)
- **MAE:** $2,651 (18.6% of mean / da média)
- **Processing Time / Tempo:** < 600ms (or < 100ms fallback)
- **Features:** 8 optimized features / features otimizadas

## 🚀 Quick Start / Início Rápido

### Local Testing / Teste Local:
```bash
cd deploy
pip install -r requirements_deploy.txt
streamlit run streamlit_app.py
```

### Streamlit Cloud Deploy:
1. Upload this `deploy/` folder as a new repository / Faça upload desta pasta `deploy/` como novo repositório
2. Point Streamlit Cloud to `streamlit_app.py` / Configure Streamlit Cloud para `streamlit_app.py`
3. The app will auto-detect and load the model / O app detectará e carregará o modelo automaticamente
4. **NEW:** If model fails, automatic fallback ensures app still works / **NOVO:** Se modelo falhar, fallback automático garante que app funcione

## 📂 File Structure / Estrutura de Arquivos

```
deploy/
├── streamlit_app.py          # Main bilingual app / App principal bilíngue
├── model_utils.py            # Independent model functions / Funções independentes
├── production_model_optimized.pkl  # Pre-trained model / Modelo pré-treinado
├── requirements_deploy.txt   # Minimal dependencies / Dependências mínimas
├── .streamlit/
│   └── config.toml          # Streamlit configuration / Configuração
├── test_deployment.py       # Deployment tests / Testes de deploy
└── README.md               # This file / Este arquivo
```

## 🌍 Language Support / Suporte a Idiomas

The app automatically detects user preference and provides:
O app detecta automaticamente a preferência do usuário e fornece:

- **🇺🇸 English:** Complete interface in English
- **🇧🇷 Português:** Interface completa em Português Brasileiro
- **🔄 Toggle:** Easy language switching / Alternância fácil de idioma

## 🎨 Features / Funcionalidades

### Individual Prediction / Predição Individual:
- User-friendly form / Formulário amigável
- Real-time validation / Validação em tempo real
- Risk analysis / Análise de risco
- Comparison charts / Gráficos de comparação

### Model Information / Informações do Modelo:
- Performance metrics / Métricas de performance
- Feature importance / Importância das features
- Processing time / Tempo de processamento
- **NEW:** Robust error handling / **NOVO:** Tratamento robusto de erros

## 🔧 Technical Details / Detalhes Técnicos

### Model Architecture / Arquitetura do Modelo:
- **Base features:** age, sex, bmi, children, smoker, region
- **Engineered features:** bmi_smoker, age_smoker interactions
- **Preprocessing:** LabelEncoder + StandardScaler
- **Training:** Optimized hyperparameters / Hiperparâmetros otimizados

### Dependencies / Dependências:
```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
plotly==5.17.0
joblib==1.3.2
```

### NEW: Robust Model Loading / NOVO: Carregamento Robusto do Modelo:
- Multiple path detection for different environments
- Automatic fallback to dummy model if main model unavailable
- Graceful degradation with user notification
- Works in both development and production environments

## 🎯 Next Phases / Próximas Fases

### Phase 2 / Fase 2:
- [ ] Model versioning / Versionamento de modelo
- [ ] A/B testing capability / Capacidade de teste A/B
- [ ] Enhanced analytics / Analytics aprimorados

### Phase 3 / Fase 3:
- [ ] Real-time model updates / Atualizações de modelo em tempo real
- [ ] User feedback integration / Integração de feedback do usuário
- [ ] Advanced monitoring / Monitoramento avançado

## ❗ Important Notes / Notas Importantes

1. **Model File Included / Arquivo de Modelo Incluído:**
   - ✅ `production_model_optimized.pkl` is now included in deploy folder
   - ✅ `production_model_optimized.pkl` agora está incluído na pasta deploy

2. **Standalone Operation / Operação Independente:**
   - This package works independently of the `src/` directory
   - Este pacote funciona independentemente do diretório `src/`

3. **Streamlit Cloud Compatibility / Compatibilidade:**
   - Optimized for Streamlit Cloud requirements
   - Otimizado para os requisitos do Streamlit Cloud
   - **NEW:** Robust path handling for cloud environments / **NOVO:** Tratamento robusto de caminhos para ambientes de nuvem

4. **Error Handling / Tratamento de Erros:**
   - **NEW:** Automatic fallback if model loading fails
   - **NOVO:** Fallback automático se carregamento do modelo falhar
   - App continues to work with reduced accuracy
   - App continua funcionando com precisão reduzida

## 🏆 Success Metrics / Métricas de Sucesso

- ✅ **Independent deployment** / Deploy independente
- ✅ **Bilingual interface** / Interface bilíngue  
- ✅ **Model performance maintained** / Performance do modelo mantida
- ✅ **Fast loading** < 600ms / Carregamento rápido
- ✅ **Minimal dependencies** / Dependências mínimas
- ✅ **NEW:** **Robust error handling** / **NOVO:** **Tratamento robusto de erros**
- ✅ **NEW:** **Graceful degradation** / **NOVO:** **Degradação elegante**

## 🐛 Issues Fixed / Problemas Corrigidos

### ✅ Model Loading Issues:
- **Problem:** Model file not found in Streamlit Cloud
- **Solution:** Multiple path detection + included model file + fallback system

### ✅ Path Issues:
- **Problem:** Different path structures in development vs production
- **Solution:** Robust path checking with multiple candidates

### ✅ Error Handling:
- **Problem:** App crashes if model unavailable
- **Solution:** Automatic fallback to dummy model with user notification

---

**Ready for production deployment! 🚀**
**Pronto para deploy em produção! 🚀** 

**All tests passing ✅ | Robust fallback system ✅ | Bilingual support ✅** 