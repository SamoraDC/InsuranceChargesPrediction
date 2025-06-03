# 🚀 Insurance Premium Predictor - Deploy Package
# 🚀 Preditor de Prêmio de Seguro - Pacote de Deploy

## 📝 Overview / Visão Geral

This deploy package contains a **standalone, bilingual version** of the Insurance Premium Predictor for **Streamlit Cloud deployment**.

Este pacote de deploy contém uma **versão standalone e bilíngue** do Preditor de Prêmio de Seguro para **deploy no Streamlit Cloud**.

## 🎯 Phase 1 Implementation / Implementação da Fase 1

### ✅ What's Included / O que está incluído:

1. **🔧 Independent model utilities** / **Utilitários de modelo independentes**
   - `model_utils.py`: Standalone prediction functions / Funções de predição independentes
   - No dependencies on `src/` directory / Sem dependências do diretório `src/`
   - Uses `production_model_optimized.pkl` / Usa `production_model_optimized.pkl`

2. **🌐 Bilingual Streamlit App** / **App Streamlit Bilíngue**
   - `streamlit_app.py`: Complete bilingual interface / Interface bilíngue completa
   - 🇺🇸 English + 🇧🇷 Portuguese support / Suporte a Inglês + Português
   - Language toggle button / Botão de alternância de idioma

3. **📦 Minimal Dependencies** / **Dependências Mínimas**
   - `requirements_deploy.txt`: Only 6 essential packages / Apenas 6 pacotes essenciais
   - Optimized for Streamlit Cloud / Otimizado para Streamlit Cloud

4. **⚙️ Streamlit Configuration** / **Configuração do Streamlit**
   - `.streamlit/config.toml`: Theme and server settings / Configurações de tema e servidor

## 🔄 Model Performance / Performance do Modelo

- **Algorithm / Algoritmo:** Gradient Boosting
- **R² Score:** 0.8795 (87.95%)
- **MAE:** $2,651 (18.6% of mean / da média)
- **Processing Time / Tempo:** < 600ms
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

## 📂 File Structure / Estrutura de Arquivos

```
deploy/
├── streamlit_app.py          # Main bilingual app / App principal bilíngue
├── model_utils.py            # Independent model functions / Funções independentes
├── requirements_deploy.txt   # Minimal dependencies / Dependências mínimas
├── .streamlit/
│   └── config.toml          # Streamlit configuration / Configuração
└── README.md               # This file / Este arquivo

Required model file (from parent directory):
../models/production_model_optimized.pkl
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

### Batch Analysis / Análise em Lote:
- CSV upload support / Suporte a upload de CSV
- Template download / Download de template
- Bulk processing / Processamento em massa
- Results export / Exportação de resultados

### Model Information / Informações do Modelo:
- Performance metrics / Métricas de performance
- Feature importance / Importância das features
- Processing time / Tempo de processamento

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

1. **Model File Required / Arquivo de Modelo Necessário:**
   - Ensure `../models/production_model_optimized.pkl` exists
   - Certifique-se de que `../models/production_model_optimized.pkl` existe

2. **Standalone Operation / Operação Independente:**
   - This package works independently of the `src/` directory
   - Este pacote funciona independentemente do diretório `src/`

3. **Streamlit Cloud Compatibility / Compatibilidade:**
   - Optimized for Streamlit Cloud requirements
   - Otimizado para os requisitos do Streamlit Cloud

## 🏆 Success Metrics / Métricas de Sucesso

- ✅ **Independent deployment** / Deploy independente
- ✅ **Bilingual interface** / Interface bilíngue  
- ✅ **Model performance maintained** / Performance do modelo mantida
- ✅ **Fast loading** < 600ms / Carregamento rápido
- ✅ **Minimal dependencies** / Dependências mínimas

---

**Ready for production deployment! 🚀**
**Pronto para deploy em produção! 🚀** 