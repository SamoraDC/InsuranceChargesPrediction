# 🚀 Guia Completo: Como Rodar o Sistema de Predição de Seguros

Este guia te ensina como executar o sistema completo, do treinamento até a aplicação Streamlit.

## 📋 Pré-requisitos

### 1. Verificar Estrutura do Projeto
```bash
# Verificar se está na raiz do projeto
pwd
# Deve mostrar: /workspaces/InsuranceChargesPrediction

# Verificar arquivos principais
ls -la
# Deve conter: src/, scripts/, data/, models/, app_new.py, requirements.txt
```

### 2. Instalar Dependências
```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Verificar instalação do Streamlit
streamlit --version
```

### 3. Verificar Dados
```bash
# Verificar se o arquivo de dados existe
ls -la data/raw/insurance.csv
# Deve mostrar o arquivo com ~54KB
```

## 🎯 PASSO 1: Treinar o Modelo

### Opção A: Treinamento Completo (Recomendado)
```bash
# Treinamento completo com otimização de hiperparâmetros
python scripts/train_model.py

# Saída esperada:
# 🚀 INICIANDO TREINAMENTO DO MODELO DE SEGUROS
# ✅ Dados carregados: (1338, 7)
# ✅ Pré-processamento concluído
# 🏆 TREINAMENTO OTIMIZADO CONCLUÍDO!
# R² Final: > 0.85
```

### Opção B: Treinamento Rápido (Para testes)
```bash
# Treinamento sem otimização (mais rápido)
python scripts/train_model.py --no-optimize
```

### Verificar se o Modelo foi Treinado
```bash
# Verificar arquivos gerados
ls -la models/
# Deve conter: gradient_boosting_model.pkl

ls -la models/model_artifacts/
# Deve conter: preprocessor.pkl
```

## 🔮 PASSO 2: Testar Predição Via Código

### Teste Simples
```python
# Executar no terminal Python
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

from insurance_prediction.models.predictor import predict_insurance_premium

# Teste de predição
result = predict_insurance_premium(
    age=35,
    sex='male',
    bmi=25.0,
    children=2,
    smoker='no',
    region='northeast'
)

print(f'Prêmio previsto: \${result[\"predicted_premium\"]:,.2f}')
print(f'Modelo: {result[\"model_type\"]}')
print(f'Tempo: {result[\"processing_time_ms\"]:.2f}ms')
"
```

### Saída Esperada
```
Prêmio previsto: $4,500.23
Modelo: GradientBoostingRegressor
Tempo: 15.45ms
```

## 🌐 PASSO 3: Executar Aplicação Streamlit

### Iniciar a Aplicação
```bash
# Executar a aplicação atualizada
streamlit run app_new.py

# OU se preferir usar a aplicação original (pode dar erro)
# streamlit run app.py
```

### Acessar a Aplicação
- **URL**: http://localhost:8501
- **Porta**: 8501 (padrão do Streamlit)

### Se Não Abrir Automaticamente
```bash
# Verificar se está rodando
ps aux | grep streamlit

# Verificar porta
netstat -tlnp | grep 8501

# Acessar manualmente
# http://localhost:8501
# ou
# http://127.0.0.1:8501
```

## 🧪 PASSO 4: Testar a Aplicação

### 4.1 Teste de Predição Individual

1. **Acesse a aba "🎯 Predição Individual"**

2. **Preencha os dados do segurado:**
   - **Idade**: 35 anos
   - **Gênero**: Masculino
   - **BMI**: 25.0
   - **Filhos**: 2
   - **Fumante**: Não
   - **Região**: Nordeste

3. **Clique em "🔮 Calcular Prêmio"**

4. **Resultado esperado:**
   - Prêmio anual: ~$4,500-6,000
   - Valor mensal: ~$375-500
   - Tempo de processamento: <50ms
   - Análise de risco automática

### 4.2 Teste com Fumante (Alto Risco)

1. **Altere apenas:**
   - **Fumante**: Sim

2. **Resultado esperado:**
   - Prêmio anual: ~$20,000-25,000 (muito maior!)
   - Alerta de "ALTO RISCO"
   - Gráfico comparativo

### 4.3 Teste de Análise em Lote

1. **Acesse a aba "📊 Análise em Lote"**

2. **Baixe o template CSV**

3. **Faça upload do arquivo**

4. **Processe as predições**

5. **Baixe os resultados**

## 🛠️ RESOLUÇÃO DE PROBLEMAS

### Problema 1: Erro "Modelo não encontrado"
```bash
# Verificar se o modelo existe
ls -la models/gradient_boosting_model.pkl

# Se não existir, treinar novamente
python scripts/train_model.py
```

### Problema 2: Erro de Import
```bash
# Verificar PYTHONPATH
echo $PYTHONPATH

# Adicionar src ao path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Verificar estrutura dos módulos
ls -la src/insurance_prediction/
```

### Problema 3: Streamlit não carrega
```bash
# Verificar se Streamlit está instalado
pip list | grep streamlit

# Reinstalar se necessário
pip install streamlit==1.28.0

# Verificar se a porta está em uso
lsof -i :8501

# Usar porta diferente se necessário
streamlit run app_new.py --server.port 8502
```

### Problema 4: Erro no Preprocessor
```bash
# Verificar se o preprocessor existe
ls -la models/model_artifacts/preprocessor.pkl

# Se houver erro, regenerar preprocessor
python scripts/train_model.py --no-optimize
```

### Problema 5: Dados não encontrados
```bash
# Verificar arquivo de dados
ls -la data/raw/insurance.csv

# Se não existir, baixar dados ou ajustar caminho
head -5 data/raw/insurance.csv
```

## 📊 COMANDOS ÚTEIS

### Verificar Performance do Sistema
```bash
# Testar tempo de carregamento
time python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))
from insurance_prediction.models.predictor import load_production_model
predictor = load_production_model()
print('Modelo carregado!')
"
```

### Logs do Sistema
```bash
# Ver logs do treinamento
tail -f logs/insurance_prediction.log

# Ver logs em tempo real durante execução
tail -f logs/insurance_prediction.log | grep -i "prediction"
```

### Limpar Cache do Streamlit
```bash
# Limpar cache do Streamlit
streamlit cache clear

# Remover arquivos temporários
rm -rf ~/.streamlit/
```

## 🎯 FLUXO COMPLETO DE TESTE

### Teste End-to-End
```bash
# 1. Limpar modelos antigos (opcional)
# rm -f models/gradient_boosting_model.pkl
# rm -f models/model_artifacts/preprocessor.pkl

# 2. Treinar modelo novo
python scripts/train_model.py --no-optimize

# 3. Testar predição via código
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))
from insurance_prediction.models.predictor import predict_insurance_premium
result = predict_insurance_premium(age=30, sex='female', bmi=22.0, children=1, smoker='yes', region='southeast')
print(f'Prêmio para fumante: \${result[\"predicted_premium\"]:,.2f}')
"

# 4. Executar Streamlit
streamlit run app_new.py

# 5. Testar no navegador (localhost:8501)
```

## 📈 MÉTRICAS DE SUCESSO

### Indicadores de que tudo está funcionando:

✅ **Modelo treinado**: R² > 0.80  
✅ **Predição rápida**: <100ms  
✅ **Streamlit carrega**: Sem erros  
✅ **Predições consistentes**: Fumantes pagam 3-4x mais  
✅ **Interface responsiva**: Gráficos carregam corretamente

### Performance Esperada:
- **Não fumantes**: $3,000-8,000/ano
- **Fumantes**: $15,000-35,000/ano
- **Tempo de resposta**: 10-50ms
- **Acurácia**: R² > 0.85

## 🎉 CONCLUSÃO

Se todos os passos foram executados com sucesso, você terá:

1. ✅ **Sistema completo funcionando**
2. ✅ **Modelo Gradient Boosting treinado e otimizado**
3. ✅ **Aplicação Streamlit moderna e responsiva**
4. ✅ **API de predição robusta**
5. ✅ **Interface amigável para usuários**

🚀 **Seu sistema de predição de seguros está pronto para produção!** 