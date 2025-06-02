import streamlit as st

def render_input_form():
    """
    Renderiza formulário de entrada para dados do segurado
    
    Returns:
        dict: Dicionário com os dados preenchidos pelo usuário
    """
    
    # Criar formulário
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Idade
            age = st.number_input(
                "👤 Idade",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="Idade do segurado em anos"
            )
            
            # BMI
            bmi = st.number_input(
                "⚖️ BMI (Índice de Massa Corporal)",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1,
                format="%.1f",
                help="Índice de Massa Corporal (peso/altura²)"
            )
            
            # Número de filhos
            children = st.number_input(
                "👶 Número de Filhos",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Número de filhos dependentes"
            )
        
        with col2:
            # Sexo
            sex = st.selectbox(
                "👥 Gênero",
                options=["male", "female"],
                format_func=lambda x: "Masculino" if x == "male" else "Feminino",
                help="Gênero do segurado"
            )
            
            # Fumante
            smoker = st.selectbox(
                "🚬 Fumante",
                options=["no", "yes"],
                format_func=lambda x: "Não" if x == "no" else "Sim",
                help="Indica se o segurado é fumante"
            )
            
            # Região
            region = st.selectbox(
                "📍 Região",
                options=["northeast", "northwest", "southeast", "southwest"],
                format_func=lambda x: {
                    "northeast": "Nordeste",
                    "northwest": "Noroeste", 
                    "southeast": "Sudeste",
                    "southwest": "Sudoeste"
                }[x],
                help="Região geográfica do segurado"
            )
        
        # Adicionar informações sobre BMI
        st.markdown("---")
        st.markdown("**💡 Referência de BMI:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("Baixo peso: < 18.5")
        with col2:
            st.caption("Normal: 18.5 - 24.9")
        with col3:
            st.caption("Sobrepeso: 25.0 - 29.9")
        with col4:
            st.caption("Obesidade: ≥ 30.0")
        
        # Botão de submit não necessário aqui, será tratado na página principal
        submitted = st.form_submit_button("📝 Validar Dados", type="secondary")
    
    # Validação dos dados
    if submitted:
        # Criar dicionário com os dados
        user_data = {
            'age': int(age),
            'sex': sex,
            'bmi': float(bmi),
            'children': int(children),
            'smoker': smoker,
            'region': region
        }
        
        # Validações adicionais
        validation_errors = []
        
        if age < 18 or age > 100:
            validation_errors.append("Idade deve estar entre 18 e 100 anos")
        
        if bmi < 10 or bmi > 60:
            validation_errors.append("BMI deve estar entre 10 e 60")
        
        if children < 0 or children > 10:
            validation_errors.append("Número de filhos deve estar entre 0 e 10")
        
        # Mostrar erros se houver
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
            return None
        else:
            st.success("✅ Dados validados com sucesso!")
            return user_data
    
    # Retornar None se formulário não foi submetido ainda
    return None

def get_bmi_category(bmi):
    """
    Classifica o BMI em categorias
    
    Args:
        bmi (float): Valor do BMI
        
    Returns:
        str: Categoria do BMI
    """
    if bmi < 18.5:
        return "Baixo peso"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Sobrepeso"
    else:
        return "Obesidade"

def get_age_category(age):
    """
    Classifica a idade em categorias
    
    Args:
        age (int): Idade em anos
        
    Returns:
        str: Categoria da idade
    """
    if age < 25:
        return "Jovem"
    elif age < 35:
        return "Jovem adulto"
    elif age < 50:
        return "Adulto"
    elif age < 65:
        return "Adulto maduro"
    else:
        return "Idoso"

def show_data_summary(user_data):
    """
    Mostra um resumo dos dados inseridos
    
    Args:
        user_data (dict): Dados do usuário
    """
    st.markdown("### 📋 Resumo dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Idade**: {user_data['age']} anos ({get_age_category(user_data['age'])})")
        st.write(f"**Gênero**: {'Masculino' if user_data['sex'] == 'male' else 'Feminino'}")
        st.write(f"**BMI**: {user_data['bmi']:.1f} ({get_bmi_category(user_data['bmi'])})")
    
    with col2:
        st.write(f"**Filhos**: {user_data['children']}")
        st.write(f"**Fumante**: {'Sim' if user_data['smoker'] == 'yes' else 'Não'}")
        region_map = {
            "northeast": "Nordeste",
            "northwest": "Noroeste", 
            "southeast": "Sudeste",
            "southwest": "Sudoeste"
        }
        st.write(f"**Região**: {region_map[user_data['region']]}") 