import streamlit as st
from simulation import run_simulation_with_temporal_learning, run_monte_carlo_analysis, calculate_scenario_probabilities, analyze_risk_metrics
from parameters import parameters, states
from inference import update_prior
from utils import show_parameter_note, show_state_note
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="Simulador Bayesiano de Impacto da IA", layout="wide")
st.title("📊 Simulador Bayesiano de Adoção de IA com Modelos Causais + Markov")

# Sistema de abas principal
tab1, tab2, tab3 = st.tabs(["🎲 Simulação Monte Carlo", "⚙️ Configurações", "📚 Benchmarks & Teoria"])

def build_transition_matrix():
    st.sidebar.markdown("### 🔄 Matriz de Transição entre Estados (Markov)")
    
    # NOVA MATRIZ: Mais volátil para refletir natureza disruptiva da IA
    default_matrix = [
        [0.60, 0.35, 0.05, 0.00, 0.00],  # S0: Possibilidade de "saltos"
        [0.00, 0.65, 0.30, 0.05, 0.00],  # S1: Mais progressão rápida
        [0.00, 0.00, 0.70, 0.25, 0.05],  # S2: Aceleração possível  
        [0.00, 0.00, 0.00, 0.80, 0.20],  # S3: Transformação mais rápida
        [0.00, 0.00, 0.00, 0.00, 1.00]   # S4: Estado absorvente
    ]
    
    # Adiciona explicação da nova matriz
    st.sidebar.info("""
    🚨 **MATRIZ ALTA VOLATILIDADE v3.0**
    
    **📚 Justificativa Teórica:**
    
    **1. DISRUPÇÃO TECNOLÓGICA:**
    - Tecnologias disruptivas não seguem progressão linear
    - "Saltos" são possíveis (S0→S2: 5%)
    - Base: Christensen "Innovator's Dilemma"
    
    **2. NETWORK EFFECTS (IA):**
    - Efeito viral quando IA "pega" na organização
    - Aceleração exponencial vs. fracasso total
    - Base: Metcalfe's Law aplicado à adoção
    
    **3. TIPPING POINT THEORY (Gladwell):**
    - Mudanças graduais → transformação súbita
    - 20-25% transições vs. 10-15% anteriores
    - Reflete "momento de virada" organizacional
    
    **✅ Resultado:** Trajetórias mais imprevisíveis
    """)

    if "reset_matrix" not in st.session_state:
        st.session_state.reset_matrix = False

    if st.sidebar.button("🔁 Resetar para benchmark v3.0"):
        st.session_state.reset_matrix = True

    rows = []
    for i, state in enumerate(states):
        probs = []
        st.sidebar.markdown(f"**{state['nome']} → ...**")
        row_sum = 0
        for j in range(len(states)):
            if i > j:
                probs.append(0.0)
                continue
            key = f"{i}_{j}"
            default = default_matrix[i][j]
            val = st.sidebar.slider(
                f"{state['nome']} → {states[j]['nome']}",
                0.0, 1.0, default if st.session_state.reset_matrix else default,
                0.01, key=key,
                help="Probabilidade de transição mensal entre estágios de adoção de IA"
            )
            probs.append(val)
        row_sum = sum(probs)
        probs = [p / row_sum if row_sum > 0 else 0.0 for p in probs]
        rows.append(probs)

    st.session_state.reset_matrix = False
    return rows

st.sidebar.header("🧪 Atualização dos Priors com Evidência")
prior_name = st.sidebar.selectbox("Parâmetro", list(parameters.keys()))
successes = st.sidebar.number_input("Sucessos observados", 0, 1000, 20)
trials = st.sidebar.number_input("Total de experimentos", 1, 1000, 30)
if st.sidebar.button("Atualizar Prior"):
    updated = update_prior(prior_name, successes, trials)
    parameters[prior_name]["alpha"] = updated["new_alpha"]
    parameters[prior_name]["beta"] = updated["new_beta"]
    st.sidebar.success(f"Prior atualizado: Beta({updated['new_alpha']}, {updated['new_beta']})")

st.sidebar.header("🎛️ Controles")
n_gerentes = st.sidebar.slider("Número de gerentes", 1000, 50000, 27000, step=1000)
n_meses = st.sidebar.slider("Horizonte (meses)", 6, 60, 36)

# Nova opção para habilitar/desabilitar aprendizado temporal
learning_enabled = st.sidebar.checkbox(
    "🧠 Aprendizado Temporal Bayesiano", 
    value=True,
    help="Se habilitado, os posteriores de cada mês se tornam priors do próximo mês"
)

# Controles de simulação
st.sidebar.header("🎯 Configuração da Simulação")

n_simulations = st.sidebar.slider(
    "Número de simulações Monte Carlo", 
    100, 2000, 500, step=100,
    help="Mais simulações = maior precisão, mas tempo maior"
)

# Cenários alvo para análise
st.sidebar.subheader("📊 Cenários Alvo")
st.sidebar.markdown("*Defina os targets de capacidade para análise probabilística*")

scenario_1 = st.sidebar.number_input(
    "🎯 Cenário Conservador", 
    min_value=2000, 
    max_value=20000, 
    value=2500,
    step=100,
    help="Target conservador - expectativa mínima realista"
)

scenario_2 = st.sidebar.number_input(
    "🎯 Cenário Moderado", 
    min_value=2000, 
    max_value=20000, 
    value=4000,
    step=100,
    help="Target moderado - expectativa provável com IA"
)

scenario_3 = st.sidebar.number_input(
    "🎯 Cenário Otimista", 
    min_value=2000, 
    max_value=20000, 
    value=7000,
    step=100,
    help="Target otimista - máximo potencial com IA avançada"
)

target_scenarios = [scenario_1, scenario_2, scenario_3]

# Validação dos cenários
if scenario_1 >= scenario_2 or scenario_2 >= scenario_3:
    st.sidebar.warning("⚠️ Os cenários devem estar em ordem crescente: Conservador < Moderado < Otimista")

# Botão para executar simulação
run_simulation = st.sidebar.button(
    "🚀 Executar Simulação Monte Carlo",
    type="primary",
    help="Executa análise probabilística completa com múltiplas simulações estocásticas"
)

custom_matrix = build_transition_matrix()

st.markdown("---")
st.markdown("<h3 style='color:#336699;'>📘 Notas Explicativas e Benchmarks</h3>", unsafe_allow_html=True)

st.markdown("<h4 style='color:#1A5276;'>🎯 Parâmetros do Modelo Bayesiano</h4>", unsafe_allow_html=True)
for name, param in parameters.items():
    show_parameter_note(name, param)

st.markdown("<h4 style='color:#1A5276;'>🔄 Estados de Adoção (Markov)</h4>", unsafe_allow_html=True)
for state in states:
    show_state_note(state)

st.markdown("---")

# Execução da simulação Monte Carlo
if run_simulation:
    st.subheader("🎲 Análise Probabilística Monte Carlo")
    
    # Progress bar para simulações
    with st.spinner(f'Executando {n_simulations} simulações estocásticas...'):
        monte_carlo_results = run_monte_carlo_analysis(
            n_gerentes=n_gerentes,
            n_months=n_meses,
            transition_matrix=custom_matrix,
            learning_enabled=learning_enabled,
            n_simulations=n_simulations
        )
    
    # Calcula probabilidades dos cenários
    scenario_probs = calculate_scenario_probabilities(monte_carlo_results, target_scenarios)
    
    # Análise de riscos
    baseline = 2000  # Capacidade sem IA
    risk_metrics = analyze_risk_metrics(monte_carlo_results, baseline=baseline)
    
    # === VISUALIZAÇÕES ===
    
    # 1. Gráfico com intervalos de confiança
    st.subheader("📈 Projeção com Intervalos de Confiança")
    
    monthly_data = pd.DataFrame({
        'Mês': range(n_meses),
        'P5': monte_carlo_results["monthly_percentiles"]["p5"],
        'P25': monte_carlo_results["monthly_percentiles"]["p25"], 
        'Mediana': monte_carlo_results["monthly_percentiles"]["p50"],
        'P75': monte_carlo_results["monthly_percentiles"]["p75"],
        'P95': monte_carlo_results["monthly_percentiles"]["p95"]
    })
    
    # Gráfico principal com bandas de confiança
    base_chart = alt.Chart(monthly_data).add_params(
        alt.selection_interval(bind='scales')
    )
    
    # Banda 90% (P5-P95)
    area_90 = base_chart.mark_area(
        opacity=0.2, color='blue'
    ).encode(
        x='Mês:Q',
        y='P5:Q',
        y2='P95:Q'
    )
    
    # Banda 50% (P25-P75)  
    area_50 = base_chart.mark_area(
        opacity=0.4, color='blue'
    ).encode(
        x='Mês:Q',
        y='P25:Q', 
        y2='P75:Q'
    )
    
    # Linha da mediana
    median_line = base_chart.mark_line(
        color='red', strokeWidth=3
    ).encode(
        x='Mês:Q',
        y='Mediana:Q'
    )
    
    confidence_chart = (area_90 + area_50 + median_line).resolve_scale(
        y='independent'
    ).properties(
        width=800, height=400,
        title="Projeção de Capacidade com Intervalos de Confiança (90% e 50%)"
    )
    
    st.altair_chart(confidence_chart, use_container_width=True)
    
    # 2. Métricas de Cenários
    st.subheader("🎯 Probabilidade dos Cenários")
    
    # Explicação dos cenários
    with st.expander("ℹ️ Interpretação dos Cenários"):
        st.markdown("""
        **🎯 Cenário Conservador:** Representa o mínimo esperado com implementação básica de IA. 
        Considera resistência organizacional e adoção lenta.
        
        **🎯 Cenário Moderado:** Expectativa realista com boa implementação de IA e treinamento adequado. 
        Representa o resultado mais provável.
        
        **🎯 Cenário Otimista:** Máximo potencial com transformação digital completa e IA avançada. 
        Requer excelência em todos os fatores organizacionais.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    scenarios = ["Conservador", "Moderado", "Otimista"]
    for i, (col, scenario_name, target) in enumerate(zip([col1, col2, col3], scenarios, target_scenarios)):
        with col:
            prob_exceed = scenario_probs[f"P(>= {target})"]
            prob_within = scenario_probs[f"P(±5% de {target})"]
            
            st.metric(
                f"🎯 {scenario_name} ({target})",
                f"{prob_exceed:.1%}",
                help=f"Probabilidade de atingir ou superar {target} contas/gerente"
            )
            st.write(f"📍 Precisão ±5%: {prob_within:.1%}")
    
    # 3. Análise de Riscos
    st.subheader("⚠️ Análise de Riscos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🛡️ VaR 95%", 
            f"{risk_metrics['var_95']:.0f}",
            help="Pior cenário em 95% dos casos"
        )
    
    with col2:
        st.metric(
            "📉 Risco de Não-Ganho",
            f"{risk_metrics['prob_no_gain']:.1%}",
            help="Probabilidade de não superar baseline"
        )
    
    with col3:
        st.metric(
            "📊 Volatilidade",
            f"{risk_metrics['coefficient_variation']:.1%}",
            help="Coeficiente de variação (incerteza relativa)"
        )
    
    with col4:
        st.metric(
            "📈 Expectativa",
            f"{risk_metrics['mean']:.0f}",
            delta=f"{risk_metrics['mean'] - risk_metrics['baseline']:.0f}"
        )
    
    # 4. Distribuição da Capacidade Final
    st.subheader("📊 Distribuição da Capacidade Final")
    
    hist_data = pd.DataFrame({
        'Capacidade Final': monte_carlo_results["final_capacities"]
    })
    
    histogram = alt.Chart(hist_data).mark_bar(
        opacity=0.7
    ).encode(
        x=alt.X('Capacidade Final:Q', bin=alt.Bin(maxbins=30), title='Contas por Gerente'),
        y=alt.Y('count()', title='Frequência'),
        color=alt.value('steelblue')
    ).properties(
        width=600, height=300,
        title=f'Distribuição da Capacidade Final ({n_simulations} simulações)'
    )
    
    # Adiciona linhas verticais para cenários
    scenario_lines = []
    colors = ['red', 'orange', 'green']
    for target, color, name in zip(target_scenarios, colors, scenarios):
        line = alt.Chart(pd.DataFrame({'target': [target]})).mark_rule(
            color=color, strokeWidth=2, strokeDash=[5,5]
        ).encode(
            x='target:Q'
        )
        scenario_lines.append(line)
    
    final_chart = histogram
    for line in scenario_lines:
        final_chart += line
    
    st.altair_chart(final_chart, use_container_width=True)
    
    # 5. Interpretação Executiva
    st.subheader("💼 Interpretação Executiva")
    
    mean_gain = risk_metrics['mean'] - risk_metrics['baseline']
    confidence_level = (1 - risk_metrics['prob_no_gain']) * 100
    
    # Determina qual cenário é mais provável
    prob_conservador = scenario_probs[f"P(>= {scenario_1})"]
    prob_moderado = scenario_probs[f"P(>= {scenario_2})"]
    prob_otimista = scenario_probs[f"P(>= {scenario_3})"]
    
    if prob_conservador >= 0.8:
        cenario_provavel = f"🎯 **Conservador** ({scenario_1} contas)"
        cor_cenario = "green"
    elif prob_moderado >= 0.5:
        cenario_provavel = f"🎯 **Moderado** ({scenario_2} contas)"
        cor_cenario = "orange"
    elif prob_otimista >= 0.2:
        cenario_provavel = f"🎯 **Otimista** ({scenario_3} contas)"
        cor_cenario = "red"
    else:
        cenario_provavel = "🎯 **Customizado** (entre cenários definidos)"
        cor_cenario = "blue"
    
    st.info(f"""
    **📈 Projeção Central:** {risk_metrics['mean']:.0f} contas/gerente (ganho de {mean_gain:.0f} contas, +{mean_gain/risk_metrics['baseline']:.1%})
    
    **🎯 Cenário Mais Provável:** {cenario_provavel}
    
    **🛡️ Confiabilidade:** {confidence_level:.1f}% de chance de ganho positivo sobre baseline ({baseline} contas)
    
    **📊 Faixa Esperada:** {risk_metrics['var_90']:.0f} - {monte_carlo_results['final_stats']['p75']:.0f} contas/gerente (80% dos cenários)
    
    **⚠️ Cenário Pessimista:** {risk_metrics['var_95']:.0f} contas/gerente (pior 5% dos casos)
    
    **🚀 Potencial Máximo:** {monte_carlo_results['final_stats']['p95']:.0f} contas/gerente (melhor 5% dos casos)
    """)
    
    # Recomendações baseadas nos resultados
    st.subheader("💡 Recomendações Estratégicas")
    
    if prob_conservador >= 0.9:
        st.success(f"""
        ✅ **Alta Probabilidade de Sucesso:** {prob_conservador:.1%} chance de atingir o cenário conservador.
        - Projeto tem baixo risco e alta viabilidade
        - Recomenda-se prosseguir com implementação
        """)
    elif prob_moderado >= 0.7:
        st.warning(f"""
        ⚠️ **Sucesso Moderado Esperado:** {prob_moderado:.1%} chance do cenário moderado.
        - Projeto viável com riscos controláveis
        - Considere investimentos adicionais em treinamento
        """)
    else:
        st.error(f"""
        🚨 **Alto Risco:** Baixa probabilidade dos cenários definidos.
        - Revisar estratégia de implementação
        - Considerar abordagem mais conservadora ou investimentos maiores
        """)

else:
    # Estado inicial - aguardando configuração
    st.info(f"""
    ### 👆 Configure os parâmetros na barra lateral e clique em "🚀 Executar Simulação Monte Carlo"
    
    **📋 O que será executado:**
    - ✅ {n_simulations} simulações estocásticas independentes
    - ✅ Análise probabilística completa com intervalos de confiança
    - ✅ Cálculo de probabilidades para os cenários definidos
    - ✅ Análise de riscos e recomendações estratégicas
    - ✅ Aprendizado temporal bayesiano em cada simulação
    
    **⏱️ Tempo estimado:** {n_simulations // 100} a {n_simulations // 50} segundos
    """)
