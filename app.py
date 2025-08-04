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

def build_transition_matrix():
    st.sidebar.markdown("### 🔄 Matriz de Transição entre Estados (Markov)")
    default_matrix = [
        [0.70, 0.30, 0.00, 0.00, 0.00],
        [0.00, 0.75, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.85, 0.15, 0.00],
        [0.00, 0.00, 0.00, 0.90, 0.10],
        [0.00, 0.00, 0.00, 0.00, 1.00]
    ]

    if "reset_matrix" not in st.session_state:
        st.session_state.reset_matrix = False

    if st.sidebar.button("🔁 Resetar para benchmark"):
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

# Adicionar controles de análise probabilística
st.sidebar.header("🎯 Análise de Cenários")
run_monte_carlo = st.sidebar.checkbox(
    "🎲 Análise Monte Carlo", 
    value=False,
    help="Executa múltiplas simulações para calcular probabilidades e intervalos de confiança"
)

if run_monte_carlo:
    n_simulations = st.sidebar.slider(
        "Número de simulações", 
        100, 2000, 500, step=100,
        help="Mais simulações = maior precisão, mas tempo maior"
    )
    
    # Cenários alvo para análise
    st.sidebar.subheader("📊 Cenários Alvo")
    scenario_1 = st.sidebar.number_input("Cenário Conservador", 2000, 4000, 2200)
    scenario_2 = st.sidebar.number_input("Cenário Moderado", 2000, 4000, 2500) 
    scenario_3 = st.sidebar.number_input("Cenário Otimista", 2000, 4000, 3000)
    
    target_scenarios = [scenario_1, scenario_2, scenario_3]

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

# Execução das simulações
if run_monte_carlo:
    st.subheader("🎲 Análise Probabilística Monte Carlo")
    
    # Progress bar para simulações
    with st.spinner(f'Executando {n_simulations} simulações...'):
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
    risk_metrics = analyze_risk_metrics(monte_carlo_results, baseline=2000)
    
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
    base_chart = alt.Chart(monthly_data).add_selection(
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
        opacity=0.7, binMaxBins=30
    ).encode(
        x=alt.X('Capacidade Final:Q', bin=True, title='Contas por Gerente'),
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
    
    st.info(f"""
    **📈 Projeção Central:** {risk_metrics['mean']:.0f} contas/gerente (ganho de {mean_gain:.0f} contas, +{mean_gain/risk_metrics['baseline']:.1%})
    
    **🎯 Confiabilidade:** {confidence_level:.1f}% de chance de ganho positivo
    
    **📊 Faixa Esperada:** {risk_metrics['var_90']:.0f} - {monte_carlo_results['final_stats']['p75']:.0f} contas/gerente (80% dos cenários)
    
    **⚠️ Cenário Pessimista:** {risk_metrics['var_95']:.0f} contas/gerente (pior 5% dos casos)
    """)

else:
    # Simulação única (comportamento original)
    st.subheader("📈 Simulação Determinística")
    result = run_simulation_with_temporal_learning(
        n_gerentes=n_gerentes, 
        n_months=n_meses, 
        transition_matrix=custom_matrix,
        learning_enabled=learning_enabled
    )

    chart = alt.Chart(result["df_monthly"]).mark_line(point=True).encode(
        x="Mês:Q", y="Contas por Gerente (média):Q"
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

    st.metric("Capacidade média final por gerente", f"{result['final_mean_accounts']:.0f} contas")
    st.metric("Capacidade total estimada", f"{result['total_capacity']:.0f} contas")

    # Nova seção: Evolução dos Parâmetros Bayesianos (só se aprendizado habilitado)
    if result.get("learning_enabled", False):
        st.markdown("---")
        st.subheader("🧠 Evolução dos Parâmetros Bayesianos com Aprendizado Temporal")
        
        if "params_evolution" in result and result["params_evolution"]:
            # Criar DataFrame da evolução dos parâmetros
            evolution_data = []
            for month, params in enumerate(result["params_evolution"]):
                for param_name, param_data in params.items():
                    evolution_data.append({
                        "Mês": month,
                        "Parâmetro": param_name,
                        "Valor Médio": param_data["mean"],
                        "Alpha": param_data["alpha"],
                        "Beta": param_data["beta"],
                        "Valor Amostrado": param_data["sampled_value"]
                    })
            
            df_evolution = pd.DataFrame(evolution_data)
            
            # Gráfico da evolução da média dos parâmetros
            evolution_chart = alt.Chart(df_evolution).mark_line(point=True).encode(
                x="Mês:Q",
                y="Valor Médio:Q",
                color="Parâmetro:N",
                tooltip=["Mês:Q", "Parâmetro:N", "Valor Médio:Q", "Alpha:Q", "Beta:Q"]
            ).properties(
                width=800, 
                height=300,
                title="Evolução da Média dos Parâmetros Bayesianos ao Longo do Tempo"
            )
            
            st.altair_chart(evolution_chart, use_container_width=True)
            
            # Métricas finais dos parâmetros
            final_params = result["params_evolution"][-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ai_inv = final_params["AI_Investment"]
                st.metric(
                    "AI Investment (Final)", 
                    f"{ai_inv['mean']:.1%}",
                    f"Beta({ai_inv['alpha']:.0f}, {ai_inv['beta']:.0f})"
                )
            
            with col2:
                change_adp = final_params["Change_Adoption"]
                st.metric(
                    "Change Adoption (Final)", 
                    f"{change_adp['mean']:.1%}",
                    f"Beta({change_adp['alpha']:.0f}, {change_adp['beta']:.0f})"
                )
            
            with col3:
                train_qual = final_params["Training_Quality"]
                st.metric(
                    "Training Quality (Final)", 
                    f"{train_qual['mean']:.1%}",
                    f"Beta({train_qual['alpha']:.0f}, {train_qual['beta']:.0f})"
                )
            
            # Log de evidências observadas
            if result.get("evidences_log"):
                with st.expander("📋 Log de Evidências Observadas (Últimos 5 meses)"):
                    recent_evidences = result["evidences_log"][-5:]
                    for i, evidence in enumerate(recent_evidences):
                        st.markdown(f"**Mês {len(result['evidences_log']) - len(recent_evidences) + i + 1}:**")
                        for param, obs in evidence.items():
                            success_rate = obs["successes"] / obs["trials"]
                            st.text(f"  {param}: {obs['successes']}/{obs['trials']} sucessos ({success_rate:.1%})")

    df_estados = pd.DataFrame({
        "Estado": [s["nome"] for s in states],
        "Proporção": result["state_distribution"]
    })
    st.bar_chart(df_estados.set_index("Estado"))
