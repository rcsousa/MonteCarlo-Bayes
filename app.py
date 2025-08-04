import streamlit as st
from simulation import run_simulation_with_temporal_learning
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

st.subheader("📈 Simulação com Cadeia de Markov + Aprendizado Temporal")
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
