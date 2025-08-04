import numpy as np
import pandas as pd
from parameters import parameters, states
from scipy.stats import beta

def run_simulation(n_gerentes=27000, n_months=36, transition_matrix=None):
    if transition_matrix is None:
        transition_matrix = [
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.85, 0.15, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]

    priors = {
        k: beta.rvs(p["alpha"], p["beta"])
        for k, p in parameters.items()
    }

    state_vector = np.zeros((n_months, len(states)))
    state_vector[0, 0] = 1.0  # Todos começam em S0

    for t in range(1, n_months):
        state_vector[t] = np.dot(state_vector[t-1], transition_matrix)

    state_counts = (state_vector[-1] * n_gerentes).astype(int)
    state_distribution = state_counts / n_gerentes

    final_mean = np.sum([
        s["multiplicador"] * state_distribution[i]
        for i, s in enumerate(states)
    ]) * 2000  # Cada gerente começa com 2000 contas

    total_capacity = final_mean * n_gerentes

    df_monthly = pd.DataFrame({
        "Mês": list(range(n_months)),
        "Contas por Gerente (média)": [
            np.sum([
                states[i]["multiplicador"] * state_vector[t][i]
                for i in range(len(states))
            ]) * 2000 for t in range(n_months)
        ]
    })

    return {
        "df_monthly": df_monthly,
        "final_mean_accounts": final_mean,
        "total_capacity": total_capacity,
        "state_distribution": state_distribution
    }
