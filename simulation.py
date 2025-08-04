import numpy as np
import pandas as pd
from parameters import parameters, states
from scipy.stats import beta
import copy

def observe_monthly_evidence(state_vector_prev, state_vector_curr, month):
    """
    Observa evidências do mês baseadas na progressão dos gerentes entre estados.
    Converte progressões em evidências bayesianas para atualizar os priors.
    
    Args:
        state_vector_prev: Distribuição de estados no mês anterior
        state_vector_curr: Distribuição de estados no mês atual
        month: Mês atual da simulação
    
    Returns:
        dict: Evidências observadas para cada parâmetro bayesiano
    """
    evidence = {}
    
    # 1. AI_Investment: Baseado na progressão geral S0→S1, S1→S2
    early_progression = (state_vector_curr[1] + state_vector_curr[2]) - (state_vector_prev[1] + state_vector_prev[2])
    ai_investment_success_rate = max(0.1, min(0.9, 0.5 + early_progression * 2))
    
    evidence["AI_Investment"] = {
        "successes": int(ai_investment_success_rate * 50),
        "trials": 50
    }
    
    # 2. Change_Adoption: Baseado na velocidade de transição S1→S2, S2→S3
    if month > 3:  # Só após alguns meses para ter dados significativos
        mid_progression = (state_vector_curr[2] + state_vector_curr[3]) - (state_vector_prev[2] + state_vector_prev[3])
        change_adoption_rate = max(0.1, min(0.9, 0.5 + mid_progression * 3))
        
        evidence["Change_Adoption"] = {
            "successes": int(change_adoption_rate * 40),
            "trials": 40
        }
    
    # 3. Training_Quality: Baseado na chegada ao estado avançado S3, S4
    if month > 6:  # Só após tempo suficiente para treinamento mostrar efeito
        advanced_progression = (state_vector_curr[3] + state_vector_curr[4]) - (state_vector_prev[3] + state_vector_prev[4])
        training_quality_rate = max(0.1, min(0.9, 0.6 + advanced_progression * 4))
        
        evidence["Training_Quality"] = {
            "successes": int(training_quality_rate * 30),
            "trials": 30
        }
    
    return evidence

def update_posterior_params(current_params, evidence):
    """
    Atualiza parâmetros bayesianos com base nas evidências observadas.
    Implementa a fórmula: Posterior = Beta(α + sucessos, β + fracassos)
    
    Args:
        current_params: Parâmetros atuais (com alpha e beta)
        evidence: Evidências observadas no mês
    
    Returns:
        dict: Parâmetros atualizados para uso no próximo mês
    """
    updated_params = copy.deepcopy(current_params)
    
    for param_name, obs in evidence.items():
        if param_name in updated_params:
            # Atualização Bayesiana: α_new = α + sucessos, β_new = β + fracassos
            updated_params[param_name]["alpha"] += obs["successes"]
            updated_params[param_name]["beta"] += (obs["trials"] - obs["successes"])
    
    return updated_params

def run_simulation_with_temporal_learning(n_gerentes=27000, n_months=36, transition_matrix=None, learning_enabled=True):
    """
    Executa simulação com aprendizado temporal bayesiano.
    Os posteriores de cada mês se tornam os priors do mês seguinte.
    
    Args:
        n_gerentes: Número de gerentes na simulação
        n_months: Horizonte temporal em meses
        transition_matrix: Matriz de transição de Markov
        learning_enabled: Se True, aplica aprendizado temporal; se False, usa método original
    
    Returns:
        dict: Resultados da simulação incluindo evolução dos parâmetros
    """
    if transition_matrix is None:
        transition_matrix = [
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.85, 0.15, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]

    # Inicialização dos parâmetros (cópia para não modificar original)
    current_params = copy.deepcopy(parameters)
    
    # Estruturas para armazenar evolução
    state_vector = np.zeros((n_months, len(states)))
    state_vector[0, 0] = 1.0  # Todos começam em S0
    
    monthly_capacities = []
    params_evolution = []  # Para rastrear evolução dos parâmetros
    evidences_log = []  # Para rastrear evidências observadas
    
    # Simulação mês a mês com aprendizado
    for month in range(n_months):
        # 1. Amostra parâmetros com distribuições atuais
        current_priors = {
            k: beta.rvs(p["alpha"], p["beta"])
            for k, p in current_params.items()
        }
        
        # 2. Registra estado dos parâmetros
        params_evolution.append({
            param: {
                "alpha": current_params[param]["alpha"],
                "beta": current_params[param]["beta"],
                "mean": current_params[param]["alpha"] / (current_params[param]["alpha"] + current_params[param]["beta"]),
                "sampled_value": current_priors[param]
            }
            for param in current_params
        })
        
        # 3. Evolução da cadeia de Markov
        if month > 0:
            state_vector[month] = np.dot(state_vector[month-1], transition_matrix)
        
        # 4. Calcula capacidade do mês
        monthly_capacity = np.sum([
            states[i]["multiplicador"] * state_vector[month][i]
            for i in range(len(states))
        ]) * 2000
        
        monthly_capacities.append(monthly_capacity)
        
        # 5. Observa evidências e atualiza parâmetros (só se learning habilitado)
        if learning_enabled and month > 0:
            evidence = observe_monthly_evidence(
                state_vector[month-1], 
                state_vector[month], 
                month
            )
            evidences_log.append(evidence)
            
            # Atualiza parâmetros para próximo mês
            current_params = update_posterior_params(current_params, evidence)
    
    # Resultados finais
    state_counts = (state_vector[-1] * n_gerentes).astype(int)
    state_distribution = state_counts / n_gerentes
    final_mean_accounts = monthly_capacities[-1]
    total_capacity = final_mean_accounts * n_gerentes
    
    df_monthly = pd.DataFrame({
        "Mês": list(range(n_months)),
        "Contas por Gerente (média)": monthly_capacities
    })
    
    return {
        "df_monthly": df_monthly,
        "final_mean_accounts": final_mean_accounts,
        "total_capacity": total_capacity,
        "state_distribution": state_distribution,
        "params_evolution": params_evolution,
        "evidences_log": evidences_log,
        "learning_enabled": learning_enabled
    }

def run_simulation(n_gerentes=27000, n_months=36, transition_matrix=None):
    """
    Função de compatibilidade - executa simulação com aprendizado temporal habilitado.
    Mantém interface original para não quebrar código existente.
    """
    return run_simulation_with_temporal_learning(
        n_gerentes=n_gerentes, 
        n_months=n_months, 
        transition_matrix=transition_matrix, 
        learning_enabled=True
    )
