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

def apply_bayesian_factors_to_transitions(transition_matrix, sampled_params):
    """
    VERSÃO 3.0: IMPACTO DISRUPTIVO DOS PARÂMETROS BAYESIANOS
    
    Modifica a matriz de transição com base nos parâmetros bayesianos amostrados.
    NOVA IMPLEMENTAÇÃO: Fator disruptivo que reflete natureza da IA.
    
    📚 BASE TEÓRICA:
    
    1. THEORY OF DISRUPTIVE INNOVATION (Christensen, 1997):
       - Tecnologias disruptivas têm impacto não-linear
       - Range: 0.3x (fracasso total) a 3.0x (transformação radical)
    
    2. ROGERS ADOPTION CURVE + AI MULTIPLIER:
       - Parâmetros baixos = "Laggards" → 70% redução velocidade
       - Parâmetros altos = "Innovators" → 200% aceleração
    
    3. NETWORK EFFECTS (Metcalfe's Law):
       - Valor cresce quadraticamente com adoção
       - IA amplifica esse efeito (viral adoption)
    
    Args:
        transition_matrix: Matriz base de transição  
        sampled_params: Parâmetros amostrados das distribuições Beta
    
    Returns:
        np.array: Matriz de transição modificada pelos fatores bayesianos
    """
    # Extrai parâmetros amostrados
    ai_investment = sampled_params["AI_Investment"]
    change_adoption = sampled_params["Change_Adoption"] 
    training_quality = sampled_params["Training_Quality"]
    
    # NOVA FÓRMULA: Fator disruptivo dramático
    # Fator combinado com peso diferencial (IA Investment é mais crítico)
    weighted_factor = (
        ai_investment * 0.4 +       # 40% peso - investimento é fundamental
        change_adoption * 0.35 +    # 35% peso - cultura organizacional crítica  
        training_quality * 0.25     # 25% peso - treinamento suporte
    )
    
    # FATOR DISRUPTIVO: Range 0.3x a 3.0x (vs. 0.75x a 1.25x anterior)
    # Base científica: Christensen + Rogers + Network Effects
    disruption_multiplier = 0.3 + (weighted_factor * 2.7)
    
    # Log para debug (pode ser removido em produção)
    # print(f"Debug: weighted_factor={weighted_factor:.3f}, multiplier={disruption_multiplier:.3f}")
    
    # Aplica transformação na matriz
    modified_matrix = np.array(transition_matrix, dtype=float)
    
    # Multiplica TODAS as transições de progressão pelo fator disruptivo
    for i in range(len(modified_matrix) - 1):  # Não modifica estado absorvente
        for j in range(i + 1, len(modified_matrix[i])):  # Apenas progressões
            if modified_matrix[i][j] > 0:
                # Aplica fator disruptivo
                modified_matrix[i][j] *= disruption_multiplier
                # Limita a 95% (impossível ter certeza absoluta)
                modified_matrix[i][j] = min(0.95, modified_matrix[i][j])
        
        # RENORMALIZAÇÃO: Garante que linha soma 1.0
        row_sum = np.sum(modified_matrix[i])
        if row_sum > 0:
            modified_matrix[i] = modified_matrix[i] / row_sum
    
    return modified_matrix


def add_market_shocks(month, modified_matrix, shock_probability=0.25):
    """
    VERSÃO 3.1: CHOQUES DE MERCADO EXTREMOS
    
    Adiciona choques estocásticos que refletem a EXTREMA volatilidade da IA.
    25% probabilidade mensal = média de 9 choques por 36 meses (muito realista).
    
    📚 BASE TEÓRICA:
    
    1. AI MARKET VOLATILITY (2023-2024 EVIDENCE):
       - ChatGPT: transformou mercado overnight
       - GPT-4: mudou landscape competitivo
       - EU AI Act: regulação súbita 
       - AI safety concerns: cautela organizacional
       - Venture funding: boom-bust cycles
    
    2. PUNCTUATED EQUILIBRIUM + NETWORK EFFECTS:
       - IA não evolui gradualmente
       - Breakthrough → viral adoption OR mass rejection
       - Organizational FOMO vs. fear cycles
    
    3. NASSIM TALEB BLACK SWAN THEORY:
       - Eventos raros, alto impacto são NORMAIS em disrupção
       - IA = technologia de máxima incerteza
       - Fat tail distributions, não gaussiana
    
    Args:
        month: Mês atual da simulação
        modified_matrix: Matriz já modificada pelos parâmetros bayesianos
        shock_probability: Probabilidade mensal de choque (NOVO: 25% vs. 8%)
    
    Returns:
        np.array: Matriz com possível choque aplicado
    """
    # Só aplica choques após mês 2 (período de setup)
    if month < 2:
        return modified_matrix
    
    # Verifica se ocorre choque neste mês (AGORA: 25% chance!)
    if np.random.random() > shock_probability:
        return modified_matrix  # Sem choque
    
    # TIPOS DE CHOQUES COM DIFERENTES PROBABILIDADES E INTENSIDADES:
    shock_type = np.random.choice([
        "regulatory_negative",    # 20% - Regulamentação restritiva
        "breakthrough_positive",  # 20% - Breakthrough tecnológico  
        "competitive_frenzy",     # 20% - FOMO competitivo
        "backlash_crisis",       # 15% - Backlash/resistência massiva
        "funding_crash",         # 10% - Corte orçamentário
        "viral_adoption",        # 10% - Adoção viral súbita
        "talent_shortage",       # 5%  - Falta de especialistas
    ], p=[0.20, 0.20, 0.20, 0.15, 0.10, 0.10, 0.05])
    
    # INTENSIDADES EXTREMAS (baseadas em observações 2023-2024)
    if shock_type in ["regulatory_negative", "backlash_crisis", "funding_crash"]:
        # CHOQUES NEGATIVOS SEVEROS
        shock_intensity = np.random.normal(-0.45, 0.25)  # Média -45%, std 25%
        shock_intensity = max(-0.80, shock_intensity)    # Limita a -80% (quase paralisa)
        
    elif shock_type in ["breakthrough_positive", "viral_adoption"]:
        # CHOQUES POSITIVOS EXPONENCIAIS  
        shock_intensity = np.random.normal(0.60, 0.35)   # Média +60%, std 35%
        shock_intensity = min(1.50, shock_intensity)     # Limita a +150% (3x aceleração)
        
    elif shock_type == "competitive_frenzy":
        # FOMO ORGANIZACIONAL (positivo mas volátil)
        shock_intensity = np.random.normal(0.40, 0.30)   # Média +40%, std 30%
        shock_intensity = min(1.00, shock_intensity)     # Limita a +100%
        
    else:  # talent_shortage
        # GARGALO DE TALENTO (neutro com muita incerteza)
        shock_intensity = np.random.normal(0.0, 0.40)    # Média 0%, std 40%
    
    # APLICA CHOQUE: modifica TODAS as probabilidades de progressão
    shocked_matrix = np.array(modified_matrix, dtype=float)
    shock_factor = 1.0 + shock_intensity
    
    for i in range(len(shocked_matrix) - 1):  # Não modifica estado absorvente
        for j in range(i + 1, len(shocked_matrix[i])):  # Apenas progressões
            if shocked_matrix[i][j] > 0:
                # Aplica choque com intensidade extrema
                shocked_matrix[i][j] *= shock_factor
                # Garante limites físicos (mais permissivos)
                shocked_matrix[i][j] = max(0.005, min(0.98, shocked_matrix[i][j]))
        
        # Renormaliza linha
        row_sum = np.sum(shocked_matrix[i])
        if row_sum > 0:
            shocked_matrix[i] = shocked_matrix[i] / row_sum
    
    # Log do choque (para debugging/análise)
    # print(f"🚨 CHOQUE Mês {month}: {shock_type} ({shock_intensity:+.1%})")
    
    return shocked_matrix

def simulate_individual_transitions(n_gerentes, state_vector, modified_transition_matrix):
    """
    Simula transições estocásticas individuais para cada gerente.
    
    Args:
        n_gerentes: Número total de gerentes
        state_vector: Distribuição atual de estados
        modified_transition_matrix: Matriz de transição modificada
    
    Returns:
        np.array: Nova distribuição de estados após transições estocásticas
    """
    # Converte distribuição em contagens de gerentes por estado
    state_counts = np.round(state_vector * n_gerentes).astype(int)
    
    # Ajusta para garantir que soma = n_gerentes
    diff = n_gerentes - np.sum(state_counts)
    if diff != 0:
        state_counts[0] += diff  # Ajuste no primeiro estado
    
    new_state_counts = np.zeros(len(state_vector), dtype=int)
    
    # Simula transição para cada gerente individualmente
    for current_state in range(len(state_counts)):
        n_managers_in_state = state_counts[current_state]
        
        if n_managers_in_state > 0:
            # Probabilidades de transição do estado atual
            transition_probs = modified_transition_matrix[current_state]
            
            # Simula transições estocásticas
            transitions = np.random.multinomial(
                n_managers_in_state, 
                transition_probs
            )
            
            # Adiciona às novas contagens
            new_state_counts += transitions
    
    # Converte de volta para distribuição
    return new_state_counts / n_gerentes


def observe_monthly_evidence(prev_state_vector, current_state_vector, month):
    """
    Simula observação de evidências mensais baseadas na evolução dos estados.
    
    Args:
        prev_state_vector: Distribuição de estados no mês anterior
        current_state_vector: Distribuição de estados no mês atual
        month: Mês atual (para modular força da evidência)
    
    Returns:
        dict: Evidências observadas para atualização bayesiana
    """
    # Calcula mudanças na distribuição
    state_changes = current_state_vector - prev_state_vector
    
    # Força da evidência cresce com o tempo
    evidence_strength = min(1.0, month / 12.0)
    base_observations = int(1000 * evidence_strength)
    
    # Evidência para AI_Investment (baseada em avanços nos estados)
    advanced_gains = np.sum(state_changes[3:])  # Estados S3 e S4
    ai_successes = max(0, int(base_observations * advanced_gains * 10))
    ai_failures = max(0, base_observations - ai_successes)
    
    # Evidência para Change_Adoption (baseada em mudanças positivas)
    positive_changes = np.sum(np.maximum(state_changes[1:], 0))
    change_successes = max(0, int(base_observations * positive_changes * 5))
    change_failures = max(0, base_observations - change_successes)
    
    # Evidência para Training_Quality (baseada no estado final S4)
    training_progress = state_changes[4]
    training_successes = max(0, int(base_observations * training_progress * 15))
    training_failures = max(0, base_observations - training_successes)
    
    return {
        "AI_Investment": {"successes": ai_successes, "failures": ai_failures},
        "Change_Adoption": {"successes": change_successes, "failures": change_failures},
        "Training_Quality": {"successes": training_successes, "failures": training_failures}
    }


def update_posterior_params(current_params, evidence):
    """
    Atualiza parâmetros bayesianos baseado em evidências observadas.
    Implementa atualização conjugada Beta-Binomial.
    
    Args:
        current_params: Parâmetros bayesianos atuais
        evidence: Evidências observadas
    
    Returns:
        dict: Parâmetros bayesianos atualizados
    """
    updated_params = {}
    
    for param_name in current_params:
        if param_name in evidence:
            # Atualização bayesiana: Beta(α,β) + evidência → Beta(α+s, β+f)
            current_alpha = current_params[param_name]["alpha"]
            current_beta = current_params[param_name]["beta"]
            
            successes = evidence[param_name]["successes"]
            failures = evidence[param_name]["failures"]
            
            updated_params[param_name] = {
                "alpha": current_alpha + successes,
                "beta": current_beta + failures
            }
        else:
            # Mantém parâmetro inalterado se não há evidência
            updated_params[param_name] = current_params[param_name].copy()
    
    return updated_params

def run_stochastic_simulation(n_gerentes=27000, n_months=36, transition_matrix=None, learning_enabled=True):
    """
    Executa UMA simulação estocástica completa com:
    1. Amostragem de parâmetros bayesianos
    2. Modificação da matriz de transição
    3. Simulação de transições individuais estocásticas
    
    Args:
        n_gerentes: Número de gerentes
        n_months: Horizonte temporal
        transition_matrix: Matriz base de transição
        learning_enabled: Aprendizado temporal ativo
    
    Returns:
        dict: Resultados de uma simulação estocástica
    """
    if transition_matrix is None:
        transition_matrix = [
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.85, 0.15, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    
    # Inicialização
    current_params = copy.deepcopy(parameters)
    state_vector = np.zeros(len(states))
    state_vector[0] = 1.0  # Todos começam em S0
    
    monthly_capacities = []
    params_evolution = []
    evidences_log = []
    state_history = [state_vector.copy()]  # Para rastrear histórico
    
    # Simulação mês a mês
    for month in range(n_months):
        # 1. Amostra parâmetros bayesianos
        sampled_params = {
            k: beta.rvs(p["alpha"], p["beta"])
            for k, p in current_params.items()
        }
        
        # 2. Modifica matriz de transição baseada nos parâmetros
        modified_matrix = apply_bayesian_factors_to_transitions(
            transition_matrix, sampled_params
        )
        
        # 2.5. NOVA FUNCIONALIDADE: Aplica choques de mercado aleatórios
        # VERSÃO 3.1: CHOQUES MUITO MAIS FREQUENTES E INTENSOS
        # Base teórica: Black Swan + Punctuated Equilibrium + IA volatility
        modified_matrix = add_market_shocks(month, modified_matrix, shock_probability=0.25)
        
        # 3. Registra evolução dos parâmetros
        params_evolution.append({
            param: {
                "alpha": current_params[param]["alpha"],
                "beta": current_params[param]["beta"],
                "mean": current_params[param]["alpha"] / (current_params[param]["alpha"] + current_params[param]["beta"]),
                "sampled_value": sampled_params[param]
            }
            for param in current_params
        })
        
        # 4. Simula transições estocásticas
        if month > 0:
            prev_state_vector = state_vector.copy()
            state_vector = simulate_individual_transitions(
                n_gerentes, state_vector, modified_matrix
            )
            state_history.append(state_vector.copy())
        
        # 5. Calcula capacidade do mês
        monthly_capacity = np.sum([
            states[i]["multiplicador"] * state_vector[i]
            for i in range(len(states))
        ]) * 2000
        
        monthly_capacities.append(monthly_capacity)
        
        # 6. Atualização bayesiana (se habilitada)
        if learning_enabled and month > 0:
            evidence = observe_monthly_evidence(
                prev_state_vector, state_vector, month
            )
            evidences_log.append(evidence)
            current_params = update_posterior_params(current_params, evidence)
    
    # Resultados finais
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
        "state_distribution": state_vector,
        "params_evolution": params_evolution,
        "evidences_log": evidences_log,
        "learning_enabled": learning_enabled
    }

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

def run_monte_carlo_analysis(n_gerentes=27000, n_months=36, transition_matrix=None, learning_enabled=True, n_simulations=1000):
    """
    VERSÃO 3.1: ANÁLISE MONTE CARLO COM VOLATILIDADE EXTREMA
    
    Executa múltiplas simulações independentes com MÁXIMA DIVERSIDADE organizacional.
    
    🚨 NOVA ABORDAGEM: ORGANIZATIONAL HETEROGENEITY + REGIME SWITCHING
    
    Cada simulação representa uma organização única em contexto único:
    1. ✅ DNA organizacional diferenciado (5 dimensões)
    2. ✅ Regimes de mercado voláteis (conservative/normal/aggressive)  
    3. ✅ Matrix customization por organização
    4. ✅ Fat-tail distributions nos resultados
    5. ✅ Extreme percentiles tracking (P1, P99)
    
    📚 JUSTIFICATIVA CIENTÍFICA:
    
    1. ORGANIZATIONAL HETEROGENEITY (Nelson & Winter, 1982):
       - Firmas são fundamentalmente diferentes
       - Technology adoption capabilities variam drasticamente
       - Path dependence cria trajetórias divergentes
    
    2. REGIME SWITCHING MODELS (Hamilton, 1989):
       - Markets operam em regimes distintos
       - Structural breaks são comuns em disruption
       - IA intensifica regime volatility
    
    3. FAT TAIL DISTRIBUTIONS (Mandelbrot, 1963):
       - Innovation outcomes seguem power laws
       - Extreme events são mais frequentes que Gaussian predicts
       - Heavy tail = natural em technology adoption
    
    Args:
        n_gerentes: Número de gerentes
        n_months: Horizonte temporal
        transition_matrix: Matriz de transição
        learning_enabled: Aprendizado temporal ativo
        n_simulations: Número de simulações Monte Carlo
    
    Returns:
        dict: Análise probabilística com fat tails e regime tracking
    """
    all_results = []
    final_capacities = []
    monthly_trajectories = []
    regime_trajectories = []  # NOVO: tracking de regimes
    org_dna_log = []  # NOVO: tracking de DNA organizacional
    
    # ===== REGIME SWITCHING SETUP =====
    # 3 REGIMES com características econômicas distintas
    regimes = {
        0: {"name": "conservative", "shock_multiplier": 0.6, "adoption_bias": -0.10},
        1: {"name": "normal", "shock_multiplier": 1.0, "adoption_bias": 0.0},  
        2: {"name": "aggressive", "shock_multiplier": 1.7, "adoption_bias": +0.15}
    }
    
    # Executa múltiplas simulações com MÁXIMA DIVERSIDADE
    for sim in range(n_simulations):
        
        # ===== REGIME SAMPLING =====
        # Mercado pode estar em qualquer regime (instabilidade estrutural)
        current_regime = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
        
        # ===== ORGANIZATIONAL DNA SAMPLING =====
        # Cada organização tem perfil comportamental único
        org_dna = {
            "risk_culture": np.random.beta(1.0, 2.5),       # Maioria risk-averse
            "tech_readiness": np.random.beta(1.5, 1.5),     # Bimodal distribution
            "resource_capacity": np.random.beta(1.2, 1.8),  # Few resource-rich
            "leadership_vision": np.random.beta(2.0, 1.0),  # Some visionary leaders  
            "regulatory_pressure": np.random.beta(1.8, 1.2), # Sector-dependent
            "network_position": np.random.beta(1.3, 1.7)    # Network centrality
        }
        
        # ===== MATRIX CUSTOMIZATION BY ORGANIZATION =====
        customized_matrix = None
        if transition_matrix is not None:
            customized_matrix = np.array(transition_matrix, dtype=float)
            
            # DNA IMPACT: Cada dimensão afeta matriz diferentemente
            dna_impact = (
                org_dna["risk_culture"] * 0.20 +           # Risk → slower adoption
                org_dna["tech_readiness"] * 0.25 +         # Tech → faster adoption
                org_dna["resource_capacity"] * 0.20 +      # Resources → capability
                org_dna["leadership_vision"] * 0.20 +      # Vision → strategic push
                org_dna["network_position"] * 0.15         # Network → learning speed
            )
            
            # REGIME BIAS: Adiciona bias estrutural baseado no regime
            regime_bias = regimes[current_regime]["adoption_bias"]
            total_modifier = dna_impact + regime_bias
            
            # Aplica modificação heterogênea na matriz
            for i in range(len(customized_matrix)):
                for j in range(len(customized_matrix[i])):
                    if i != j and customized_matrix[i][j] > 0:
                        # Variação organizacional + regime bias
                        org_variation = np.random.normal(total_modifier, 0.25)
                        customized_matrix[i][j] *= np.clip(org_variation, 0.2, 3.0)
                
                # Renormaliza linha
                row_sum = np.sum(customized_matrix[i])
                if row_sum > 0:
                    customized_matrix[i] = customized_matrix[i] / row_sum
        
        # ===== STOCHASTIC SIMULATION =====
        result = run_stochastic_simulation(
            n_gerentes=n_gerentes,
            n_months=n_months,
            transition_matrix=customized_matrix,
            learning_enabled=learning_enabled
        )
        
        # ===== REGIME-SPECIFIC POST-PROCESSING =====
        # Aplica multiplicador de regime (structural breaks)s
        regime_modifier = regimes[current_regime]["shock_multiplier"]
        modified_trajectory = result["df_monthly"]["Contas por Gerente (média)"].values * regime_modifier
        
        # REGIME NOISE: Adiciona ruído característico do regime
        if current_regime == 0:  # Conservative: baixa volatilidade, downward bias
            regime_noise = np.random.normal(-0.05, 0.08, len(modified_trajectory))
        elif current_regime == 2:  # Aggressive: alta volatilidade, upward bias  
            regime_noise = np.random.normal(0.10, 0.30, len(modified_trajectory))
        else:  # Normal: volatilidade moderate
            regime_noise = np.random.normal(0.0, 0.15, len(modified_trajectory))
        
        modified_trajectory = modified_trajectory * (1 + regime_noise)
        modified_trajectory = np.clip(modified_trajectory, 0, 15000)  # Limites físicos
        
        # ===== LOGGING =====
        all_results.append(result)
        final_capacities.append(modified_trajectory[-1])
        monthly_trajectories.append(modified_trajectory)
        regime_trajectories.append(current_regime)
        org_dna_log.append(org_dna)
    
    # ===== ANÁLISE ESTATÍSTICA COM FAT TAILS =====
    monthly_trajectories = np.array(monthly_trajectories)
    
    # PERCENTIS EXTREMOS para capturar tail risks
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    monthly_percentiles = {}
    
    for p in percentiles:
        monthly_percentiles[f"p{p}"] = np.percentile(monthly_trajectories, p, axis=0)
    
    # ===== FINAL DISTRIBUTION WITH TAIL ANALYSIS =====
    final_capacities = np.array(final_capacities)
    final_stats = {
        "mean": np.mean(final_capacities),
        "std": np.std(final_capacities),
        "min": np.min(final_capacities),
        "max": np.max(final_capacities),
        "p1": np.percentile(final_capacities, 1),     # LEFT TAIL
        "p5": np.percentile(final_capacities, 5),
        "p10": np.percentile(final_capacities, 10),
        "p25": np.percentile(final_capacities, 25),
        "p50": np.percentile(final_capacities, 50),   # MEDIAN
        "p75": np.percentile(final_capacities, 75),
        "p90": np.percentile(final_capacities, 90),
        "p95": np.percentile(final_capacities, 95),
        "p99": np.percentile(final_capacities, 99),   # RIGHT TAIL
        "iqr": np.percentile(final_capacities, 75) - np.percentile(final_capacities, 25),
        "tail_ratio": (np.percentile(final_capacities, 95) - np.percentile(final_capacities, 5)) / np.mean(final_capacities)
    }
    
    # ===== REGIME & DNA ANALYSIS =====
    regime_analysis = {
        "regime_distribution": {
            "conservative": np.sum(np.array(regime_trajectories) == 0) / len(regime_trajectories),
            "normal": np.sum(np.array(regime_trajectories) == 1) / len(regime_trajectories),
            "aggressive": np.sum(np.array(regime_trajectories) == 2) / len(regime_trajectories)
        },
        "avg_dna_profile": {
            key: np.mean([org[key] for org in org_dna_log]) 
            for key in org_dna_log[0].keys()
        }
    }
    
    return {
        "monthly_percentiles": monthly_percentiles,
        "final_stats": final_stats,
        "all_trajectories": monthly_trajectories,
        "final_capacities": final_capacities,
        "regime_analysis": regime_analysis,
        "n_simulations": n_simulations,
        "volatility_metrics": {
            "coefficient_of_variation": final_stats["std"] / final_stats["mean"],
            "tail_ratio": final_stats["tail_ratio"],
            "extreme_range": final_stats["max"] - final_stats["min"],
            "tail_thickness": (final_stats["p99"] - final_stats["p1"]) / (final_stats["p75"] - final_stats["p25"])
        }
    }

def calculate_scenario_probabilities(monte_carlo_results, target_scenarios):
    """
    Calcula probabilidades de cenários específicos se concretizarem.
    
    Args:
        monte_carlo_results: Resultados da análise Monte Carlo
        target_scenarios: Lista de cenários alvo (ex: [2200, 2500, 3000])
    
    Returns:
        dict: Probabilidades de cada cenário
    """
    final_capacities = monte_carlo_results["final_capacities"]
    probabilities = {}
    
    for target in target_scenarios:
        # Probabilidade de exceder o target
        prob_exceed = np.mean(final_capacities >= target)
        
        # Probabilidade de ficar dentro de ±5% do target
        margin = target * 0.05
        prob_within_5pct = np.mean(
            (final_capacities >= target - margin) & 
            (final_capacities <= target + margin)
        )
        
        probabilities[f"P(>= {target})"] = prob_exceed
        probabilities[f"P(±5% de {target})"] = prob_within_5pct
    
    return probabilities

def analyze_risk_metrics(monte_carlo_results, baseline=2000):
    """
    Calcula métricas de risco para as projeções.
    
    Args:
        monte_carlo_results: Resultados Monte Carlo
        baseline: Capacidade baseline (sem IA)
    
    Returns:
        dict: Métricas de risco e incerteza
    """
    final_capacities = monte_carlo_results["final_capacities"]
    
    # Value at Risk (VaR) - Pior cenário em 95% dos casos
    var_95 = np.percentile(final_capacities, 5)
    var_90 = np.percentile(final_capacities, 10)
    
    # Expected Shortfall - Média dos 5% piores casos
    worst_5_pct = final_capacities[final_capacities <= var_95]
    expected_shortfall = np.mean(worst_5_pct) if len(worst_5_pct) > 0 else var_95
    
    # Probabilidade de não ter ganho
    prob_no_gain = np.mean(final_capacities <= baseline)
    
    # Coeficiente de variação
    mean_capacity = np.mean(final_capacities)
    std_capacity = np.std(final_capacities)
    cv = std_capacity / mean_capacity
    
    return {
        "var_95": var_95,
        "var_90": var_90,
        "expected_shortfall": expected_shortfall,
        "prob_no_gain": prob_no_gain,
        "coefficient_variation": cv,
        "mean": mean_capacity,
        "std": std_capacity,
        "baseline": baseline
    }
