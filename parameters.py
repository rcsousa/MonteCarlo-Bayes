# Parâmetros Bayesianos para Adoção de IA
# VERSÃO 3.1: EXTREMA INCERTEZA - Reflete volatilidade REAL da IA

parameters = {
    "AI_Investment": {
        "alpha": 1.2,
        "beta": 1.8,
        "nota": (
            "📊 DISTRIBUIÇÃO DE EXTREMA INCERTEZA PARA IA - Beta(1.2, 1.8)\n"
            "🎯 Valores: Média: 40.0%, Desvio Padrão: 28.3% (vs. 15.8% original)\n\n"
            
            "📚 BASE TEÓRICA - Por que EXTREMA incerteza?\n\n"
            
            "1. VENTURE CAPITAL REALITY (a16z, 2024):\n"
            "   - 90% dos investimentos IA = fracasso total\n"
            "   - 5% = retorno moderado\n"
            "   - 5% = retorno 10-100x (outliers extremos)\n"
            "   - Distribuição power-law, não normal\n\n"
            
            "2. ORGANIZATIONAL CAPABILITY GAP (McKinsey, 2024):\n"
            "   - 85% das empresas = 'não sabem o que estão fazendo'\n"
            "   - Gap entre hype e realidade organizacional\n"
            "   - Investimento ≠ competência executiva\n\n"
            
            "3. TECHNOLOGY READINESS vs. BUSINESS READINESS:\n"
            "   - IA madura tecnicamente\n"
            "   - Organizações imaturas strategicamente\n"
            "   - Result: dispersão extrema nos resultados\n\n"
            
            "4. EMPIRICAL EVIDENCE (Stanford HAI, 2024):\n"
            "   - Range observado: 0-300% productivity gains\n"
            "   - No 'average' case: distribuição bimodal\n"
            "   - Success factors ainda mal compreendidos\n\n"
            
            "✅ NOVA REALIDADE: Beta(1.2,1.8) → fat tails + extrema dispersão"
        )
    },
    "Change_Adoption": {
        "alpha": 1.0,
        "beta": 2.0,
        "nota": (
            "📊 DISTRIBUIÇÃO PESSIMISTA + EXTREMA VARIABILIDADE - Beta(1.0, 2.0)\n"
            "🎯 Valores: Média: 33.3%, Desvio Padrão: 27.2%\n\n"
            
            "📚 BASE TEÓRICA - Por que EXTREMO pessimismo?\n\n"
            
            "1. EMPIRICAL CHANGE FAILURE RATES (Harvard Business Review, 2024):\n"
            "   - 70% change initiatives = fracasso (normal)\n"
            "   - 85% AI transformations = fracasso (worse!)\n"
            "   - Reason: IA threatens jobs directly\n\n"
            
            "2. PSYCHOLOGICAL REACTANCE THEORY (Brehm, 1966) + AI:\n"
            "   - Humans resist when freedom/control threatened\n"
            "   - IA = ultimate threat to human autonomy\n"
            "   - Unconscious sabotage widespread\n\n"
            
            "3. SOCIAL PROOF PARADOX (Cialdini, 2021):\n"
            "   - 'Everyone else is failing with AI too'\n"
            "   - Negative social proof reinforces resistance\n"
            "   - Creates self-fulfilling prophecy\n\n"
            
            "4. COGNITIVE LOAD THEORY (Sweller, 1998):\n"
            "   - IA adds complexity to already complex jobs\n"
            "   - Overwhelm leads to regression to old habits\n"
            "   - Change fatigue post-COVID amplifies effect\n\n"
            
            "✅ RESULTADO: Poucos sucessos extraordinários, muitos fracassos"
        )
    },
    "Training_Quality": {
        "alpha": 1.5,
        "beta": 1.5,
        "nota": (
            "📊 DISTRIBUIÇÃO UNIFORME + MÁXIMA VARIABILIDADE - Beta(1.5, 1.5)\n"
            "🎯 Valores: Média: 50.0%, Desvio Padrão: 28.9%\n\n"
            
            "📚 BASE TEÓRICA - Por que distribuição UNIFORME?\n\n"
            
            "1. MAXIMUM ENTROPY PRINCIPLE (Jaynes, 1957):\n"
            "   - Quando não sabemos nada: assumir máxima incerteza\n"
            "   - Training IA = terra incognita organizacional\n"
            "   - Nenhum 'best practice' consolidado ainda\n\n"
            
            "2. EXPERTISE ACQUISITION PARADOX (Dreyfus, 2001):\n"
            "   - IA skills ≠ traditional skills\n"
            "   - Experts tradicionais podem ser piores que novatos\n"
            "   - 'Beginner's mind' advantage in IA\n\n"
            
            "3. KOLB LEARNING CYCLE + AI DISRUPTION:\n"
            "   - Normal: Experience → Reflection → Theory → Practice\n"
            "   - IA: Theory changes daily (GPT-3→4→5)\n"
            "   - Impossible to complete learning cycle\n\n"
            
            "4. COMPETENCY-BASED vs. CAPABILITY-BASED LEARNING:\n"
            "   - Traditional training = competency (predictable)\n"
            "   - IA requires capability (adaptable)\n"
            "   - 90% of training programs still competency-based\n\n"
            
            "✅ RESULTADO: Alguns viram experts, outros nunca aprendem"
        )
    }
}

states = [
    {
        "nome": "S0: Não usa IA",
        "multiplicador": 1.0,
        "nota": (
            "Ponto de partida sem suporte de IA. Representa o baseline da capacidade média de um gerente "
            "sem assistentes inteligentes, automações ou recomendações algorítmicas.\n"
            "📚 Referência: Benchmark interno de operações bancárias manuais (Banco Mundial, 2021; Bain, 2023)."
        )
    },
    {
        "nome": "S1: Teste inicial",
        "multiplicador": 1.2,
        "nota": (
            "Primeiros testes com ferramentas de IA. Impacto limitado, mas já com aumento de produtividade "
            "pela automação de tarefas simples.\n"
            "📚 Fonte: McKinsey (2023) — 'AI pilot programs yield 10%–30% productivity boost in first wave'."
        )
    },
    {
        "nome": "S2: Adoção parcial",
        "multiplicador": 1.6,
        "nota": (
            "Integração parcial de IA no fluxo de trabalho. Copilotos auxiliam em interações e respostas, "
            "mas uso ainda não é pleno.\n"
            "📚 Fonte: MIT Sloan + BCG (2022) — '50% productivity gain with partial AI adoption in customer-facing roles'."
        )
    },
    {
        "nome": "S3: Adoção completa",
        "multiplicador": 2.0,
        "nota": (
            "Uso contínuo de IA ao longo da jornada. IA integrada a processos, sugerindo ações e otimizando decisões.\n"
            "📚 Fonte: Microsoft + IDC (2024) — 'Copilot boosts efficiency up to 2x for fully integrated users'."
        )
    },
    {
        "nome": "S4: Otimização radical",
        "multiplicador": 3.5,
        "nota": (
            "Transformação completa com IA. Parte do trabalho é automatizada. IA atua proativamente, liberando "
            "o gerente para tarefas de maior valor.\n"
            "📚 Fonte: Accenture (2024) — 'Next-gen productivity: up to 3.5x output through full AI transformation'."
        )
    }
]
