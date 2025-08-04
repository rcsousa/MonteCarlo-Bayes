parameters = {
    "AI_Investment": {
        "alpha": 5,
        "beta": 3,
        "nota": (
            "Reflete intensidade de investimento em IA em organizações financeiras. "
            "Benchmark: McKinsey (2024) aponta que empresas com alto investimento em IA têm "
            "até 50% mais chance de gerar valor significativo. A distribuição Beta(5,3) traduz "
            "um viés otimista, mas conservador, com valor esperado ~0.625 e incerteza moderada.\n"
            "📚 Fonte: McKinsey Global Survey on AI 2024, 'The State of AI in Financial Services'."
        )
    },
    "Change_Adoption": {
        "alpha": 4,
        "beta": 4,
        "nota": (
            "Reflete prontidão organizacional para absorver mudança induzida por IA. "
            "Segundo PwC (2023), apenas 30–35% das empresas relatam alta maturidade em gestão de mudança "
            "para tecnologia disruptiva. A Beta(4,4) assume neutralidade (valor esperado ~0.5) "
            "e reflete alta variabilidade no mercado.\n"
            "📚 Fonte: PwC AI Readiness Index 2023."
        )
    },
    "Training_Quality": {
        "alpha": 3,
        "beta": 2,
        "nota": (
            "Representa a qualidade percebida dos programas de capacitação para IA. "
            "Em estudos do BCG (2025), empresas líderes treinam seus times em IA com estruturas robustas "
            "e engajamento contínuo. A Beta(3,2) sugere boa expectativa de qualidade com leve otimismo.\n"
            "📚 Fonte: BCG Report 2025, 'Scaling Generative AI in the Enterprise'."
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
