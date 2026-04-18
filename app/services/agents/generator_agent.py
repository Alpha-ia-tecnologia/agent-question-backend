"""
Agente Gerador de Questões.

Responsável por criar questões educacionais baseadas em
habilidades e níveis de proficiência, seguindo padrões SAEB/BNCC.
"""

import logging
import json
import os
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from app.services.agents.state import AgentState
from app.enums.agente_prompt_template import AgentPromptTemplates, get_prompt
from app.core.llm_config import get_question_llm, get_runnable_config
from app.schemas.question_schema import QuestionListSchema
from app.services.progress_manager import get_current_progress

logger = logging.getLogger(__name__)


def _parse_json_response(response_text: str) -> dict:
    """
    Faz parsing manual de JSON da resposta do LLM.
    
    Args:
        response_text: Texto da resposta do LLM
        
    Returns:
        Dicionário com dados parseados
    """
    text = response_text.strip()
    
    # Remove blocos de código markdown
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    # Encontra o início do JSON
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("Nenhum objeto JSON encontrado na resposta")
    
    # Extrai o primeiro objeto JSON completo
    brace_count = 0
    end_idx = start_idx
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    json_str = text[start_idx:end_idx]
    return json.loads(json_str)


def _select_template(query: Any, has_feedback: bool) -> str:
    """
    Seleciona o template apropriado baseado no contexto.
    
    Args:
        query: Parâmetros da requisição
        has_feedback: Se há feedback de revisão anterior
        
    Returns:
        Template string para o prompt
    """
    if has_feedback:
        # Template com instruções de correção baseadas no feedback
        return get_prompt(AgentPromptTemplates.SOURCE_PT_TEMPLATE)
    
    # Lógica de seleção de template
    # use_real_text usa o mesmo template - textos são injetados dinamicamente
    if query.authentic:
        return get_prompt(AgentPromptTemplates.AUTHENTIC_PT_TEMPLATE)
    else:
        return get_prompt(AgentPromptTemplates.SOURCE_PT_TEMPLATE)


# ── Mapeamento componente → arquivo de referência ──
_COMPONENT_MAP = {
    "math": {
        "file": "app/prompts/math_skills_reference.txt",
        "key": "math",
        "keywords": [
            "matemática", "matematica", "math",
            "álgebra", "geometria", "aritmética", "estatística",
            "probabilidade", "grandezas", "medidas", "números",
        ],
    },
    "portuguese": {
        "file": "app/prompts/portuguese_skills_reference.txt",
        "key": "portuguese",
        "keywords": [
            "língua portuguesa", "lingua portuguesa", "português", "portugues",
            "leitura", "escrita", "gramática", "interpretação de texto",
            "gênero textual", "ortografia", "produção textual",
        ],
    },
    "science": {
        "file": "app/prompts/science_skills_reference.txt",
        "key": "science",
        "keywords": [
            "ciências", "ciencias", "ciências da natureza", "biologia",
            "física", "fisica", "química", "quimica", "natureza",
        ],
    },
    "humanities": {
        "file": "app/prompts/humanities_skills_reference.txt",
        "key": "humanities",
        "keywords": [
            "ciências humanas", "história", "historia", "geografia",
            "sociologia", "filosofia", "humanities", "humanas",
        ],
    },
}


def _load_skills_reference_for(query) -> dict[str, str]:
    """
    Carrega APENAS a referência de habilidades do componente curricular correto,
    ou múltiplos se a consulta for interdisciplinar.
    """
    result = {}
    
    def process_component(component: str, skill: str):
        search_text = f"{component} {skill}".lower().strip()
        detected = None
        best_score = 0
        
        for comp_id, info in _COMPONENT_MAP.items():
            score = sum(1 for kw in info["keywords"] if kw in search_text)
            if score > best_score:
                best_score = score
                detected = comp_id
        
        if not detected and component:
            component_lower = component.lower().strip()
            for comp_id, info in _COMPONENT_MAP.items():
                if comp_id in component_lower or info["key"] in component_lower:
                    detected = comp_id
                    break
        
        if detected:
            info = _COMPONENT_MAP[detected]
            if info["key"] not in result:
                ref_path = os.path.abspath(info["file"])
                try:
                    with open(ref_path, "r", encoding="utf-8") as f:
                        result[info["key"]] = f.read()
                    logger.info(f"⚡ Carregada referência: {detected} ({os.path.getsize(ref_path) // 1024}KB)")
                except FileNotFoundError:
                    logger.warning(f"⚠️ {info['file']} não encontrado")
                    
    # Processa as habilidades combinadas se for interdisciplinar
    if getattr(query, "is_interdisciplinary", False) and hasattr(query, "combined_skills"):
        for item in query.combined_skills:
            process_component(item.curriculum_component, item.skill)
    else:
        # Modo padrão
        component = getattr(query, "curriculum_component", "")
        skill = getattr(query, "skill", "")
        process_component(component, skill)
        
    if not result:
        # Fallback: componente não detectado → carrega todos
        logger.warning("⚠️ Componente não detectado — carregando todas as referências (fallback)")
        for comp_id, info in _COMPONENT_MAP.items():
            ref_path = os.path.abspath(info["file"])
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    result[info["key"]] = f.read()
            except FileNotFoundError:
                pass
                
    return result


def generator_node(state: AgentState) -> AgentState:
    """
    Nó do Agente Gerador.
    
    Gera questões educacionais usando o LLM configurado (DeepSeek/OpenAI/Gemini).
    Se houver feedback de uma revisão anterior, incorpora as correções.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com questões geradas
    """
    query = state["query"]
    feedback = state.get("revision_feedback")
    retry_count = state.get("retry_count", 0)
    
    logger.info(
        f"🔵 Agente Gerador - Tentativa {retry_count + 1} | "
        f"Habilidade: {query.skill[:40]}..."
    )
    
    progress = get_current_progress()
    
    try:
        # Obtém LLM e template (permite override por requisição via query.llm_model)
        llm_model_override = getattr(query, "llm_model", None)
        if progress:
            progress.log(
                "generator",
                f"Iniciando LLM: {llm_model_override or 'padrão'}",
                "",
                "🔌",
            )
        llm = get_question_llm(model=llm_model_override)
        template_str = _select_template(query, feedback is not None)
        image_dep = query.image_dependency
        if progress:
            progress.log("generator", f"📐 Habilidade: {query.skill[:60]}", "", "🎯")
            progress.log("generator", f"Série: {query.grade} · Proficiência: {query.proficiency_level}", "", "🏫")
            dep_label = {"none": "Sem imagem", "optional": "Imagem opcional", "required": "Imagem obrigatória"}
            progress.log("generator", f"Regras de imagem: {dep_label.get(image_dep, image_dep)}", "", "🖼️")
            progress.log("generator", "Carregando metodologia de distratores (7 tipos de erro)", "", "🧠")
            tpl = "com feedback" if feedback else "padrão"
            progress.log("generator", f"Template selecionado: {tpl}", "", "📄")
        
        # Mapeia instruções de imagem (REGRAS CRÍTICAS de coerência)
        image_instructions = {
            "none": """⚠️ QUESTÃO SEM IMAGEM - REGRAS ABSOLUTAS ⚠️

❌ PROIBIDO:
   - NÃO use "[IMAGEM: ...]" no texto
   - NÃO mencione "observe a figura", "observe a imagem", "observe a tirinha"
   - NÃO faça questões sobre gráficos visuais, charges ou tirinhas
   - NÃO referencie elementos visuais

✅ OBRIGATÓRIO:
   - Questões 100% resolvidas apenas com LEITURA DO TEXTO
   - TODO conteúdo necessário deve estar ESCRITO no texto-base
   - O campo "text" deve conter o texto completo para resolver a questão

🎯 O aluno resolve APENAS LENDO, sem precisar de nenhuma imagem.""",
            "optional": "As questões podem ter imagens ilustrativas decorativas opcionais, mas a resolução NÃO deve depender da imagem.",
            "required": """⚠️⚠️⚠️ QUESTÃO OBRIGATORIAMENTE DEPENDENTE DE IMAGEM ⚠️⚠️⚠️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 REGRA PRINCIPAL: A questão SÓ PODE ser resolvida OLHANDO para a imagem.
   Se o aluno conseguir responder APENAS lendo o texto, a questão está ERRADA!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ESTRUTURA OBRIGATÓRIA:
   1. O campo "text" deve ser CURTO: "Observe a imagem/gráfico/figura a seguir."
      - NÃO coloque dados, tabelas ou descrições no campo "text"
      - Os dados essenciais estarão NA IMAGEM (que será gerada depois)
   2. O "question_statement" deve EXIGIR análise visual:
      - "De acordo com o gráfico, qual foi..."
      - "Observando a figura, qual é a medida de..."
      - "Com base na imagem, é possível concluir que..."
      - "A partir dos dados apresentados no gráfico..."
   3. As alternativas devem requerer INTERPRETAÇÃO da imagem + raciocínio

🎯 TIPOS DE QUESTÃO COM IMAGEM (escolha um):
   📊 GRÁFICOS: barras, pizza, linha — o aluno precisa LER valores do gráfico
   📐 FIGURAS GEOMÉTRICAS: triângulos, retângulos — o aluno precisa EXTRAIR medidas da figura
   🗺️ MAPAS/DIAGRAMAS: o aluno precisa INTERPRETAR o diagrama visual
   📈 TABELAS VISUAIS: dados organizados que só existem na imagem
   🖼️ CENAS/TIRINHAS: o aluno precisa OBSERVAR elementos visuais

🚫🚫🚫 REGRA DE OURO: JAMAIS DESCREVA A IMAGEM NO TEXTO 🚫🚫🚫
   A imagem será gerada SEPARADAMENTE por outra IA.
   Você NÃO SABE como a imagem será. NÃO INVENTE descrições.

   ❌ ERRADO (descreve a imagem no texto ou enunciado):
      - "O gráfico mostra que 40% dos alunos preferem futebol"
      - "Na imagem, há um triângulo retângulo com catetos de 3cm e 4cm"
      - "A tirinha apresenta um personagem surpreso"
      - "[IMAGEM: gráfico de barras com dados de...]"

   ✅ CORRETO (apenas referencia sem descrever):
      - "Observe o gráfico a seguir." (SEM dizer o que o gráfico mostra)
      - "Observe a figura a seguir." (SEM descrever a figura)
      - "Analise a imagem e responda." (SEM descrever o conteúdo)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 REGRAS CRÍTICAS PARA GRÁFICOS E TABELAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ⚠️ GRÁFICO DE SETORES (PIZZA):
      - Gráficos de pizza mostram PROPORÇÕES, não valores absolutos
      - Se a pergunta pede QUANTIDADE (ex: "quantos livros?"), você DEVE:
        → Informar o TOTAL no texto: "No total, a turma leu 35 livros."
        → OU perguntar sobre PERCENTUAL/PROPORÇÃO: "Qual setor representa a maior parte?"
      - NUNCA pergunte um valor absoluto sem fornecer o total para cálculo
      - ❌ ERRADO: "Quantos livros em abril?" (sem total, pizza não mostra valores)
      - ✅ CORRETO: "Sabendo que no total foram 35 livros, quantos foram em abril?"
      - ✅ CORRETO: "Qual mês teve o maior percentual de livros lidos?"

   ⚠️ GRÁFICO DE BARRAS/COLUNAS:
      - DEVE ter eixo Y com escala numérica visível
      - Os valores devem ser LIDOS do gráfico (não informados no texto)
      - A pergunta pode pedir valores absolutos (a escala estará visível)

   ⚠️ GRÁFICO DE LINHAS:
      - DEVE ter eixos X e Y rotulados com valores
      - Pode pedir tendências, valores específicos ou comparações

   🔴🔴🔴 VERIFICAÇÃO OBRIGATÓRIA DE CONSISTÊNCIA 🔴🔴🔴
      ANTES de finalizar uma questão com gráfico, verifique:

      1. CONTAGEM: Se o texto/enunciado diz "quatro municípios",
         o image_data DEVE ter EXATAMENTE 4 itens (não 3, não 5).
         ❌ ERRADO: "quatro municípios" + gráfico com 3 setores
         ✅ CORRETO: "quatro municípios" + gráfico com 4 setores

      2. SOMA DE PERCENTUAIS: Todos os percentuais DEVEM somar 100%.
         ❌ ERRADO: 35% + 40% + 25% = 100% mas texto diz "4 itens"
         ✅ CORRETO: 35% + 25% + 20% + 20% = 100% com 4 itens

      3. CÁLCULO DA RESPOSTA: valor = (percentual × total) / 100
         Verifique que o resultado bate com a alternativa correta.

      4. NOMES CONSISTENTES: Os nomes no image_data devem ser
         EXATAMENTE iguais aos mencionados no enunciado/alternativas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 REGRAS CRÍTICAS PARA GEOMETRIA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ⚠️ IDENTIFICAÇÃO DOS LADOS (FUNDAMENTAL):
      - HIPOTENUSA = lado OPOSTO ao ângulo de 90°, SEMPRE o MAIOR lado
      - CATETOS = os dois lados que FORMAM o ângulo de 90°
      - Se a questão pede "o comprimento da rampa" → rampa = HIPOTENUSA
      - Se a questão pede "a altura/sustentação" → é CATETO VERTICAL
      - NUNCA confunda cateto com hipotenusa na explicação!

   ⚠️ MARCAÇÃO DO "?" NA FIGURA:
      - O "?" DEVE marcar EXATAMENTE o que a questão pede
      - Se pergunta "qual o comprimento da rampa?" → "?" na rampa (hipotenusa)
      - Se pergunta "qual a altura?" → "?" no segmento vertical (cateto)
      - NUNCA coloque "?" em um lado diferente do que a questão pede

   ⚠️ CONSISTÊNCIA VALORES ↔ RESPOSTA:
      - Se catetos = a e b, então hipotenusa = √(a² + b²)
      - Verifique que a resposta correta bate com o cálculo
      - A explicação deve identificar CORRETAMENTE qual é cateto e qual é hipotenusa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 REGRAS CRÍTICAS PARA GEOMETRIA ESPACIAL (3D):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   🔺 PIRÂMIDE DE BASE QUADRADA - TERMINOLOGIA OBRIGATÓRIA:
      - ARESTA LATERAL = segmento do vértice V a um VÉRTICE da base
        → Cálculo: √(h² + (d/2)²) onde d = diagonal da base = lado × √2
        → Exemplo: base 10cm, h=12cm → √(144 + 50) = √194 ≈ 13,93 cm
      - APÓTEMA DA PIRÂMIDE = segmento do vértice V ao PONTO MÉDIO de uma aresta da base
        → Cálculo: √(h² + a²) onde a = apótema da base = lado / 2
        → Exemplo: base 10cm, h=12cm → √(144 + 25) = √169 = 13 cm
      - APÓTEMA DA BASE = distância do centro ao ponto médio do lado = lado / 2
      - METADE DA DIAGONAL = distância do centro a um vértice = (lado × √2) / 2

   🔴🔴🔴 ERRO GRAVÍSSIMO A EVITAR 🔴🔴🔴
      ❌ ERRADO: Pedir "aresta lateral" e calcular usando apótema da base (lado/2)
      ❌ ERRADO: Pedir "apótema da pirâmide" e calcular usando metade da diagonal
      ✅ CORRETO: Se pede "aresta lateral" → usar √(h² + ((lado×√2)/2)²)
      ✅ CORRETO: Se pede "apótema da pirâmide" → usar √(h² + (lado/2)²)

   ⚠️ CHECKLIST OBRIGATÓRIO PARA PIRÂMIDES:
      1. Identifique qual medida a questão PEDE (aresta lateral OU apótema)
      2. Use a fórmula CORRETA para essa medida
      3. Verifique que a imagem marca "?" no segmento CORRETO
      4. Verifique que a resposta numérica bate com a fórmula
      5. Na explicação, nomeie CORRETAMENTE cada segmento

   🔵 CONE:
      - GERATRIZ = segmento do vértice à circunferência da base
        → Cálculo: √(h² + r²) onde r = raio da base
      - APÓTEMA = mesmo que geratriz (em cones)

   🟢 PRISMA:
      - ARESTA LATERAL = altura do prisma (perpendicular às bases)
      - DIAGONAL DA FACE = √(aresta_lateral² + aresta_base²)
      - DIAGONAL DO PRISMA = √(aresta_lateral² + diagonal_base²)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 CAMPO OBRIGATÓRIO: "image_data"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Você DEVE incluir o campo "image_data" em cada questão com TODOS os dados
   que a imagem precisa mostrar. Isso garante que a IA geradora de imagens
   saiba EXATAMENTE quais valores, rótulos e medidas incluir na imagem.

   Exemplos por tipo:

   📊 Para gráficos:
   "image_data": {{
       "tipo": "grafico_barras",
       "titulo": "Livros Lidos no Bimestre",
       "eixo_x": ["Março", "Abril", "Maio", "Junho"],
       "eixo_y": "Quantidade de livros",
       "valores": [5, 8, 10, 12],
       "destaque": "Abril"
   }}

   📐 Para geometria (triângulo 2D):
   "image_data": {{
       "tipo": "triangulo_retangulo",
       "lados": {{"cateto_horizontal": "4 m", "cateto_vertical": "?", "hipotenusa": "5 m"}},
       "angulo_reto": "entre catetos",
       "incognita": "cateto_vertical"
   }}

   📐 Para geometria espacial (pirâmide 3D):
   "image_data": {{
       "tipo": "piramide_base_quadrada",
       "base_lado": "10 cm",
       "altura": "12 cm",
       "apotema_base": "5 cm",
       "meia_diagonal": "5√2 cm",
       "incognita": "aresta_lateral",
       "marcacao_interrogacao": "segmento V→vértice da base"
   }}

   🖼️ Para ilustração:
   "image_data": {{
       "tipo": "cena_ilustrativa",
       "descricao": "Balança comercial com frutas",
       "elementos": ["balança", "3 maçãs", "peso de 500g"]
   }}

🎯 TESTE FINAL: Leia sua questão SEM a imagem. Se conseguir responder, REFAÇA!
   O aluno DEVE OLHAR a imagem + RACIOCINAR para responder."""
        }
        
        # ⚡ Carrega APENAS a referência do componente correto (ou componentes combinados)
        skills_ref = _load_skills_reference_for(query)
        
        # Formata a string de habilidade
        skill_str = query.skill
        if getattr(query, "is_interdisciplinary", False) and hasattr(query, "combined_skills"):
            skill_str = "⚠️ OBJETIVO INTERDISCIPLINAR: A questão deve avaliar simultaneamente TODAS as habilidades abaixo em um único contexto coeso:\n"
            skill_str += "\n".join([
                f"- Componente: {item.curriculum_component} | Habilidade: {item.skill}"
                for item in query.combined_skills
            ])
            if progress:
                progress.log("generator", f"Modo Interdisciplinar ativado com {len(query.combined_skills)} habilidades", "", "🔗")

        # Tema/assunto complementar (orientação temática livre)
        context_theme = getattr(query, "context_theme", None)

        # Feedback de observações anteriores (HITL loop) — carrega observações
        # recentes em questões do mesmo contexto e injeta no prompt.
        corrective_observations = []
        try:
            from app.utils.connect_db import get_session_context
            from app.repositories.question_repository import QuestionRepository
            with get_session_context() as _s:
                _repo = QuestionRepository(_s)
                corrective_observations = _repo.get_corrective_observations(
                    skill=getattr(query, "skill", None),
                    grade=getattr(query, "grade", None),
                    curriculum_component=getattr(query, "curriculum_component", None),
                    limit=5,
                )
            if corrective_observations and progress:
                progress.log(
                    "generator",
                    f"Carregadas {len(corrective_observations)} observações corretivas anteriores",
                    "",
                    "📝",
                )
        except Exception as obs_err:
            logger.warning(f"⚠️ Falha ao carregar observações corretivas: {obs_err}")

        # Prepara inputs para o template
        inputs = {
            "count_questions": query.count_questions,
            "count_alternatives": query.count_alternatives,
            "skill": skill_str,
            "proficiency_level": query.proficiency_level,
            "grade": query.grade,
            "model_evaluation_type": query.model_evaluation_type.value,
            "image_dependency_instruction": image_instructions.get(
                image_dep, image_instructions["none"]
            ),
            "math_skills_reference": skills_ref.get("math", ""),
            "portuguese_skills_reference": skills_ref.get("portuguese", ""),
            "science_skills_reference": skills_ref.get("science", ""),
            "humanities_skills_reference": skills_ref.get("humanities", "")
        }
        
        # Se houver textos reais encontrados, injeta no prompt
        real_texts = state.get("real_texts")
        if real_texts:
            real_texts_str = "\n\n".join([
                f"--- TEXTO {i+1} ---\n"
                f"Título: {t.get('title', 'Sem título')}\n"
                f"Autor: {t.get('author', 'Desconhecido')}\n"
                f"Fonte: {t.get('source_name', 'Fonte Online')} ({t.get('source_url', '')})\n"
                f"Texto:\n{t.get('text', '')[:1500]}"
                for i, t in enumerate(real_texts[:query.count_questions])
            ])
            
            template_str = f"""
{template_str}

⚠️ ATENÇÃO: USE OS TEXTOS REAIS ABAIXO COMO BASE PARA AS QUESTÕES ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS PARA USO DOS TEXTOS REAIS:
1. Use EXATAMENTE os textos fornecidos abaixo (não invente textos)
2. Cite CORRETAMENTE a fonte e o autor no campo "source"
3. Adapte a extensão se necessário, mas mantenha a autoria original
4. Se não houver texto suficiente, use o texto mais adequado disponível

TEXTOS ENCONTRADOS NA BUSCA:
{real_texts_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            logger.info(f"📚 Injetando {len(real_texts)} textos reais no prompt")
            if progress:
                for i, t in enumerate(real_texts[:query.count_questions]):
                    title = t.get('title', 'Sem título')[:60]
                    author = t.get('author', 'Desconhecido')
                    progress.log("generator", f"Texto {i+1}: \"{title}\"", f"Autor: {author}", "📖")
        
        # Tema complementar — injetado no TOPO do template com alta prioridade,
        # para que o LLM contextualize o texto-base da questão nesse assunto
        # sem perder o foco na habilidade avaliada.
        if context_theme and context_theme.strip():
            theme = context_theme.strip()
            template_str = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 TEMA OBRIGATÓRIO DA QUESTÃO (orientação temática do usuário):
"{theme}"

COMO USAR O TEMA — OBRIGATÓRIO:

1. TEXTO-BASE (campo "text" do JSON):
   - Se o tema é uma OBRA ESPECÍFICA (poema, canção, conto, crônica, lei, reportagem):
     → Use O PRÓPRIO TEXTO dessa obra como texto-base.
     → Ex.: tema "Canção do Exílio" (Gonçalves Dias) → o texto-base DEVE ser o poema
       "Minha terra tem palmeiras / Onde canta o Sabiá / As aves que aqui gorjeiam / Não
       gorjeiam como lá…" com atribuição de autor no campo "source_author" e fonte real.
     → Se não souber o texto exato, escolha uma obra ESTRUTURALMENTE equivalente sobre o tema.
   - Se o tema é um EVENTO/ASSUNTO GERAL (ex.: "Semana da Água"):
     → Construa um texto-base informativo (artigo, crônica curta, notícia) sobre o tema.

2. ENUNCIADO (campo "question_statement" do JSON) — PADRÃO SAEB/SEAMA:
   ❌ NÃO descreva cenário, personagem ou contexto no enunciado.
   ❌ NÃO narre a história do texto-base no enunciado.
   ❌ NÃO introduza elementos novos no enunciado — tudo que o aluno precisa está no texto-base.
   ✅ O enunciado é UMA PERGUNTA DIRETA e CURTA sobre o texto-base já lido.
   ✅ Exemplos no padrão SAEB:
      • "Nesse texto, o trecho 'Minha terra tem palmeiras' expressa"
      • "A ideia principal desse poema é"
      • "No verso 'Onde canta o Sabiá', a palavra 'Sabiá' refere-se a"
      • "A intenção do eu-lírico ao comparar sua terra natal com outro lugar é"

3. REGRAS GERAIS DE TEMA:
   - O tema é o PANO DE FUNDO; a habilidade avaliada abaixo continua sendo o OBJETIVO.
   - NÃO troque o tema por outro (ex.: usuário pediu "Canção do Exílio" → NÃO substitua por "Bumba-meu-boi").
   - Se usar uma obra real, cite autor e fonte nos campos "source_author" e "source".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{template_str}
"""
            if progress:
                progress.log("generator", f"Tema complementar injetado: {theme[:80]}", "", "🎯")

        # Regras universais de DISTRATORES e ENUNCIADO — aplicadas
        # a toda questão para evitar respostas triviais por cópia literal.
        template_str = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛑 REGRAS CRÍTICAS DE CONSTRUÇÃO DO ITEM (padrão SAEB/SEAMA):

A) ALTERNATIVAS — DE ONDE VÊM OS DISTRATORES:
   1. TODOS os distratores DEVEM citar informações que APARECEM NO TEXTO-BASE,
      mas que NÃO respondem à pergunta. O aluno deve DISCRIMINAR entre informações
      concorrentes do próprio texto, não eliminar por conhecimento externo.
   2. PROIBIDO: distratores formados por fatos gerais fora do texto
      (ex.: "21 de abril = Tiradentes", se o texto não menciona Tiradentes).
   3. O texto-base DEVE conter ≥ 3 informações "concorrentes" ao dado-alvo
      (ex.: várias datas, várias ações, vários lugares, vários personagens)
      para que os distratores tenham de onde sair.

B) ALTERNATIVA CORRETA — NÃO PODE SER CÓPIA LITERAL:
   1. PROIBIDO copiar IPSIS LITTERIS o trecho do texto-base que responde à pergunta.
   2. A correta DEVE ser uma PARÁFRASE MÍNIMA do trecho-alvo:
      - Trocar preposição/artigo ("no dia 19" → "em 19 de abril")
      - Reformular sintaticamente ("os alunos pintam o rosto" → "a pintura do rosto é feita pelos alunos")
      - Nominalizar verbos ("ouvir histórias" → "escuta de histórias")
   3. A paráfrase NÃO pode exigir inferência (é D1/localizar): mantém SEMÂNTICA IDÊNTICA.

C) PARALELISMO DAS ALTERNATIVAS:
   1. TODAS as 4 (ou 5) alternativas devem ter a MESMA estrutura sintática:
      - Se uma começa com "No dia…", todas começam com "No/Em dia/mês…"
      - Se uma é sintagma nominal, todas são sintagma nominal.
   2. PROIBIDO: uma alternativa ser sensivelmente mais longa/curta que as outras.
   3. PROIBIDO: misturar formas (ex.: 3 datas exatas + 1 referência vaga como "no fim do mês").

D) ENUNCIADO — NÃO PODE ENTREGAR A RESPOSTA:
   1. O enunciado NÃO pode conter o dado que a pergunta procura.
   2. O enunciado NÃO pode ser redundante com a informação-alvo do texto.
   3. Frases canônicas: "De acordo com o texto…", "No texto, o trecho '…' mostra…",
      "A informação que indica X está em…".

E) ESTRATÉGIA ANTES DE GERAR:
   1. Escolha o "dado-alvo" da pergunta (ex.: a data 19/04).
   2. LISTE explicitamente 3 outras informações concorrentes do mesmo tipo no texto
      (ex.: "há duas semanas", "dia da apresentação", "última semana do mês").
      Se o texto não tem 3 concorrentes, REESCREVA o texto incluindo-os.
   3. Só então construa as alternativas: 1 correta (paráfrase do dado-alvo) + 3 distratores
      (cada um citando uma das informações concorrentes).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{template_str}
"""

        # Se houver feedback, adiciona ao prompt
        if feedback and progress:
            progress.log("generator", "Incorporando feedback da revisão anterior", feedback[:100] if feedback else "", "📝")
        if feedback:
            template_str = f"""
{template_str}

ATENÇÃO - FEEDBACK DA REVISÃO ANTERIOR (CORRIJA ESTES PROBLEMAS):
{feedback}

Gere novas questões corrigindo os problemas apontados acima.
"""

        # Observações corretivas de gerações anteriores (HITL). Injeta as
        # observações mais recentes de questões desta skill/série/componente
        # como regras a seguir nesta nova geração.
        if corrective_observations:
            bullets = "\n".join(
                f'- (Q#{o["question_id"]}) {o["observation"][:300]}'
                for o in corrective_observations
            )
            template_str = f"""
{template_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 FEEDBACK DE REVISÕES ANTERIORES (evite repetir estes problemas):
{bullets}

REGRA: cada uma dessas observações foi feita por um revisor humano ao
avaliar questões produzidas para esta mesma habilidade/série/componente.
Use-as como LIÇÕES APRENDIDAS na nova geração: corrija o padrão que
gerou a observação, não o texto específico.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Lembrete final do tema (última instrução = maior peso no LLM).
        # Reforça para que o texto-base incorpore o tema, mesmo quando a
        # habilidade avaliada for de componente diferente.
        if context_theme and context_theme.strip():
            theme_reminder = context_theme.strip()
            template_str = f"""
{template_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔔 LEMBRETE CRÍTICO — NÃO IGNORE:
O texto-base (campo "text"), o título (campo "title") e pelo menos
um elemento mencionado nas alternativas DEVEM citar EXPLICITAMENTE
"{theme_reminder}" ou aspectos diretamente ligados a esse tema
(personagens, datas, locais, causas, consequências da Conjuração/
do evento/da obra). Se você produzir um texto genérico sem mencionar
"{theme_reminder}", a questão será REPROVADA.

Exemplo ruim: "Os estados do Nordeste são…" (genérico, sem o tema)
Exemplo bom:  "A Conjuração Baiana, ocorrida em Salvador em 1798…"

Mesmo que a habilidade avaliada seja de componente diferente
(ex.: Geografia com tema de História), o texto-base DEVE ser sobre o
tema — o aluno responde à habilidade USANDO o texto temático.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Cria e executa a chain
        prompt = PromptTemplate(
            input_variables=list(inputs.keys()),
            template=template_str
        )
        
        chain = prompt | llm
        config = get_runnable_config(
            run_name=f"generator-attempt-{retry_count + 1}",
            tags=["langgraph", "generator"]
        )
        
        if progress:
            progress.log("generator", f"Chamando API do modelo...", f"Gerando {query.count_questions} questão(ões)", "🚀")
            progress.log("generator", "Aplicando regras de alinhamento BNCC/SAEB", "", "📏")
            progress.log("generator", "Construindo distratores plausíveis com base na taxonomia de erros", "", "🎭")
        response = chain.invoke(inputs, config=config)
        if progress:
            progress.log("generator", "Resposta recebida — validando estrutura JSON", "", "📥")
        
        # Extrai conteúdo — normaliza para string, já que Gemini pode retornar
        # content como lista de parts (ex: [{"type":"text","text":"..."}]).
        raw_content = response.content if hasattr(response, 'content') else str(response)
        if isinstance(raw_content, list):
            parts = []
            for part in raw_content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text") or part.get("content") or "")
                else:
                    parts.append(str(part))
            response_text = "\n".join(p for p in parts if p)
        else:
            response_text = str(raw_content)
        
        # Parse do JSON
        if progress:
            progress.log("generator", "Interpretando resposta JSON", "", "🔧")
        parsed_data = _parse_json_response(response_text)
        questions = parsed_data.get("questions", [])
        
        logger.info(f"✅ Gerador produziu {len(questions)} questões")
        if progress:
            progress.metric("generator", "Questões geradas", len(questions), "📝")
            for i, q in enumerate(questions):
                stmt = q.get("question_statement", "")[:80]
                progress.log("generator", f"Q{i+1}: {stmt}...", "", "✏️")
        
        return {
            **state,
            "questions": questions,
            "retry_count": retry_count + 1,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no Agente Gerador: {e}")
        if progress:
            progress.log("generator", f"Erro: {str(e)[:120]}", "", "❌")
        return {
            **state,
            "questions": [],
            "retry_count": retry_count + 1,
            "error": str(e)
        }
