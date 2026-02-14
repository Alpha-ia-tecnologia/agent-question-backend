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
# Apenas Matemática e Língua Portuguesa (CN e CH removidos para velocidade)
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
}


def _load_skills_reference_for(query) -> dict[str, str]:
    """
    Carrega APENAS a referência de habilidades do componente curricular correto.
    
    Reduz o prompt de ~97KB (4 arquivos) para ~17-31KB (1 arquivo),
    acelerando significativamente a resposta do LLM.
    """
    component = getattr(query, "curriculum_component", "").lower().strip()
    skill = getattr(query, "skill", "").lower().strip()
    search_text = f"{component} {skill}"
    
    # Detecta componente por keywords
    detected = None
    best_score = 0
    
    for comp_id, info in _COMPONENT_MAP.items():
        score = sum(1 for kw in info["keywords"] if kw in search_text)
        if score > best_score:
            best_score = score
            detected = comp_id
    
    # Se não detectou, tenta pelo curriculum_component direto
    if not detected and component:
        for comp_id, info in _COMPONENT_MAP.items():
            if comp_id in component or info["key"] in component:
                detected = comp_id
                break
    
    result = {}
    
    if detected:
        info = _COMPONENT_MAP[detected]
        ref_path = os.path.abspath(info["file"])
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                result[info["key"]] = f.read()
            logger.info(f"⚡ Carregada referência: {detected} ({os.path.getsize(ref_path) // 1024}KB)")
        except FileNotFoundError:
            logger.warning(f"⚠️ {info['file']} não encontrado")
    else:
        # Fallback: componente não detectado → carrega todos (comportamento antigo)
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
        # Obtém LLM e template
        if progress:
            progress.log("generator", "Initializing DeepSeek LLM", "", "🔌")
        llm = get_question_llm()
        template_str = _select_template(query, feedback is not None)
        image_dep = query.image_dependency
        if progress:
            progress.log("generator", f"📐 Skill: {query.skill[:60]}", "", "🎯")
            progress.log("generator", f"Grade: {query.grade} · Proficiency: {query.proficiency_level}", "", "🏫")
            dep_label = {"none": "No image", "optional": "Optional image", "required": "Image required"}
            progress.log("generator", f"Image rules: {dep_label.get(image_dep, image_dep)}", "", "🖼️")
            progress.log("generator", "Loading distractor methodology (7 error types)", "", "🧠")
            tpl = "with feedback" if feedback else "standard"
            progress.log("generator", f"Template selected: {tpl}", "", "📄")
        
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
        
        # ⚡ Carrega APENAS a referência do componente correto (~30KB vs ~97KB)
        skills_ref = _load_skills_reference_for(query)
        
        # Prepara inputs para o template
        inputs = {
            "count_questions": query.count_questions,
            "count_alternatives": query.count_alternatives,
            "skill": query.skill,
            "proficiency_level": query.proficiency_level,
            "grade": query.grade,
            "model_evaluation_type": query.model_evaluation_type.value,
            "image_dependency_instruction": image_instructions.get(
                image_dep, image_instructions["none"]
            ),
            "math_skills_reference": skills_ref.get("math", ""),
            "portuguese_skills_reference": skills_ref.get("portuguese", ""),
            "science_skills_reference": "",
            "humanities_skills_reference": ""
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
        
        # Se houver feedback, adiciona ao prompt
        if feedback and progress:
            progress.log("generator", "Incorporating feedback from previous review", feedback[:100] if feedback else "", "📝")
        if feedback:
            template_str = f"""
{template_str}

ATENÇÃO - FEEDBACK DA REVISÃO ANTERIOR (CORRIJA ESTES PROBLEMAS):
{feedback}

Gere novas questões corrigindo os problemas apontados acima.
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
            progress.log("generator", f"Calling DeepSeek API...", f"Generating {query.count_questions} question(s)", "🚀")
            progress.log("generator", "Applying BNCC/SAEB alignment rules", "", "📏")
            progress.log("generator", "Crafting plausible distractors based on error taxonomy", "", "🎭")
        response = chain.invoke(inputs, config=config)
        if progress:
            progress.log("generator", "API response received — Validating JSON structure", "", "📥")
        
        # Extrai conteúdo
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse do JSON
        if progress:
            progress.log("generator", "Parsing JSON response", "", "🔧")
        parsed_data = _parse_json_response(response_text)
        questions = parsed_data.get("questions", [])
        
        logger.info(f"✅ Gerador produziu {len(questions)} questões")
        if progress:
            progress.metric("generator", "Questions generated", len(questions), "📝")
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
            progress.log("generator", f"Error: {str(e)[:120]}", "", "❌")
        return {
            **state,
            "questions": [],
            "retry_count": retry_count + 1,
            "error": str(e)
        }
