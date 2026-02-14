"""
Agente Revisor de Questões.

Responsável por avaliar a qualidade pedagógica das questões geradas,
verificando alinhamento BNCC, distratores, clareza e proficiência.
"""

import logging
import json
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate

from app.services.agents.state import AgentState
from app.core.llm_config import get_question_llm, get_runnable_config
from app.services.progress_manager import get_current_progress

logger = logging.getLogger(__name__)


REVIEWER_PROMPT = """
Você é um especialista em avaliações educacionais brasileiras (SAEB, SEAMA, BNCC).

Sua tarefa é revisar as questões geradas e avaliar a qualidade pedagógica.

QUESTÕES PARA REVISAR:
{questions_json}

HABILIDADE SOLICITADA: {skill}
NÍVEL DE PROFICIÊNCIA: {proficiency_level}
ANO/SÉRIE: {grade}

---

Para CADA questão, avalie os seguintes CRITÉRIOS (nota de 0 a 10):

1. **ALINHAMENTO_BNCC**: A questão contempla corretamente a habilidade?
2. **DISTRATORES**: Avalie rigorosamente a qualidade dos distratores:
   a) Cada alternativa incorreta é PLAUSÍVEL (não óbvia/absurda)?
   b) Cada distrator representa um ERRO CONCEITUAL REAL (leitura superficial, extrapolação, redução, contradição, foco irrelevante, erro de cálculo, confusão de conceitos)?
   c) Os distratores têm MESMA ESTRUTURA GRAMATICAL e TAMANHO SIMILAR (±20%) ao gabarito?
   d) O campo "distractor" de cada alternativa EXPLICA o erro conceitual específico?
   e) NENHUM distrator é absurdo, humorístico, fora de contexto ou gramaticalmente inconsistente?
   Nota 10: todos os critérios. Nota 5: plausíveis mas sem tipagem de erro. Nota 0: absurdos/óbvios.
3. **CLAREZA**: O enunciado é claro, sem ambiguidades ou erros?
4. **PROFICIENCIA**: O nível de dificuldade está adequado ao nível solicitado?
5. **TEXTO_BASE**: O texto suporte é relevante, autêntico e apropriado?
6. **COERENCIA_IMAGEM**: Se a questão usa imagem, verifique:
   - A resposta NÃO é diretamente visível na imagem (requer cálculo/inferência)?
   - Se for matemática: hipotenusa é o maior lado? Proporções são corretas?
   - O enunciado NÃO repete dados que já aparecem na imagem?
   - O texto NÃO descreve detalhadamente a imagem?
   (Se não usa imagem, dê nota 10)
7. **COERENCIA_MATEMATICA_3D**: Se a questão envolve geometria espacial (pirâmide, cone, prisma), verifique:
   - A TERMINOLOGIA está correta? (aresta lateral ≠ apótema da pirâmide)
   - Aresta lateral de pirâmide = V→vértice da base, usa metade da DIAGONAL ((lado×√2)/2)
   - Apótema da pirâmide = V→ponto médio do lado, usa apótema da BASE (lado/2)
   - O CÁLCULO numérico usa a fórmula correspondente ao TERMO do enunciado?
   - A resposta numérica bate com a fórmula correta?
   - O "?" na imagem marca o segmento correto?
   (Se não envolve geometria 3D, dê nota 10)

---

RESPONDA EXCLUSIVAMENTE no formato JSON abaixo:
{{
    "reviews": [
        {{
            "question_number": 1,
            "scores": {{
                "alinhamento_bncc": X,
                "distratores": X,
                "clareza": X,
                "proficiencia": X,
                "texto_base": X,
                "coerencia_imagem": X,
                "coerencia_matematica_3d": X
            }},
            "issues": ["Lista de problemas encontrados, se houver"],
            "suggestions": ["Sugestões de melhoria, se necessário"]
        }}
    ],
    "overall_score": X.X,
    "approved": true/false,
    "summary_feedback": "Resumo geral do feedback para regeneração, se reprovado"
}}

REGRAS:
- overall_score = média de todas as notas / 10 (resultado entre 0.0 e 1.0)
- approved = true se overall_score >= 0.7
- Se approved = false, preencha summary_feedback com instruções claras de correção
- ESPECIALMENTE verifique se questões com imagem têm a RESPOSTA visível (isso é GRAVE)
"""


def _parse_review_response(response_text: str) -> dict:
    """Parse da resposta JSON do revisor."""
    text = response_text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("JSON não encontrado na resposta do revisor")
    
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


def reviewer_node(state: AgentState) -> AgentState:
    """
    Nó do Agente Revisor.
    
    Avalia a qualidade pedagógica das questões e retorna
    uma pontuação de qualidade e feedback para regeneração.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com pontuação e feedback
    """
    questions = state.get("questions", [])
    query = state["query"]
    
    if not questions:
        logger.warning("⚠️ Revisor: Nenhuma questão para revisar")
        return {
            **state,
            "quality_score": 0.0,
            "revision_feedback": "Nenhuma questão foi gerada. Tente novamente."
        }
    
    logger.info(f"🟡 Agente Revisor - Avaliando {len(questions)} questões")
    
    progress = get_current_progress()
    
    try:
        if progress:
            progress.log("reviewer", "Initializing review LLM", "", "🔌")
            progress.log("reviewer", "Loading 7 quality criteria (BNCC, Distractors, Clarity...)", "", "📋")
        llm = get_question_llm()
        
        # Prepara o prompt
        prompt = PromptTemplate(
            input_variables=["questions_json", "skill", "proficiency_level", "grade"],
            template=REVIEWER_PROMPT
        )
        
        chain = prompt | llm
        config = get_runnable_config(
            run_name="reviewer-evaluation",
            tags=["langgraph", "reviewer"]
        )
        
        inputs = {
            "questions_json": json.dumps(questions, ensure_ascii=False, indent=2),
            "skill": query.skill,
            "proficiency_level": query.proficiency_level,
            "grade": query.grade
        }
        
        if progress:
            progress.log("reviewer", "Building evaluation prompt", f"{len(questions)} questions to review", "📋")
            progress.log("reviewer", "Checking distractor plausibility (5 sub-criteria)", "", "🎭")
            progress.log("reviewer", "Calling DeepSeek API (review)...", "", "🚀")
        response = chain.invoke(inputs, config=config)
        response_text = response.content if hasattr(response, 'content') else str(response)
        if progress:
            progress.log("reviewer", "Review response received", "", "📥")
            progress.log("reviewer", "Analyzing scores per criterion...", "", "📊")
        
        # Parse da revisão
        if progress:
            progress.log("reviewer", "Parsing evaluation response", "", "🔧")
        review_data = _parse_review_response(response_text)
        
        overall_score = review_data.get("overall_score", 0.0)
        approved = review_data.get("approved", False)
        feedback = review_data.get("summary_feedback", None) if not approved else None
        
        logger.info(
            f"{'✅' if approved else '⚠️'} Revisor - Score: {overall_score:.2f} | "
            f"Aprovado: {approved}"
        )
        
        if progress:
            reviews = review_data.get("reviews", [])
            for rev in reviews:
                qnum = rev.get("question_number", "?")
                scores = rev.get("scores", {})
                for criteria, score_val in scores.items():
                    criteria_label = criteria.replace("_", " ").title()
                    progress.log("reviewer", f"Q{qnum} — {criteria_label}: {score_val}/10", "", "📏")
                issues = rev.get("issues", [])
                for issue in issues:
                    progress.log("reviewer", f"Q{qnum} issue: {issue[:80]}", "", "⚠️")
            
            score_pct = f"{overall_score * 100:.0f}%"
            progress.metric("reviewer", "Overall quality score", score_pct, "🎯")
            progress.metric("reviewer", "Approved", "✅ Yes" if approved else "❌ No", "📋")
        
        if not approved and feedback:
            logger.info(f"📝 Feedback: {feedback[:100]}...")
            if progress:
                progress.log("reviewer", f"Feedback: {feedback[:120]}", "", "📝")
        
        return {
            **state,
            "quality_score": overall_score,
            "revision_feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no Agente Revisor: {e}")
        if progress:
            progress.log("reviewer", f"Error: {str(e)[:120]}", "", "❌")
        # Em caso de erro, aprova para não travar o fluxo
        return {
            **state,
            "quality_score": 0.75,  # Score neutro para continuar
            "revision_feedback": None,
            "error": f"Erro na revisão: {e}"
        }
