"""
DistractorSyncAgent - Agente de Sincronização de Distratores com Imagem.

Ao regenerar uma imagem, este agente analisa se os distratores (explicações das
alternativas) continuam coerentes com a nova imagem. Se necessário, atualiza
os distratores automaticamente.
"""

import logging
import json
from typing import Optional, Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas.question_schema import QuestionSchema
from app.core.llm_config import get_question_llm, get_runnable_config

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE - Análise e Atualização de Distratores
# ============================================================================

DISTRACTOR_SYNC_TEMPLATE = """Você é um especialista em avaliação educacional responsável por garantir a coerência entre distratores (explicações das alternativas) e a imagem de uma questão.

═══════════════════════════════════════════════════════════════════════════════
📋 DADOS DA QUESTÃO
═══════════════════════════════════════════════════════════════════════════════

🏷️ TÍTULO: {title}

📖 TEXTO-BASE:
{text}

❓ ENUNCIADO:
{question_statement}

✅ RESPOSTA CORRETA: {correct_answer}

💡 EXPLICAÇÃO:
{explanation}

📋 ALTERNATIVAS E DISTRATORES ATUAIS:
{alternatives_text}

═══════════════════════════════════════════════════════════════════════════════
🔄 INSTRUÇÕES DE MODIFICAÇÃO DA IMAGEM
═══════════════════════════════════════════════════════════════════════════════

O usuário solicitou a seguinte alteração na imagem:
{image_instructions}

═══════════════════════════════════════════════════════════════════════════════
🎯 SUA TAREFA
═══════════════════════════════════════════════════════════════════════════════

Analise cada distrator (explicação) e verifique se ele ainda é coerente após a mudança na imagem.

REGRAS:
1. Se um distrator faz referência a um elemento visual que MUDOU na imagem → ATUALIZE o distrator
2. Se um distrator é puramente textual/conceitual e NÃO depende da imagem → MANTENHA inalterado
3. Se a mudança na imagem altera a lógica de por que uma alternativa é correta/incorreta → ATUALIZE
4. Mantenha o mesmo estilo, tom e nível de detalhamento dos distratores originais
5. Distratores devem ser pedagogicamente plausíveis e coerentes com o nível escolar da questão
6. NUNCA remova informação pedagógica relevante - apenas ajuste referências visuais

═══════════════════════════════════════════════════════════════════════════════
📝 FORMATO DE RESPOSTA (JSON)
═══════════════════════════════════════════════════════════════════════════════

Responda EXATAMENTE neste formato JSON:

{{
    "alternatives": [
        {{
            "letter": "A",
            "text": "texto original da alternativa (NÃO altere o texto)",
            "distractor": "distrator atualizado ou mantido",
            "modified": true ou false
        }},
        ...
    ],
    "summary": "Breve resumo do que foi alterado (ou 'Nenhuma alteração necessária')"
}}

IMPORTANTE:
- O campo "text" deve conter o texto ORIGINAL da alternativa, sem alterações
- O campo "modified" deve ser true APENAS se o distrator foi alterado
- Retorne TODAS as alternativas, mesmo as não modificadas
"""


def _parse_sync_response(response_text: str) -> Dict[str, Any]:
    """Parse a resposta JSON do agente de sincronização."""
    text = response_text.strip()

    # Remove markdown code blocks se presentes
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Encontra o JSON na resposta
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("JSON não encontrado na resposta do DistractorSyncAgent")

    # Encontra o fechamento do JSON
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


class DistractorSyncAgent:
    """
    Agente que analisa e sincroniza distratores após regeneração de imagem.

    Fluxo:
    1. Recebe a questão completa + instruções de modificação da imagem
    2. Analisa cada distrator em relação às mudanças visuais
    3. Atualiza distratores que fazem referência a elementos visuais alterados
    4. Retorna alternativas com distratores atualizados
    """

    def __init__(self):
        """Inicializa o agente com o LLM configurado."""
        self.llm = get_question_llm()
        self.prompt_template = PromptTemplate(
            input_variables=[
                "title", "text", "question_statement",
                "correct_answer", "explanation",
                "alternatives_text", "image_instructions"
            ],
            template=DISTRACTOR_SYNC_TEMPLATE
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("🔄 DistractorSyncAgent inicializado")

    def _format_alternatives(self, question: QuestionSchema) -> str:
        """Formata as alternativas e distratores para o prompt."""
        lines = []
        for alt in question.alternatives:
            is_correct = alt.letter == question.correct_answer
            status = "✅ CORRETA" if is_correct else "❌ INCORRETA"
            lines.append(f"{alt.letter}) {alt.text} [{status}]")
            if alt.distractor:
                lines.append(f"   Distrator: {alt.distractor}")
            else:
                lines.append("   Distrator: (não definido)")
            lines.append("")
        return "\n".join(lines)

    def _extract_correct_answer(self, question: QuestionSchema) -> str:
        """Extrai o texto da alternativa correta."""
        for alt in question.alternatives:
            if alt.letter == question.correct_answer:
                return f"{alt.letter}) {alt.text}"
        return "N/A"

    def sync_distractors(
        self,
        question: QuestionSchema,
        image_instructions: str
    ) -> Dict[str, Any]:
        """
        Analisa e sincroniza distratores com a nova imagem.

        Args:
            question: Questão educacional completa
            image_instructions: Instruções de modificação da imagem

        Returns:
            Dict com alternativas atualizadas e metadados
        """
        logger.info(f"🔄 Sincronizando distratores para: {question.title[:50]}...")

        inputs = {
            "title": question.title,
            "text": question.text[:500] if question.text else "Observe a imagem a seguir.",
            "question_statement": question.question_statement[:500],
            "correct_answer": self._extract_correct_answer(question),
            "explanation": question.explanation_question[:400] if question.explanation_question else "N/A",
            "alternatives_text": self._format_alternatives(question),
            "image_instructions": image_instructions,
        }

        try:
            config = get_runnable_config(
                run_name="distractor-sync",
                tags=["distractor", "sync", "image"]
            )

            response = self.chain.invoke(inputs, config=config)
            result = _parse_sync_response(response)

            # Verifica se houve alguma mudança
            alternatives = result.get("alternatives", [])
            any_modified = any(alt.get("modified", False) for alt in alternatives)

            logger.info(
                f"{'✅' if any_modified else '✨'} "
                f"Sincronização concluída: "
                f"{'Distratores atualizados' if any_modified else 'Nenhuma alteração necessária'}"
            )

            return {
                "alternatives": alternatives,
                "distractors_updated": any_modified,
                "summary": result.get("summary", "")
            }

        except Exception as e:
            logger.error(f"❌ Erro na sincronização de distratores: {e}")
            # Em caso de erro, retorna as alternativas originais sem modificação
            return {
                "alternatives": [
                    {
                        "letter": alt.letter,
                        "text": alt.text,
                        "distractor": alt.distractor,
                        "modified": False
                    }
                    for alt in question.alternatives
                ],
                "distractors_updated": False,
                "summary": f"Erro na análise: {str(e)}"
            }


# ============================================================================
# Singleton Instance
# ============================================================================

_agent_instance: Optional[DistractorSyncAgent] = None


def get_distractor_sync_agent() -> DistractorSyncAgent:
    """
    Obtém a instância singleton do DistractorSyncAgent.

    Returns:
        Instância do agente
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DistractorSyncAgent()
    return _agent_instance
