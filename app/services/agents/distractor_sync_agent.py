"""
DistractorSyncAgent - Agente de Sincronização de Distratores com Imagem.

Ao regenerar uma imagem, este agente analisa se os distratores (explicações das
alternativas) continuam coerentes com a nova imagem. Se necessário, atualiza
os distratores automaticamente.

MODO MULTIMODAL: Quando recebe a imagem gerada (base64), usa Gemini Vision
para analisar o conteúdo real da imagem e corrigir alternativas, resposta
correta e distratores que não correspondam ao que foi gerado.
"""

import logging
import json
import base64
import os
from typing import Optional, Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas.question_schema import QuestionSchema
from app.core.llm_config import get_question_llm, get_runnable_config

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE - Análise e Atualização (modo texto, sem imagem)
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


# ============================================================================
# PROMPT - Validação Multimodal (com imagem real)
# ============================================================================

MULTIMODAL_VALIDATION_PROMPT = """Você é um especialista em avaliação educacional. Analise a IMAGEM GERADA e compare com as alternativas da questão abaixo.

═══════════════════════════════════════════════════════════════════════════════
📋 DADOS DA QUESTÃO
═══════════════════════════════════════════════════════════════════════════════

🏷️ TÍTULO: {title}

📖 TEXTO-BASE:
{text}

❓ ENUNCIADO:
{question_statement}

✅ RESPOSTA CORRETA ATUAL: {correct_answer}

💡 EXPLICAÇÃO:
{explanation}

📋 ALTERNATIVAS ATUAIS:
{alternatives_text}

═══════════════════════════════════════════════════════════════════════════════
🎯 SUA TAREFA
═══════════════════════════════════════════════════════════════════════════════

OBSERVE A IMAGEM GERADA com atenção e responda:

1. PARA CADA ALTERNATIVA: O elemento visual que ela menciona EXISTE na imagem?
   - Se a alternativa diz "há um gráfico mostrando X" → o gráfico existe na imagem?
   - Se a alternativa diz "uma pessoa segura um copo" → essa pessoa aparece?
   - Se a alternativa menciona "cores vibrantes" → o cartaz tem cores vibrantes?

2. A ALTERNATIVA CORRETA ainda é válida com base no que a imagem realmente mostra?
   - Se a imagem mudou e agora outra alternativa é a correta → MUDE a resposta correta

3. Os DISTRATORES (explicações) fazem sentido com o conteúdo real da imagem?

REGRAS DE CORREÇÃO:
- Se uma alternativa menciona algo que NÃO existe na imagem → REESCREVA a alternativa para mencionar algo que EXISTE na imagem, mantendo o mesmo tipo de erro pedagógico
- Se a resposta correta não corresponde mais à imagem → mude para a alternativa que melhor corresponde
- Atualize os distratores para refletir o conteúdo real da imagem
- Mantenha o nível pedagógico e o estilo original
- Cada alternativa DEVE referenciar um elemento visual REAL da imagem

═══════════════════════════════════════════════════════════════════════════════
📝 FORMATO DE RESPOSTA (JSON)
═══════════════════════════════════════════════════════════════════════════════

Responda EXATAMENTE neste formato JSON:

{{
    "correct_answer": "C",
    "alternatives": [
        {{
            "letter": "A",
            "text": "texto da alternativa (corrigido se necessário)",
            "distractor": "distrator atualizado",
            "modified": true ou false,
            "text_modified": true ou false
        }},
        ...
    ],
    "summary": "Breve resumo das correções feitas"
}}

CAMPO "correct_answer": letra da alternativa correta (pode ser diferente da original se a imagem mudou)
CAMPO "text_modified": true SOMENTE se o TEXTO da alternativa foi alterado (não o distrator)
CAMPO "modified": true se QUALQUER coisa foi alterada (texto ou distrator)
Retorne TODAS as alternativas, mesmo as não modificadas.
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
    Agente que analisa e sincroniza distratores/alternativas com a imagem gerada.

    Dois modos:
    1. TEXTO: Analisa baseado nas instruções de modificação (sem ver a imagem)
    2. MULTIMODAL: Analisa a imagem real gerada e corrige alternativas/distratores
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
        
        # Inicializa cliente Gemini para análise multimodal
        self._genai_client = None
        logger.info("🔄 DistractorSyncAgent inicializado")

    def _get_genai_client(self):
        """Obtém o cliente GenAI para análise multimodal (lazy init)."""
        if self._genai_client is None:
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY não configurada")
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

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
        Analisa e sincroniza distratores com a nova imagem (modo texto).

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
            return {
                "alternatives": [
                    {
                        "letter": alt.letter,
                        "text": alt.text,
                        "distractor": alt.distractor,
                        "modified": False,
                        "text_modified": False
                    }
                    for alt in question.alternatives
                ],
                "distractors_updated": False,
                "summary": f"Erro na análise: {str(e)}"
            }

    def validate_with_image(
        self,
        question: QuestionSchema,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Analisa a imagem REAL gerada e corrige alternativas/distratores.
        
        Usa Gemini Vision para comparar o conteúdo visual da imagem com
        cada alternativa, corrigindo textos, resposta correta e distratores.

        Args:
            question: Questão educacional completa
            image_base64: Imagem gerada em base64

        Returns:
            Dict com alternativas corrigidas, resposta correta atualizada e metadados
        """
        logger.info(f"🔍 Validando imagem vs alternativas para: {question.title[:50]}...")

        try:
            from google.genai import types
            
            client = self._get_genai_client()
            
            # Decodifica a imagem
            image_bytes = base64.b64decode(image_base64)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            
            # Monta o prompt com dados da questão
            prompt_text = MULTIMODAL_VALIDATION_PROMPT.format(
                title=question.title,
                text=question.text[:500] if question.text else "Observe a imagem a seguir.",
                question_statement=question.question_statement[:500],
                correct_answer=self._extract_correct_answer(question),
                explanation=question.explanation_question[:400] if question.explanation_question else "N/A",
                alternatives_text=self._format_alternatives(question),
            )
            
            # Envia imagem + prompt para Gemini Vision
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=[image_part, prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            # Parse da resposta
            result = _parse_sync_response(response.text)
            
            alternatives = result.get("alternatives", [])
            new_correct = result.get("correct_answer", question.correct_answer)
            any_modified = any(alt.get("modified", False) for alt in alternatives)
            any_text_modified = any(alt.get("text_modified", False) for alt in alternatives)
            correct_changed = new_correct != question.correct_answer
            
            changes_desc = []
            if any_text_modified:
                changes_desc.append("textos de alternativas")
            if any_modified and not any_text_modified:
                changes_desc.append("distratores")
            if correct_changed:
                changes_desc.append(f"resposta correta ({question.correct_answer}→{new_correct})")
            
            if changes_desc:
                logger.info(f"✅ Validação multimodal: alterados {', '.join(changes_desc)}")
            else:
                logger.info("✨ Validação multimodal: nenhuma alteração necessária")
            
            return {
                "alternatives": alternatives,
                "distractors_updated": any_modified or correct_changed,
                "correct_answer": new_correct,
                "correct_answer_changed": correct_changed,
                "summary": result.get("summary", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na validação multimodal: {e}")
            return {
                "alternatives": [
                    {
                        "letter": alt.letter,
                        "text": alt.text,
                        "distractor": alt.distractor,
                        "modified": False,
                        "text_modified": False
                    }
                    for alt in question.alternatives
                ],
                "distractors_updated": False,
                "correct_answer": question.correct_answer,
                "correct_answer_changed": False,
                "summary": f"Erro na validação: {str(e)}"
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
