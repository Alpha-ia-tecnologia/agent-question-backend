"""
ImageValidatorAgent - Agente de Validação de Imagem com Gemini Vision.

Analisa a imagem gerada JUNTO com a questão para verificar:
1. Contagem de itens (setores, barras) bate com o enunciado
2. Rótulos/valores visíveis e legíveis
3. "?" no lugar correto (geometria)
4. Imagem permite resolver a questão
"""

import logging
import json
import base64
import os
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

VALIDATION_PROMPT = """Você é um revisor especializado em questões educacionais com imagem.

Analise a IMAGEM fornecida junto com os DADOS DA QUESTÃO abaixo e verifique se a imagem
é coerente, completa e permite que o aluno resolva a questão corretamente.

═══════════════════════════════════════════════════════════════
📋 DADOS DA QUESTÃO
═══════════════════════════════════════════════════════════════

🏷️ TÍTULO: {title}
📖 TEXTO-BASE: {text}
❓ ENUNCIADO: {question_statement}
✅ RESPOSTA CORRETA: {correct_answer}
💡 EXPLICAÇÃO: {explanation}

📊 DADOS ESTRUTURADOS (o que a imagem DEVERIA conter):
{image_data}

═══════════════════════════════════════════════════════════════
🔍 CHECKLIST DE VALIDAÇÃO
═══════════════════════════════════════════════════════════════

Verifique CADA item abaixo e marque como ✅ OK ou ❌ FALHA:

1. CONTAGEM: Se o enunciado menciona N itens (ex: "quatro municípios"),
   a imagem tem EXATAMENTE N elementos?

2. RÓTULOS: Os rótulos/legendas estão visíveis e legíveis?
   (nomes, percentuais, valores numéricos)

3. VALORES: Se é um gráfico, os valores/percentuais estão presentes?
   (eixos numéricos, percentuais nos setores, valores nas barras)

4. COERÊNCIA: A imagem é coerente com o tema da questão?
   (não mostra informação contraditória com o enunciado)

5. RESOLUBILIDADE: É possível RESOLVER a questão usando a imagem?
   (os dados necessários para o cálculo estão visíveis)

6. GEOMETRIA (se aplicável):
   - O "?" está no lado CORRETO? (cateto vs hipotenusa)
   - O ângulo de 90° está marcado?
   - As medidas visíveis batem com o cálculo?

═══════════════════════════════════════════════════════════════
📝 RESPONDA EXATAMENTE NESTE FORMATO JSON:
═══════════════════════════════════════════════════════════════

{{
    "valid": true ou false,
    "score": 0.0 a 1.0,
    "checks": {{
        "contagem": {{"ok": true/false, "detail": "..."}},
        "rotulos": {{"ok": true/false, "detail": "..."}},
        "valores": {{"ok": true/false, "detail": "..."}},
        "coerencia": {{"ok": true/false, "detail": "..."}},
        "resolubilidade": {{"ok": true/false, "detail": "..."}}
    }},
    "issues": ["lista de problemas encontrados"],
    "corrections": "Instruções específicas para corrigir a imagem (se inválida)"
}}

REGRAS:
- "valid" = true SOMENTE se TODOS os checks forem OK
- "score" = proporção de checks que passaram (5/5 = 1.0, 4/5 = 0.8, etc)
- Se "valid" = false, "corrections" DEVE conter instruções claras para regenerar
"""


def _parse_validation_response(response_text: str) -> Dict[str, Any]:
    """Parse a resposta JSON do validador."""
    text = response_text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    start_idx = text.find('{')
    if start_idx == -1:
        return {"valid": False, "score": 0, "issues": ["Resposta inválida do validador"], "corrections": "Regenerar a imagem"}
    
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
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"valid": False, "score": 0, "issues": ["JSON inválido na resposta"], "corrections": "Regenerar a imagem"}


class ImageValidatorAgent:
    """
    Agente que valida imagens usando Gemini Vision (multimodal).
    
    Recebe a questão + imagem base64 e verifica se a imagem é coerente,
    completa e permite resolver a questão.
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)
        self.model = "gemini-2.0-flash"
        logger.info("👁️ ImageValidatorAgent inicializado (Gemini Vision)")
    
    def validate(self, question: dict, image_base64: str) -> Dict[str, Any]:
        """
        Valida uma imagem contra os dados da questão.
        
        Args:
            question: Dicionário com dados da questão
            image_base64: Imagem em base64
            
        Returns:
            Dict com resultado da validação: {valid, score, issues, corrections}
        """
        title = question.get("title", "N/A")
        logger.info(f"👁️ Validando imagem para: {title[:50]}...")
        
        # Extrair alternativa correta
        correct_answer_text = "N/A"
        correct_letter = question.get("correct_answer", "")
        for alt in question.get("alternatives", []):
            if alt.get("letter") == correct_letter:
                correct_answer_text = f"{alt['letter']}) {alt.get('text', '')}"
                break
        
        # Formatar image_data
        image_data_str = "Nenhum dado estruturado disponível."
        if question.get("image_data"):
            try:
                image_data_str = json.dumps(question["image_data"], ensure_ascii=False, indent=2)
            except Exception:
                image_data_str = str(question["image_data"])
        
        # Montar prompt
        prompt_text = VALIDATION_PROMPT.format(
            title=title,
            text=question.get("text", "N/A")[:500],
            question_statement=question.get("question_statement", "N/A")[:500],
            correct_answer=correct_answer_text,
            explanation=question.get("explanation_question", "N/A")[:400],
            image_data=image_data_str
        )
        
        try:
            # Decodificar imagem
            image_bytes = base64.b64decode(image_base64)
            
            # Enviar ao Gemini Vision (multimodal)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_text(text=prompt_text),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="text/plain",
                ),
            )
            
            result = _parse_validation_response(response.text)
            
            is_valid = result.get("valid", False)
            score = result.get("score", 0)
            issues = result.get("issues", [])
            
            if is_valid:
                logger.info(f"✅ Imagem VÁLIDA (score: {score}) para: {title[:50]}")
            else:
                logger.warning(
                    f"❌ Imagem INVÁLIDA (score: {score}) para: {title[:50]} | "
                    f"Issues: {issues}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na validação de imagem: {e}")
            return {
                "valid": False,
                "score": 0,
                "issues": [f"Erro na validação: {str(e)}"],
                "corrections": "Regenerar a imagem devido a erro na validação"
            }


# Singleton
_validator_instance: Optional[ImageValidatorAgent] = None


def get_image_validator_agent() -> ImageValidatorAgent:
    """Obtém instância singleton do ImageValidatorAgent."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ImageValidatorAgent()
    return _validator_instance
