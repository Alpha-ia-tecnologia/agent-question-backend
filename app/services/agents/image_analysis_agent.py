"""
Agente de Análise de Imagem.

Responsável por analisar o contexto da questão (título, enunciado, alternativa correta)
e gerar um prompt inteligente para criação de imagem coerente.
"""

import logging
import json
from typing import Optional

from langchain_core.prompts import PromptTemplate

from app.schemas.question_schema import QuestionSchema
from app.core.llm_config import get_question_llm, get_runnable_config

logger = logging.getLogger(__name__)


IMAGE_ANALYSIS_PROMPT = """
Você é um especialista em criar prompts de imagem para questões educacionais.

Sua tarefa é analisar a questão abaixo e gerar um PROMPT DETALHADO para uma IA de geração de imagens (DALL-E/GPT-Image).

📋 DADOS DA QUESTÃO:
- TÍTULO: {title}
- ENUNCIADO: {question_statement}
- ALTERNATIVA CORRETA: {correct_answer_text}
- EXPLICAÇÃO: {explanation}

🎯 SUA ANÁLISE DEVE:

1. **IDENTIFICAR PERSONAGENS**:
   - Extraia nomes do título (ex: "A Escolha de Bia" → personagem chamado "Bia")
   - Determine o GÊNERO de cada personagem pelo nome (Bia, Maria, Ana → menina; João, Pedro → menino)
   - Se não houver nome, use "criança" ou "estudante"

2. **IDENTIFICAR O TEMA CENTRAL**:
   - Qual é o conflito/situação principal? (ex: escolha entre videogame e estudo)
   - Quais objetos são essenciais? (ex: controle, livro, celular)

3. **DEFINIR A CENA**:
   - Onde se passa? (quarto, sala, escola)
   - Quantos personagens aparecem?
   - Qual a expressão/emoção do personagem?

4. **VERIFICAR FORMATO**:
   - O enunciado menciona "tirinha" ou "quadrinhos"? → Gerar imagem com múltiplos quadros
   - Menciona apenas "imagem" ou "cena"? → Gerar cena única
   - Menciona "figura", "diagrama"? → Gerar ilustração técnica

⚠️ REGRAS CRÍTICAS:
- A imagem NÃO pode revelar a resposta da questão
- Se o título diz "Bia", a imagem DEVE mostrar uma MENINA, não um menino
- Se menciona "tirinha", a imagem DEVE ser uma sequência de quadros

Responda APENAS no formato JSON:
{{
    "character_analysis": {{
        "names": ["lista de nomes encontrados"],
        "genders": {{"nome": "masculino/feminino"}},
        "count": número de personagens
    }},
    "scene_analysis": {{
        "location": "descrição do local",
        "key_objects": ["lista de objetos importantes"],
        "main_emotion": "emoção principal do personagem",
        "conflict": "descrição breve do conflito"
    }},
    "format": "tirinha_3_quadros" | "tirinha_4_quadros" | "cena_unica" | "diagrama",
    "image_prompt": "PROMPT COMPLETO E DETALHADO PARA GERAR A IMAGEM EM PORTUGUÊS"
}}
"""


def _parse_analysis_response(response_text: str) -> dict:
    """Parse JSON da resposta de análise."""
    text = response_text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("JSON não encontrado")
    
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
    
    return json.loads(text[start_idx:end_idx])


class ImageAnalysisAgent:
    """
    Agente que analisa a questão e gera um prompt de imagem contextualizado.
    
    Fluxo:
    1. Recebe a questão completa
    2. Analisa título, enunciado, resposta correta
    3. Extrai personagens, gênero, cenário, formato
    4. Gera prompt detalhado para imagem
    """
    
    def __init__(self):
        """Inicializa o agente com LLM."""
        self.llm = get_question_llm()
        logger.info("🖼️ ImageAnalysisAgent inicializado")
    
    def analyze_and_generate_prompt(self, question: QuestionSchema) -> str:
        """
        Analisa a questão e gera um prompt inteligente para imagem.
        
        Args:
            question: Questão educacional completa
            
        Returns:
            Prompt detalhado para geração de imagem
        """
        logger.info(f"🔍 Analisando questão: {question.title[:50]}...")
        
        # Extrai alternativa correta
        correct_answer_text = ""
        for alt in question.alternatives:
            if alt.letter == question.correct_answer:
                correct_answer_text = alt.text
                break
        
        try:
            # Cria o prompt de análise
            prompt = PromptTemplate(
                input_variables=["title", "question_statement", "correct_answer_text", "explanation"],
                template=IMAGE_ANALYSIS_PROMPT
            )
            
            chain = prompt | self.llm
            config = get_runnable_config(
                run_name="image-analysis",
                tags=["image", "analysis"]
            )
            
            inputs = {
                "title": question.title,
                "question_statement": question.question_statement,
                "correct_answer_text": correct_answer_text,
                "explanation": question.explanation_question[:500] if question.explanation_question else ""
            }
            
            response = chain.invoke(inputs, config=config)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse da análise
            analysis = _parse_analysis_response(response_text)
            
            # Log da análise
            char_info = analysis.get("character_analysis", {})
            scene_info = analysis.get("scene_analysis", {})
            format_type = analysis.get("format", "cena_unica")
            
            logger.info(
                f"📊 Análise: Personagens={char_info.get('names', [])} | "
                f"Gêneros={char_info.get('genders', {})} | "
                f"Formato={format_type}"
            )
            
            # Retorna o prompt gerado
            image_prompt = analysis.get("image_prompt", "")
            
            if not image_prompt:
                # Fallback: gera prompt básico
                image_prompt = self._generate_fallback_prompt(question, analysis)
            
            return image_prompt
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de imagem: {e}")
            # Fallback para prompt simples
            return self._generate_simple_prompt(question)
    
    def _generate_fallback_prompt(self, question: QuestionSchema, analysis: dict) -> str:
        """Gera prompt de fallback baseado na análise parcial."""
        char_info = analysis.get("character_analysis", {})
        scene_info = analysis.get("scene_analysis", {})
        format_type = analysis.get("format", "cena_unica")
        
        # Determina personagem principal
        names = char_info.get("names", [])
        genders = char_info.get("genders", {})
        
        if names:
            main_char = names[0]
            gender = genders.get(main_char, "neutro")
            char_desc = f"uma {gender}" if gender == "feminino" else f"um {gender}" if gender == "masculino" else "uma criança"
        else:
            char_desc = "uma criança"
        
        location = scene_info.get("location", "um ambiente escolar")
        objects = ", ".join(scene_info.get("key_objects", ["livro", "caderno"]))
        emotion = scene_info.get("main_emotion", "pensativa")
        
        prompt = f"""
Crie uma ilustração educacional de alta qualidade mostrando {char_desc} com expressão {emotion} em {location}.
Elementos na cena: {objects}.
Estilo: Cartoon educativo premium, cores vibrantes.
Formato: {"Tirinha com 3 quadros mostrando progressão" if "tirinha" in format_type else "Cena única centralizada"}.
IDIOMA: Qualquer texto deve estar em PORTUGUÊS.
"""
        return prompt.strip()
    
    def _generate_simple_prompt(self, question: QuestionSchema) -> str:
        """Prompt simples como último fallback."""
        return f"""
Crie uma ilustração educacional para uma questão escolar.
Tema: {question.title}
Estilo: Cartoon educativo premium, cores vibrantes, fundo simples.
A imagem deve auxiliar na interpretação da questão sem revelar a resposta.
IDIOMA: Qualquer texto deve estar em PORTUGUÊS.
"""


# Singleton
_image_agent_instance = None


def get_image_analysis_agent() -> ImageAnalysisAgent:
    """Obtém instância do agente de análise de imagem."""
    global _image_agent_instance
    if _image_agent_instance is None:
        _image_agent_instance = ImageAnalysisAgent()
    return _image_agent_instance
