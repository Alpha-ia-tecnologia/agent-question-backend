"""
ImagePromptEngineerAgent - Agente Especializado em Engenharia de Prompts para Imagens.

Este agente analisa detalhadamente:
1. Título da questão
2. Texto-base
3. Enunciado
4. Alternativa correta
5. Explicação da resposta

E gera um prompt otimizado seguindo regras de engenharia de prompt para
garantir que a imagem gerada seja 100% coerente com o conteúdo da questão.
"""

import logging
import json
from typing import Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas.question_schema import QuestionSchema
from app.core.llm_config import get_question_llm, get_runnable_config

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE - Análise e Geração de Prompt de Imagem
# ============================================================================

IMAGE_PROMPT_ENGINEER_TEMPLATE = """Você é um Engenheiro de Prompts especializado em criar prompts precisos para geração de imagens educacionais.

═══════════════════════════════════════════════════════════════════════════════
📋 DADOS COMPLETOS DA QUESTÃO PARA ANÁLISE
═══════════════════════════════════════════════════════════════════════════════

🏷️ TÍTULO: {title}

📖 TEXTO-BASE:
{text}

❓ ENUNCIADO:
{question_statement}

✅ ALTERNATIVA CORRETA: {correct_answer}

📋 TODAS AS ALTERNATIVAS (incluindo incorretas):
{all_alternatives}

💡 EXPLICAÇÃO DA RESPOSTA:
{explanation}

📋 DADOS ESTRUTURADOS PARA A IMAGEM:
{image_data}

═══════════════════════════════════════════════════════════════════════════════
🔴 REGRA CRÍTICA DE COERÊNCIA IMAGEM ↔ ALTERNATIVAS
═══════════════════════════════════════════════════════════════════════════════

TODOS os elementos visuais mencionados nas alternativas (corretas E incorretas)
DEVEM estar presentes na imagem gerada.

Por exemplo:
- Se uma alternativa diz "há um gráfico mostrando X" → o gráfico DEVE existir na imagem
- Se uma alternativa diz "uma pessoa segura um copo" → essa pessoa DEVE aparecer
- Se uma alternativa diz "um logotipo aparece no canto" → o logotipo DEVE estar lá
- Se uma alternativa menciona "cores vibrantes" → o cartaz DEVE ter cores vibrantes

Isso é ESSENCIAL porque o aluno precisa OBSERVAR a imagem para decidir qual
alternativa é correta. Se um elemento mencionado não existir na imagem, a
questão fica incoerente e o aluno não consegue avaliar corretamente.

ANALISE cada alternativa e EXTRAIA todos os elementos visuais mencionados.
Seu prompt de imagem DEVE incluir TODOS eles.

═══════════════════════════════════════════════════════════════════════════════
🎯 SUA TAREFA
═══════════════════════════════════════════════════════════════════════════════

ETAPA 1 - DETECTAR TIPO DE QUESTÃO:

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. GEOMETRIA/MATEMÁTICA TÉCNICA? → Diagrama técnico                        │
│    (triângulo, retângulo, ângulo, área, perímetro, polígono)               │
│                                                                             │
│ 2. FÁBULA/CONTO/TIRINHA? → Sequência de quadrinhos                         │
│    (moral, personagem, expressão facial, último quadrinho, humor)          │
│                                                                             │
│ 3. OUTRA? → Ilustração educacional única                                   │
└─────────────────────────────────────────────────────────────────────────────┘

ETAPA 2 - ANÁLISE DE COERÊNCIA ENUNCIADO ↔ IMAGEM:

⚠️ REGRA CRÍTICA DE COERÊNCIA:
Se o enunciado menciona:
• "último quadrinho" → A imagem DEVE ser uma TIRINHA com múltiplos quadros
• "expressão facial" → O rosto do personagem deve ter EXPRESSÃO CLARA identificável
• "contraste" → A imagem deve mostrar DOIS estados diferentes
• "sequência" → Mostrar MÚLTIPLOS momentos/quadros
• "observe a figura" → A figura deve conter a INFORMAÇÃO necessária

ETAPA 3 - REGRAS POR TIPO:

📚 PARA FÁBULAS/CONTOS (Cigarra e Formiga, Pastor e Lobo, etc.):
• Criar TIRINHA de 3-4 quadros mostrando:
  - Quadro 1-2: Situação inicial (a mentira, a preguiça, o engano)
  - Quadro 3: Conflito/consequência
  - Quadro 4: Desfecho com EXPRESSÃO CLARA do personagem (arrependimento, medo, etc.)
• Incluir BALÃO com a moral da história no último quadro
• Expressões faciais EXAGERADAS e identificáveis
• Estilo cartoon educativo brasileiro

📐 PARA GEOMETRIA 2D (Triângulos, Retângulos):
• Desenho técnico em fundo branco
• Linhas pretas/azuis, estilo limpo
• Mostrar figura ORIGINAL + divisão se houver
• O "?" DEVE marcar EXATAMENTE o lado que a questão pergunta
• Se a questão pede o cateto → "?" no cateto, NÃO na hipotenusa
• Se a questão pede a hipotenusa → "?" na hipotenusa
• Todos os outros lados devem ter seus valores numéricos VISÍVEIS
• Marcar o ângulo de 90° com o quadradinho
• HIPOTENUSA é SEMPRE o lado MAIOR e OPOSTO ao ângulo de 90°

📐 PARA GEOMETRIA ESPACIAL 3D (Pirâmides, Prismas, Cones):
• Desenho técnico 3D em fundo branco, perspectiva clara
• Marcar vértice V (topo) e centro O (base) com letras visíveis
• Marcar altura (h) com linha tracejada de V até O
• 🔴 ARESTA LATERAL vs APÓTEMA DA PIRÂMIDE:
  - Se a questão pede "aresta lateral" → "?" no segmento V→VÉRTICE da base
    (NÃO no segmento V→ponto médio do lado!)
  - Se a questão pede "apótema da pirâmide" → "?" no segmento V→PONTO MÉDIO da aresta da base
  - NUNCA confundir os dois! São segmentos DIFERENTES
• Para pirâmides: mostrar claramente as arestas da base com medidas
• Para cones: marcar raio r, altura h, e geratriz com "?" se pedida
• Todos os valores numéricos conhecidos devem ser VISÍVEIS na imagem

📊 PARA GRÁFICOS DE BARRAS/COLUNAS:
• Eixo Y OBRIGATÓRIO com escala numérica VISÍVEL (0, 2, 4, 6, 8, 10...)
• Eixo X com rótulos claros (nomes das categorias)
• Título do gráfico no topo
• Valores numéricos SOBRE ou DENTRO de cada barra
• Cores distintas para cada barra com legenda

📊 PARA GRÁFICOS DE PIZZA/SETORES:
• Cada setor DEVE ter rótulo/legenda com nome da categoria
• Se questão pede valor absoluto → mostrar PERCENTUAL em cada setor
• Se questão pede proporção → mostrar TAMANHO PROPORCIONAL claro
• NUNCA gerar pizza sem rótulos numéricos (% ou valores)
• Legenda com cores correspondentes

📈 PARA GRÁFICOS DE LINHAS:
• Eixos X e Y com rótulos e escala numérica
• Pontos marcados claramente na linha
• Valores numéricos nos pontos importantes

🖼️ PARA ILUSTRAÇÃO GERAL:
• Cenário contextualizado
• Cores vibrantes, estilo cartoon
• Sem revelar a resposta

🔴 VERIFICAÇÃO OBRIGATÓRIA PARA GRÁFICOS:
• Se "DADOS ESTRUTURADOS" tem N categorias → a imagem DEVE ter EXATAMENTE N setores/barras
• Se o enunciado diz "quatro municípios" → o gráfico DEVE ter 4 setores, não 3
• Todos os percentuais/valores dos DADOS ESTRUTURADOS devem aparecer NA IMAGEM como rótulos
• A soma dos percentuais deve ser EXATAMENTE 100%
• O prompt de imagem deve especificar cada valor individualmente
  Exemplo: "Setor 1: São Luís 35%, Setor 2: Caxias 20%, Setor 3: Imperatriz 25%, Setor 4: Chapadinha 20%"

ETAPA 4 - PROIBIÇÕES ABSOLUTAS:

❌ PROIBIDO:
• Gerar imagem ÚNICA quando enunciado menciona "quadrinho/sequência"
• Gerar imagem com expressão NEUTRA quando enunciado pede "expressão facial"
• Revelar resposta numérica/nome da figura na imagem
• Usar estilo genérico que não transmite a emoção da cena
• Desenhar personagem feliz em cena de consequência negativa
• Omitir a moral em questões de fábula que pedem análise de moral

═══════════════════════════════════════════════════════════════════════════════
📝 FORMATO DA SUA RESPOSTA
═══════════════════════════════════════════════════════════════════════════════

Responda EXATAMENTE neste formato JSON:

{{
    "tipo": "diagrama_tecnico" | "tirinha_fabula" | "ilustracao_educacional",
    "analise": {{
        "figura_principal": "nome da figura ou 'N/A'",
        "tem_divisao": true ou false,
        "personagens": ["lista de nomes/animais"],
        "cenario": "descrição do local",
        "requer_multiplos_quadros": true ou false,
        "expressao_facial_importante": true ou false,
        "moral_da_historia": "texto da moral ou null"
    }},
    "prompt_imagem": "PROMPT COMPLETO E DETALHADO"
}}

EXEMPLO PARA FÁBULA "O Pastor e o Lobo":
{{
    "tipo": "tirinha_fabula",
    "analise": {{
        "personagens": ["pastor/menino", "aldeões", "lobo", "ovelhas"],
        "requer_multiplos_quadros": true,
        "expressao_facial_importante": true,
        "moral_da_historia": "Quem mente perde a confiança"
    }},
    "prompt_imagem": "Tirinha educacional de 4 quadros em estilo cartoon brasileiro. Quadro 1: Menino pastor gritando 'Socorro! Lobo!' com expressão risonha, aldeões correndo assustados. Quadro 2: Pastor rindo sozinho com ovelhas calmas, aldeões irritados ao fundo. Quadro 3: Lobo real atacando ovelhas, pastor com expressão de PAVOR, gritando desesperado. Quadro 4: Aldeões de costas, ignorando, pastor com lágrimas e expressão de ARREPENDIMENTO PROFUNDO, balão com texto 'Quem mente perde a confiança'. Estilo ilustração didática, cores vibrantes, expressões faciais exageradas e claras."
}}

IMPORTANTE: O campo "prompt_imagem" deve ser completo e autocontido.
"""


def _parse_engineer_response(response_text: str) -> Dict[str, Any]:
    """
    Parse a resposta JSON do agente engenheiro de prompts.
    
    Args:
        response_text: Texto da resposta do LLM
        
    Returns:
        Dicionário com a análise e prompt gerado
    """
    text = response_text.strip()
    
    # Remove markdown code blocks se presentes
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove primeira linha (```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    # Encontra o JSON na resposta
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("JSON não encontrado na resposta")
    
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


class ImagePromptEngineerAgent:
    """
    Agente especializado em Engenharia de Prompts para geração de imagens.
    
    Este agente utiliza um LLM para analisar profundamente todos os elementos
    de uma questão educacional e gerar um prompt otimizado que garante
    coerência visual com o conteúdo da questão.
    
    Fluxo:
    1. Recebe a questão completa (título, texto, enunciado, alternativas, explicação)
    2. Analisa usando técnicas de engenharia de prompt
    3. Identifica o tipo de imagem necessária (diagrama técnico vs ilustração)
    4. Gera um prompt detalhado e específico
    5. Retorna o prompt pronto para o modelo de imagem
    """
    
    def __init__(self):
        """Inicializa o agente com o LLM configurado."""
        self.llm = get_question_llm()
        self.prompt_template = PromptTemplate(
            input_variables=["title", "text", "question_statement", "correct_answer", "explanation", "image_data"],
            template=IMAGE_PROMPT_ENGINEER_TEMPLATE
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("🎨 ImagePromptEngineerAgent inicializado")
    
    def _extract_correct_answer(self, question: QuestionSchema) -> str:
        """Extrai o texto da alternativa correta."""
        for alt in question.alternatives:
            if alt.letter == question.correct_answer:
                return f"{alt.letter}) {alt.text}"
        return "N/A"
    
    def analyze_and_generate_prompt(self, question: QuestionSchema) -> str:
        """
        Analisa a questão e gera um prompt otimizado para geração de imagem.
        
        Args:
            question: Questão educacional completa
            
        Returns:
            Prompt otimizado para geração de imagem
        """
        logger.info(f"🔍 Analisando questão: {question.title[:50]}...")
        
        # Prepara os inputs
        image_data_str = "Nenhum dado estruturado disponível."
        if hasattr(question, 'image_data') and question.image_data:
            try:
                image_data_str = json.dumps(question.image_data, ensure_ascii=False, indent=2)
            except Exception:
                image_data_str = str(question.image_data)
        
        # Formata TODAS as alternativas
        all_alts_text = ""
        for alt in question.alternatives:
            is_correct = alt.letter == question.correct_answer
            marker = "✅" if is_correct else "❌"
            distractor_info = f" | Distrator: {alt.distractor[:100]}" if alt.distractor else ""
            all_alts_text += f"{marker} {alt.letter}) {alt.text}{distractor_info}\n"
        
        inputs = {
            "title": question.title,
            "text": question.text[:500] if question.text else "Observe a imagem a seguir.",
            "question_statement": question.question_statement[:500],
            "correct_answer": self._extract_correct_answer(question),
            "all_alternatives": all_alts_text.strip(),
            "explanation": question.explanation_question[:400] if question.explanation_question else "N/A",
            "image_data": image_data_str
        }
        
        try:
            # Executa a análise com o LLM
            config = get_runnable_config(
                run_name="image-prompt-engineer",
                tags=["image", "prompt-engineering"]
            )
            
            response = self.chain.invoke(inputs, config=config)
            
            # Parse da resposta
            result = _parse_engineer_response(response)
            
            # Log da análise
            analise = result.get("analise", {})
            tipo = result.get("tipo", "desconhecido")
            
            logger.info(
                f"📊 Análise concluída: Tipo={tipo} | "
                f"Figura={analise.get('figura_principal', 'N/A')} | "
                f"Divisão={analise.get('tem_divisao', False)}"
            )
            
            # Retorna o prompt gerado
            prompt_imagem = result.get("prompt_imagem", "")
            
            if not prompt_imagem:
                logger.warning("⚠️ Prompt vazio, usando fallback")
                return self._generate_fallback_prompt(question)
            
            return prompt_imagem
            
        except Exception as e:
            logger.error(f"❌ Erro na análise: {e}")
            return self._generate_fallback_prompt(question)
    
    def _generate_fallback_prompt(self, question: QuestionSchema) -> str:
        """Gera um prompt de fallback simples."""
        correct_answer = self._extract_correct_answer(question)
        
        return f"""Crie uma ilustração educacional para esta questão:

Título: {question.title}
Tema: {question.question_statement[:200]}

A ilustração deve:
- Ser clara e educacional
- Usar estilo apropriado (técnico para geometria, cartoon para outros)
- NÃO revelar a resposta
- Estar em português

Resposta correta (para referência, NÃO mostrar na imagem): {correct_answer}
"""
    
    def get_analysis_details(self, question: QuestionSchema) -> Dict[str, Any]:
        """
        Retorna a análise completa da questão, incluindo metadados.
        
        Útil para debugging ou para exibir detalhes da análise ao usuário.
        """
        image_data_str = "Nenhum dado estruturado disponível."
        if hasattr(question, 'image_data') and question.image_data:
            try:
                image_data_str = json.dumps(question.image_data, ensure_ascii=False, indent=2)
            except Exception:
                image_data_str = str(question.image_data)
        
        inputs = {
            "title": question.title,
            "text": question.text[:500] if question.text else "Observe a imagem a seguir.",
            "question_statement": question.question_statement[:500],
            "correct_answer": self._extract_correct_answer(question),
            "explanation": question.explanation_question[:400] if question.explanation_question else "N/A",
            "image_data": image_data_str
        }
        
        try:
            config = get_runnable_config(
                run_name="image-prompt-engineer-analysis",
                tags=["image", "analysis", "debug"]
            )
            
            response = self.chain.invoke(inputs, config=config)
            return _parse_engineer_response(response)
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter detalhes: {e}")
            return {"error": str(e)}


# ============================================================================
# Singleton Instance
# ============================================================================

_agent_instance: Optional[ImagePromptEngineerAgent] = None


def get_image_prompt_engineer_agent() -> ImagePromptEngineerAgent:
    """
    Obtém a instância singleton do ImagePromptEngineerAgent.
    
    Returns:
        Instância do agente
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ImagePromptEngineerAgent()
    return _agent_instance
