"""
Serviço de Geração de Imagens Educacionais.

Utiliza a API Google Gemini 3 Pro Image Preview para gerar
imagens educacionais de alta qualidade.

Modelo: gemini-3-pro-image-preview (Nano Banana Pro)
"""

from google import genai
from google.genai import types
import logging
import os
import base64

from app.enums.agente_prompt_template import AgentPromptTemplates, get_prompt
from app.schemas.image_response import ImageResponse
from app.schemas.question_schema import QuestionSchema
from app.core.llm_config import ImageGenerationError

# Logger para este módulo
logger = logging.getLogger(__name__)

# Chave Google GenAI
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# Configuração do modelo de imagem
IMAGE_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro
IMAGE_ASPECT_RATIO = "1:1"


class GenerateImageAgentService:
    """
    Serviço para geração de imagens educacionais usando Google Gemini 3 Pro Image (Nano Banana Pro).
    """
    
    def __init__(self):
        """Inicializa o serviço com a API Google GenAI."""
        api_key = os.getenv("GOOGLE_GENAI_API_KEY") or GOOGLE_GENAI_API_KEY
        
        # Cliente Google GenAI
        self.client = genai.Client(api_key=api_key)
        
        self.prompt_template = get_prompt(AgentPromptTemplates.GENERATE_IMAGE_TEMPLATE)
        self.model = IMAGE_MODEL
        self.aspect_ratio = IMAGE_ASPECT_RATIO
        
        logger.info(f"🎨 GenerateImageAgentService inicializado com {self.model} (Nano Banana Pro)")
    
    
    def _build_image_prompt(self, question: QuestionSchema) -> str:
        """
        Constrói um prompt otimizado para geração de imagem educacional.
        
        Args:
            question: Questão educacional para ilustrar
            
        Returns:
            Prompt otimizado para Gemini 2.5 Flash Image
        """
        # Extrai a alternativa correta
        correct_alt_text = ""
        for alt in question.alternatives:
            if alt.letter == question.correct_answer:
                correct_alt_text = alt.text
                break
        
        # Detecta se é uma questão de geometria/matemática técnica
        geometry_keywords = [
            'triângulo', 'triangulo', 'quadrado', 'retângulo', 'retangulo',
            'pentágono', 'pentagono', 'hexágono', 'hexagono', 'círculo', 'circulo',
            'diagonal', 'ângulo', 'angulo', 'vértice', 'vertice', 'lado',
            'paralelo', 'perpendicular', 'bissetriz', 'mediana', 'altura',
            'hipotenusa', 'cateto', 'pitágoras', 'pitagoras',
            'figura geométrica', 'figura geometrica', 'polígono', 'poligono',
            'área', 'area', 'perímetro', 'perimetro', 'graus', 'radianos',
            'segmento', 'reta', 'ponto', 'intersecta', 'paralela', 'perpendicular'
        ]
        
        question_text = f"{question.title} {question.question_statement} {question.text}".lower()
        is_geometry = any(kw in question_text for kw in geometry_keywords)
        
        if is_geometry:
            # Prompt para DIAGRAMAS TÉCNICOS de geometria
            # Inclui análise completa: enunciado, resposta correta, explicação
            explanation_snippet = ""
            if hasattr(question, 'explanation_question') and question.explanation_question:
                explanation_snippet = question.explanation_question[:300]
            
            prompt = f"""Você é um especialista em criar diagramas geométricos para questões educacionais.

TAREFA: Analise TODOS os elementos abaixo e crie um DIAGRAMA TÉCNICO que seja 100% coerente.

═══════════════════════════════════════════════════════════════
📋 ANÁLISE COMPLETA DA QUESTÃO
═══════════════════════════════════════════════════════════════

1️⃣ TÍTULO: {question.title}

2️⃣ TEXTO-BASE: {question.text[:300] if question.text else "Observe a imagem a seguir."}

3️⃣ ENUNCIADO: {question.question_statement[:400]}

4️⃣ ALTERNATIVA CORRETA: "{correct_alt_text}"

5️⃣ EXPLICAÇÃO DA RESPOSTA: {explanation_snippet}

═══════════════════════════════════════════════════════════════
⚠️⚠️⚠️ REGRAS CRÍTICAS DE COERÊNCIA ⚠️⚠️⚠️
═══════════════════════════════════════════════════════════════

REGRA 1 - LEIA O ENUNCIADO COM ATENÇÃO:
Se o texto diz "terreno RETANGULAR" → desenhe um RETÂNGULO
Se o texto diz "triângulo" → desenhe um TRIÂNGULO
A FIGURA MENCIONADA NO TEXTO deve aparecer na imagem!

REGRA 2 - CONTEXTO COMPLETO PARA DIVISÕES:
Se o enunciado fala de algo sendo "DIVIDIDO", "CORTADO" ou "SEPARADO":
→ Mostre a FIGURA ORIGINAL COMPLETA
→ Mostre a LINHA DE DIVISÃO (tracejada se mencionada)
→ DESTAQUE a parte resultante mencionada

Exemplo: "Terreno retangular dividido" = RETÂNGULO + linha de divisão + triângulo destacado

REGRA 3 - NÃO MOSTRE APENAS O RESULTADO:
Se a questão fala de uma transformação, mostre o ANTES e o RESULTADO juntos.
Não desenhe apenas a figura resultante sem contexto.

═══════════════════════════════════════════════════════════════
🎯 O QUE VOCÊ DEVE DESENHAR
═══════════════════════════════════════════════════════════════

✅ A figura geométrica EXATA mencionada no enunciado (retângulo, triângulo, etc.)
✅ Todas as linhas de divisão mencionadas (tracejadas se especificado)
✅ Destaque visual (cinza/azul claro) na região de interesse
✅ Vértices com letras maiúsculas (A, B, C, D...)
✅ Medidas conhecidas quando fornecidas no problema
✅ Título da imagem igual ao título da questão (completo, sem cortar)

═══════════════════════════════════════════════════════════════
❌ PROIBIDO ABSOLUTAMENTE
═══════════════════════════════════════════════════════════════
• NÃO desenhe uma figura DIFERENTE da mencionada no enunciado
• NÃO mostre apenas o resultado SEM o contexto original
• NÃO corte/trunque o título da imagem
• NÃO mostre valores que são a RESPOSTA
• NÃO use cores vibrantes, gradientes ou elementos decorativos

ESTILO: Diagrama técnico de livro didático - fundo branco, linhas pretas/azuis, limpo e profissional.

Crie agora o diagrama COMPLETO e COERENTE com o enunciado."""
        else:
            # Obtém a explicação para contexto
            explanation_snippet = ""
            if hasattr(question, 'explanation_question') and question.explanation_question:
                explanation_snippet = question.explanation_question[:250]
            
            # Detecta se a resposta é uma figura geométrica específica
            geometric_figures = {
                'trapézio': 'um TRAPÉZIO (4 lados, exatamente 2 paralelos)',
                'trapezio': 'um TRAPÉZIO (4 lados, exatamente 2 paralelos)',
                'losango': 'um LOSANGO (4 lados iguais, ângulos NÃO são 90°)',
                'quadrado': 'um QUADRADO (4 lados iguais, 4 ângulos de 90°)',
                'retângulo': 'um RETÂNGULO (lados opostos iguais, 4 ângulos de 90°)',
                'retangulo': 'um RETÂNGULO (lados opostos iguais, 4 ângulos de 90°)',
                'triângulo': 'um TRIÂNGULO (3 lados)',
                'triangulo': 'um TRIÂNGULO (3 lados)',
                'pentágono': 'um PENTÁGONO (5 lados)',
                'pentagono': 'um PENTÁGONO (5 lados)',
                'hexágono': 'um HEXÁGONO (6 lados)',
                'hexagono': 'um HEXÁGONO (6 lados)',
                'círculo': 'um CÍRCULO (sem lados, curvo)',
                'circulo': 'um CÍRCULO (sem lados, curvo)',
            }
            
            figure_instruction = ""
            correct_lower = correct_alt_text.lower()
            for fig_name, fig_desc in geometric_figures.items():
                if fig_name in correct_lower:
                    figure_instruction = f"""
⚠️⚠️⚠️ REGRA CRÍTICA DE COERÊNCIA GEOMÉTRICA ⚠️⚠️⚠️
A resposta correta é "{correct_alt_text}".
Se a imagem mostra alguém DESENHANDO ou uma FIGURA sendo criada,
essa figura DEVE ser {fig_desc}.
NÃO desenhe outra figura diferente!
"""
                    break
            
            # Formata TODAS as alternativas
            all_alts_text = ""
            for alt in question.alternatives:
                is_correct = alt.letter == question.correct_answer
                marker = "✅" if is_correct else "❌"
                all_alts_text += f"{marker} {alt.letter}) {alt.text}\n"
            
            prompt = f"""Você é um especialista em criar ilustrações educacionais para questões de provas.

TAREFA: Analise TODOS os elementos abaixo e crie uma ILUSTRAÇÃO que seja 100% coerente.

═══════════════════════════════════════════════════════════════
📋 ANÁLISE COMPLETA DA QUESTÃO
═══════════════════════════════════════════════════════════════

1️⃣ TÍTULO: {question.title}

2️⃣ TEXTO-BASE: {question.text[:300] if question.text else "(sem texto-base)"}

3️⃣ ENUNCIADO: {question.question_statement[:350]}

4️⃣ ALTERNATIVA CORRETA: "{correct_alt_text}"
{figure_instruction}
5️⃣ EXPLICAÇÃO: {explanation_snippet}

6️⃣ TODAS AS ALTERNATIVAS:
{all_alts_text}

═══════════════════════════════════════════════════════════════
🔴 REGRA CRÍTICA: COERÊNCIA IMAGEM ↔ ALTERNATIVAS
═══════════════════════════════════════════════════════════════

ATENÇÃO: A imagem deve tornar APENAS a alternativa correta identificável.

✅ REGRA 1 - ALTERNATIVA CORRETA:
Elementos visuais da alternativa CORRETA DEVEM existir na imagem.

❌ REGRA 2 - ALTERNATIVAS INCORRETAS:
Elementos visuais das alternativas INCORRETAS NÃO devem existir na imagem.
Se uma incorreta menciona "logotipo no canto" → NÃO coloque logotipo.
Se uma incorreta menciona "pessoa com copo" → NÃO coloque pessoa com copo.

MOTIVO: Se todos os elementos de todas as alternativas existirem na imagem,
qualquer alternativa pode parecer correta, tornando a questão ambígua.

Analise CADA alternativa e identifique os elementos visuais:
- Correta → INCLUA na imagem
- Incorretas → EXCLUA da imagem

═══════════════════════════════════════════════════════════════
🎯 O QUE VOCÊ DEVE FAZER
═══════════════════════════════════════════════════════════════

ANTES DE DESENHAR, ANALISE:
• Qual é o TEMA/CENÁRIO principal da questão?
• Existem PERSONAGENS mencionados? (extraia nomes e gêneros)
• Quais elementos visuais são referenciados nas alternativas?
• Se a resposta é uma FIGURA GEOMÉTRICA, essa figura deve aparecer na imagem!

AGORA DESENHE:
✅ Ilustração cartoon educativo premium
✅ Cores vibrantes e atrativas
✅ TODOS os elementos visuais mencionados nas alternativas
✅ Personagens com expressões compatíveis com o contexto
✅ Qualquer texto na imagem DEVE ser em PORTUGUÊS

═══════════════════════════════════════════════════════════════
❌ PROIBIDO ABSOLUTAMENTE
═══════════════════════════════════════════════════════════════
• NÃO desenhe uma figura geométrica DIFERENTE da resposta correta
• NÃO escreva o NOME da figura na imagem (isso revelaria a resposta)
• NÃO adicione banners ou textos decorativos
• NÃO inclua emojis ou mascotes desnecessários
• NÃO omita elementos visuais mencionados nas alternativas

Crie agora uma ilustração educacional de alta qualidade e COERENTE com TODAS as alternativas."""
        
        return prompt.strip()
    
    def generate_image(self, question: QuestionSchema) -> ImageResponse:
        """
        Gera uma imagem ilustrativa para a questão.
        
        Usa Imagen 3.0 (Nano Banana Pro) como modelo principal,
        com fallback para Gemini Flash se necessário.
        
        Args:
            question: Questão educacional para ilustrar
            
        Returns:
            ImageResponse contendo a imagem em Base64
            
        Raises:
            ImageGenerationError: Se ocorrer erro na geração
        """
        # Tenta usar o agente de engenharia de prompt para análise mais inteligente
        try:
            from app.services.agents.image_prompt_engineer_agent import get_image_prompt_engineer_agent
            
            logger.info("🤖 Usando ImagePromptEngineerAgent para análise inteligente...")
            agent = get_image_prompt_engineer_agent()
            prompt = agent.analyze_and_generate_prompt(question)
            logger.info(f"📝 Prompt gerado pelo agente: {prompt[:150]}...")
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback para prompt local: {e}")
            prompt = self._build_image_prompt(question)
            logger.info(f"📝 Prompt gerado localmente: {prompt[:150]}...")
        
        # Gera com Gemini 2.5 Flash Image (Nano Banana)
        try:
            logger.info(f"🎨 Gerando imagem com {self.model} (Nano Banana)...")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self.aspect_ratio,
                    ),
                ),
            )
            
            # Extrai a imagem da resposta
            for part in response.parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    logger.info(f"✅ Imagem gerada! ({len(image_base64)} chars)")
                    return ImageResponse(image_base64=image_base64)
            
            raise ImageGenerationError("Resposta não contém dados de imagem.")
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar imagem: {e}")
            raise ImageGenerationError(f"Falha ao gerar imagem: {e}") from e

    def generate_image_with_instructions(
        self, 
        question: QuestionSchema, 
        custom_instructions: str,
        existing_image_base64: str = None
    ) -> ImageResponse:
        """
        Edita ou regenera uma imagem com instruções personalizadas de correção.
        
        Se existing_image_base64 for fornecido, edita a imagem existente.
        Caso contrário, gera uma nova imagem do zero.
        
        Args:
            question: Questão educacional para ilustrar
            custom_instructions: Instruções do usuário para correção/melhoria
            existing_image_base64: Imagem atual em base64 (para edição)
            
        Returns:
            ImageResponse contendo a imagem em Base64
        """
        try:
            if existing_image_base64:
                # ===== MODO EDIÇÃO: envia imagem existente + instruções =====
                logger.info(f"✏️ Editando imagem existente com instruções usando {self.model}...")
                
                # Decodifica a imagem base64 para bytes
                image_bytes_input = base64.b64decode(existing_image_base64)
                
                # Cria o Part com os bytes da imagem
                image_part = types.Part.from_bytes(
                    data=image_bytes_input,
                    mime_type="image/png"
                )
                
                edit_prompt = f"""Edite esta imagem aplicando as seguintes correções:

{custom_instructions}

REGRAS IMPORTANTES:
- Mantenha o estilo visual e a estética da imagem original
- Aplique APENAS as correções solicitadas
- NÃO altere elementos que não foram mencionados nas instruções
- NÃO revele a resposta da questão na imagem
- Mantenha textos em português
- Mantenha o estilo educativo da imagem"""
                
                # Envia imagem + prompt de edição ao Gemini
                contents = [image_part, edit_prompt]
            else:
                # ===== MODO GERAÇÃO: cria imagem nova do zero =====
                logger.info(f"🔄 Gerando nova imagem com instruções usando {self.model}...")
                
                base_prompt = self._build_image_prompt(question)
                contents = f"""{base_prompt}

INSTRUÇÕES ADICIONAIS DO USUÁRIO (PRIORIDADE MÁXIMA):
{custom_instructions}

LEMBRE-SE: Aplique as correções solicitadas pelo usuário mantendo as regras básicas (sem resposta na imagem, português, estilo educativo)."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self.aspect_ratio,
                    ),
                ),
            )
            
            for part in response.parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    mode = "editada" if existing_image_base64 else "regenerada"
                    logger.info(f"✅ Imagem {mode} com sucesso!")
                    return ImageResponse(image_base64=image_base64)
            
            raise ImageGenerationError("Resposta não contém dados de imagem.")
                
        except Exception as e:
            logger.error(f"❌ Falha na regeneração/edição: {e}")
            raise ImageGenerationError(f"Falha ao processar imagem: {e}") from e

    def set_aspect_ratio(self, aspect_ratio: str) -> None:
        """
        Altera a proporção das imagens geradas.
        
        Args:
            aspect_ratio: "1:1", "9:16", "16:9", "3:4", "4:3"
        """
        valid_ratios = ["1:1", "9:16", "16:9", "3:4", "4:3"]
        if aspect_ratio not in valid_ratios:
            raise ValueError(f"Proporção inválida. Use: {valid_ratios}")
        
        self.aspect_ratio = aspect_ratio
        logger.info(f"🔧 Proporção alterada para: {self.aspect_ratio}")


# Singleton do serviço
_image_service_instance = None

def get_image_service() -> GenerateImageAgentService:
    """Retorna instância singleton do serviço de geração de imagens."""
    global _image_service_instance
    if _image_service_instance is None:
        _image_service_instance = GenerateImageAgentService()
    return _image_service_instance
