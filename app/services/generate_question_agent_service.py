"""
Serviço de Geração de Questões Educacionais.

Utiliza LLM (DeepSeek/OpenAI/Gemini) para gerar questões de múltipla escolha
baseadas em habilidades e níveis de proficiência específicos.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
import json
import re

from app.schemas.question_schema import QuestionListSchema
from app.schemas.request_body_agent import RequestBodyAgentQuestion
from app.enums.agente_prompt_template import AgentPromptTemplates, get_prompt
from app.core.llm_config import (
    get_question_llm,
    get_runnable_config,
    QuestionGenerationError,
    RETRY_CONFIG
)
from app.services.text_search_service import TextSearchService, TextSearchError

# Logger para este módulo
logger = logging.getLogger(__name__)


class GenerateQuestionAgentService:
    """
    Serviço para geração de questões educacionais usando IA.
    
    Utiliza o modelo Gemini para gerar questões
    estruturadas com base em parâmetros de habilidade e proficiência.
    
    Attributes:
        llm: Instância do modelo de linguagem configurado
        _chain_cache: Cache de chains criadas para cada template
        text_search_service: Serviço para busca de textos reais
    """
    
    def __init__(self):
        """Inicializa o serviço com configuração centralizada do LLM."""
        self.llm = get_question_llm()
        self._chain_cache: dict[str, RunnableSequence] = {}
        
        logger.info("GenerateQuestionAgentService inicializado")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Faz parsing manual de JSON da resposta do LLM.
        
        Args:
            response_text: Texto da resposta do LLM
            
        Returns:
            Dicionário com dados parseados
        """
        # Remove possíveis blocos de código markdown
        text = response_text.strip()
        
        # Remove ```json e ``` se presentes
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove primeira linha (```json ou similar)
            lines = lines[1:]
            # Remove última linha se for ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Encontra o início do JSON
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("Nenhum objeto JSON encontrado na resposta")
        
        # Extrai apenas o primeiro objeto JSON completo usando contador de chaves
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
    
    def _get_or_create_chain(
        self, 
        template: str,
        input_variables: list[str] = None
    ) -> RunnableSequence:
        """
        Obtém ou cria uma chain para o template especificado.
        
        Utiliza cache para evitar recriar chains para o mesmo template.
        
        Args:
            template: Template do prompt
            input_variables: Lista de variáveis do template
            
        Returns:
            Chain configurada (retorna texto bruto para parsing manual)
        """
        template_hash = hash(template)
        
        if template_hash not in self._chain_cache:
            # Variáveis padrão
            default_vars = [
                "count_questions",
                "count_alternatives", 
                "skill",
                "proficiency_level",
                "grade",
                "model_evaluation_type",
                "image_dependency_instruction"
            ]
            
            prompt = PromptTemplate(
                input_variables=input_variables or default_vars,
                template=template
            )
            
            # Usa LLM direto sem structured output para compatibilidade com DeepSeek
            self._chain_cache[template_hash] = prompt | self.llm
            logger.debug(f"Chain criada e cacheada (hash: {template_hash})")
        
        return self._chain_cache[template_hash]
    
    @retry(
        stop=stop_after_attempt(RETRY_CONFIG["stop_after_attempt"]),
        wait=wait_exponential(
            multiplier=RETRY_CONFIG["wait_exponential_multiplier"],
            min=RETRY_CONFIG["wait_exponential_min"],
            max=RETRY_CONFIG["wait_exponential_max"]
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _invoke_chain(
        self,
        chain: RunnableSequence,
        inputs: dict,
        run_name: str
    ) -> QuestionListSchema:
        """
        Invoca a chain com retry automático.
        
        Args:
            chain: Chain a ser executada
            inputs: Dicionário de inputs para o prompt
            run_name: Nome da execução para rastreamento
            
        Returns:
            Schema de questões geradas
            
        Raises:
            QuestionGenerationError: Se falhar após todas as tentativas
        """
        config = get_runnable_config(
            run_name=run_name,
            tags=["question-generation"]
        )
        
        # Invoca e obtém resposta bruta
        response = chain.invoke(inputs, config=config)
        
        # Extrai conteúdo do AIMessage
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse manual do JSON
        try:
            parsed_data = self._parse_json_response(response_text)
            return QuestionListSchema(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Erro ao parsear resposta JSON: {e}")
            logger.debug(f"Resposta recebida: {response_text[:500]}...")
            raise QuestionGenerationError(f"Resposta do LLM não é JSON válido: {e}") from e

    def send_to_llm(
        self,
        template: str,
        query: RequestBodyAgentQuestion,
        extra_inputs: dict = None
    ) -> QuestionListSchema:
        """
        Envia requisição para o LLM gerar questões.
        
        Args:
            template: Template do prompt a ser usado
            query: Parâmetros da requisição (quantidade, habilidade, etc.)
            extra_inputs: Inputs adicionais para o template (ex: texto real)
            
        Returns:
            Lista de questões geradas estruturadas
            
        Raises:
            QuestionGenerationError: Se ocorrer erro na geração
        """
        logger.info(
            f"Gerando {query.count_questions} questões - "
            f"Habilidade: {query.skill[:30]}... | "
            f"Nível: {query.proficiency_level} | "
            f"Modelo: {query.model_evaluation_type.value}"
        )
        
        # Determina as variáveis do template
        input_variables = [
            "count_questions",
            "count_alternatives", 
            "skill",
            "proficiency_level",
            "grade",
            "model_evaluation_type"
        ]
        
        # Adiciona variáveis extras se houver
        if extra_inputs:
            input_variables.extend(extra_inputs.keys())
        
        chain = self._get_or_create_chain(template, input_variables)
        
        # Mapeia a dependência de imagem para instruções
        image_instructions = {
            "none": """⚠️ QUESTÃO SEM IMAGEM - REGRAS ABSOLUTAS ⚠️

❌ PROIBIDO GERAR QUESTÕES QUE DEPENDAM DE IMAGEM:
   - NÃO use "[IMAGEM: ...]" no texto
   - NÃO mencione "observe a figura", "observe a imagem", "observe a tirinha"
   - NÃO faça questões sobre gráficos, tabelas visuais, charges ou tirinhas
   - NÃO referencie elementos visuais que o aluno precisaria ver

✅ OBRIGATÓRIO:
   - As questões devem ser 100% resolvidas apenas com LEITURA DO TEXTO
   - Todo o conteúdo necessário deve estar ESCRITO no texto-base
   - Se precisar de dados numéricos, ESCREVA-OS no texto (não em gráficos)
   - O campo "text" deve conter o texto completo necessário para resolver a questão

🎯 PRINCÍPIO: O aluno resolve a questão APENAS LENDO, sem precisar de nenhuma imagem.""",
            "optional": "As questões podem opcionalmente ter imagens ilustrativas decorativas, mas a resolução NÃO deve depender da imagem. O aluno deve conseguir responder apenas com o texto.",
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

🚫🚫🚫 REGRA DE OURO: JAMAIS DESCREVA A IMAGEM 🚫🚫🚫
   A imagem será gerada SEPARADAMENTE por outra IA.
   Você NÃO SABE como a imagem será. NÃO INVENTE descrições.

   ❌ ERRADO (descreve a imagem no texto ou enunciado):
      - "O gráfico mostra que 40% dos alunos preferem futebol"
      - "Na imagem, há um triângulo retângulo com catetos de 3cm e 4cm"
      - "A tirinha apresenta um personagem surpreso"
      - "O mapa indica a região Nordeste do Brasil"
      - "A tabela mostra os dados de vendas de janeiro a março"
      - "Observe o gráfico de barras que apresenta a quantidade de livros lidos"
      - "[IMAGEM: gráfico de barras com dados de...]"

   ✅ CORRETO (apenas referencia sem descrever):
      - "Observe o gráfico a seguir." (SEM dizer o que o gráfico mostra)
      - "Observe a figura a seguir." (SEM descrever a figura)
      - "Analise a imagem e responda." (SEM descrever o conteúdo)

🚫 TAMBÉM PROIBIDO:
   - NÃO coloque os dados do problema no campo "text" (eles devem estar NA IMAGEM)
   - NÃO escreva tabelas/gráficos em formato texto — eles serão visuais
   - NÃO crie questões que possam ser respondidas sem ver a imagem
   - NÃO use colchetes [IMAGEM: ...] ou [FIGURA: ...]

📐 COERÊNCIA MATEMÁTICA:
   - Valores numéricos devem ser matematicamente consistentes
   - A RESPOSTA deve exigir CÁLCULO ou INFERÊNCIA, não observação direta
   - HIPOTENUSA é SEMPRE o maior lado do triângulo retângulo

🎯 TESTE FINAL: Leia sua questão SEM a imagem. Se conseguir responder, REFAÇA!
   O aluno DEVE OLHAR a imagem + RACIOCINAR para responder."""
        }
        
        inputs = {
            "count_questions": query.count_questions,
            "count_alternatives": query.count_alternatives,
            "skill": query.skill,
            "proficiency_level": query.proficiency_level,
            "grade": query.grade,
            "model_evaluation_type": query.model_evaluation_type.value,
            "image_dependency_instruction": image_instructions.get(query.image_dependency, image_instructions["none"])
        }
        
        # Mescla inputs extras
        if extra_inputs:
            inputs.update(extra_inputs)
        
        try:
            response = self._invoke_chain(
                chain=chain,
                inputs=inputs,
                run_name=f"question-gen-{query.grade}-{query.count_questions}q"
            )
            
            logger.info(
                f"✅ Geração concluída: {len(response.questions)} questões"
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar questões: {e}")
            raise QuestionGenerationError(
                f"Falha ao gerar questões após múltiplas tentativas: {e}"
            ) from e
    
    def generate_with_real_text(
        self,
        query: RequestBodyAgentQuestion
    ) -> QuestionListSchema:
        """
        Gera questões usando conhecimento interno do LLM para recuperar textos reais.
        
        Args:
            query: Parâmetros da requisição
            
        Returns:
            Lista de questões geradas com textos autênticos recuperados da memória do LLM
        """
        logger.info("🔍 Modo: Recuperação de Textos Reais via Conhecimento da IA (Sem busca externa)")
        
        # Usa DIRETAMENTE o template de retrieval da IA
        # A IA será responsável por buscar em sua base de treinamento textos que atendam à habilidade
        template = get_prompt(AgentPromptTemplates.AI_RETRIEVAL_PT_TEMPLATE)
        
        # Adicionamos instrução extra para garantir variabilidade
        extra_inputs = {
            "variability_instruction": "IMPORTANTE: Gere questões TOTALMENTE DISTINTAS. Se houver mais de uma questão, use TEXTOS DIFERENTES para cada uma. DIVERSIFIQUE OS AUTORES DA LITERATURA BRASILEIRA."
        }
        
        return self.send_to_llm(template, query, extra_inputs)
    
    def generate_questions(
        self,
        query: RequestBodyAgentQuestion
    ) -> QuestionListSchema:
        """
        Gera questões usando pipeline multi-agente LangGraph com garantia de qualidade.
        
        O fluxo LangGraph é:
        1. Router → decide se busca textos reais
        2. Searcher → busca textos (se use_real_text=True)
        3. Agente Gerador → cria questões padrão SAEB
        4. Agente Revisor → avalia qualidade BNCC/SAEB
        5. Router de Qualidade → regenera se score < 0.7, máx 3 tentativas
        
        Args:
            query: Parâmetros da requisição
            
        Returns:
            Lista de questões geradas e validadas
            
        Raises:
            QuestionGenerationError: Se ocorrer erro na geração
        """
        from app.services.langgraph_orchestrator import get_orchestrator
        
        logger.info("🚀 Usando LangGraph Multi-Agent para geração (padrão SAEB)")
        
        try:
            orchestrator = get_orchestrator()
            return orchestrator.generate(query)
        except Exception as e:
            logger.error(f"❌ Erro na geração LangGraph: {e}")
            raise QuestionGenerationError(f"Falha na geração de questões: {e}") from e