"""
Estado compartilhado para o sistema multi-agente LangGraph.

Define a estrutura de dados que é passada entre os agentes
durante a geração de questões com garantia de qualidade.
"""

from typing import TypedDict, List, Optional, Any
from app.schemas.question_schema import QuestionSchema
from app.schemas.request_body_agent import RequestBodyAgentQuestion


class AgentState(TypedDict):
    """
    Estado compartilhado entre todos os agentes do grafo.
    
    Attributes:
        query: Parâmetros originais da requisição
        questions: Lista de questões geradas
        real_texts: Textos reais encontrados na busca (se use_real_text=True)
        revision_feedback: Feedback do revisor para regeneração
        quality_score: Pontuação de qualidade (0.0 - 1.0)
        retry_count: Número de tentativas de regeneração
        error: Mensagem de erro se houver falha
    """
    
    # Inputs originais
    query: RequestBodyAgentQuestion
    
    # Busca de textos
    real_texts: Optional[List[dict]]  # Textos encontrados na internet
    
    # Geração
    questions: List[dict]  # Lista de dicts (questões)
    
    # Revisão e Qualidade
    revision_feedback: Optional[str]
    quality_score: Optional[float]
    retry_count: int
    
    # Controle de erros
    error: Optional[str]
    
    # Imagens (pipeline integrado)
    image_results: Optional[List[dict]]  # [{question_index, image_base64, validation_status, attempts}]
    image_retry_count: int

