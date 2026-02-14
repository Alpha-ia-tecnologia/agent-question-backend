"""
Router de Qualidade.

Decide o próximo passo no grafo baseado na pontuação
de qualidade e número de tentativas de regeneração.
"""

import logging
from typing import Literal

from app.services.agents.state import AgentState

logger = logging.getLogger(__name__)

# Configurações do loop de qualidade
QUALITY_THRESHOLD = 0.7  # Score mínimo para aprovação
MAX_RETRIES = 3  # Máximo de tentativas de regeneração


def quality_router(state: AgentState) -> Literal["regenerate", "finish"]:
    """
    Decide o próximo passo baseado na qualidade.
    
    Lógica:
    - Se score < 0.7 E retry < 3: regenerar
    - Caso contrário: finalizar
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        "regenerate" para voltar ao gerador ou "finish" para encerrar
    """
    score = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)
    error = state.get("error")
    
    # Se houver erro crítico, não tentar mais
    if error and retry_count >= MAX_RETRIES:
        logger.warning(f"🛑 Atingido limite de tentativas ({MAX_RETRIES}) com erro")
        return "finish"
    
    # Verifica se precisa regenerar
    if score < QUALITY_THRESHOLD and retry_count < MAX_RETRIES:
        logger.info(
            f"🔄 Qualidade insuficiente ({score:.2f} < {QUALITY_THRESHOLD}) - "
            f"Regenerando (tentativa {retry_count + 1}/{MAX_RETRIES})"
        )
        return "regenerate"
    
    # Aprovado ou atingiu limite
    if score >= QUALITY_THRESHOLD:
        logger.info(f"✅ Qualidade aprovada: {score:.2f}")
    else:
        logger.warning(
            f"⚠️ Limite de tentativas atingido. "
            f"Score final: {score:.2f}"
        )
    
    return "finish"
