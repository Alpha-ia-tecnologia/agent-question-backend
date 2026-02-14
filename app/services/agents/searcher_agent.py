"""
Agente de Busca de Textos Reais.

Responsável por buscar textos autênticos de autores reais na internet
para serem usados como base nas questões educacionais.
"""

import logging
from typing import Any

from app.services.agents.state import AgentState
from app.services.text_search_service import TextSearchService, TextSearchError
from app.services.progress_manager import get_current_progress

logger = logging.getLogger(__name__)


def searcher_node(state: AgentState) -> AgentState:
    """
    Nó do Agente Buscador de Textos.
    
    Busca textos reais na internet usando DuckDuckGo quando
    use_real_text=True na query.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com textos encontrados
    """
    query = state["query"]
    
    # Verifica se deve buscar textos reais
    if not query.use_real_text:
        logger.info("⏭️ Busca de textos reais desabilitada, pulando...")
        return {
            **state,
            "real_texts": None
        }
    
    logger.info(
        f"🔎 Agente Buscador - Buscando textos reais | "
        f"Habilidade: {query.skill[:40]}..."
    )
    
    progress = get_current_progress()
    
    try:
        if progress:
            progress.log("searcher", "Initializing DuckDuckGo search engine", "", "🔍")
        service = TextSearchService()
        
        # Busca textos para cada questão solicitada
        if progress:
            progress.log("searcher", f"Building search query for: {query.skill[:50]}", "", "📝")
            progress.log("searcher", f"Searching texts for skill: {query.skill[:40]}", f"Grade: {query.grade}", "🚀")
            progress.log("searcher", "Scanning for authentic texts matching BNCC criteria", "", "🎯")
        texts = service.search_multiple_texts(
            skill=query.skill,
            grade=query.grade,
            count=query.count_questions
        )
        
        if texts:
            # Converte para dicts para serialização
            real_texts = [
                {
                    "text": t.text,
                    "title": t.title,
                    "author": t.author,
                    "source_url": t.source_url,
                    "source_name": t.source_name
                }
                for t in texts
            ]
            
            logger.info(f"✅ Buscador encontrou {len(real_texts)} textos reais")
            if progress:
                for i, t in enumerate(real_texts):
                    title = t.get('title', 'No title')[:60]
                    author = t.get('author', 'Unknown')
                    progress.log("searcher", f"Text {i+1}: \"{title}\"", f"Author: {author}", "📖")
                progress.metric("searcher", "Texts found", len(real_texts), "📚")
            
            return {
                **state,
                "real_texts": real_texts
            }
        else:
            logger.warning("⚠️ Nenhum texto encontrado, geração usará textos internos")
            if progress:
                progress.log("searcher", "No texts found, using internal generation", "", "⚠️")
            return {
                **state,
                "real_texts": None
            }
            
    except TextSearchError as e:
        logger.error(f"❌ Erro no Agente Buscador: {e}")
        logger.info("⚠️ Continuando geração sem textos reais")
        if progress:
            progress.log("searcher", f"Search error: {str(e)[:100]}", "Continuing without real texts", "❌")
        return {
            **state,
            "real_texts": None
        }
    except Exception as e:
        logger.error(f"❌ Erro inesperado no Agente Buscador: {e}")
        logger.info("⚠️ Continuando geração sem textos reais")
        return {
            **state,
            "real_texts": None
        }
