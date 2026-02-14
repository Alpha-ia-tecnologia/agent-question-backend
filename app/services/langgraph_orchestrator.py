"""
Orquestrador LangGraph para Geração de Questões.

Implementa o grafo de agentes para geração de questões
educacionais com garantia de qualidade.

Fluxo:
1. Router inicial → decide se busca textos ou vai direto
2. Searcher → busca textos reais (se use_real_text=True)
3. Gerador → cria questões
4. Revisor → avalia qualidade
5. Quality Router → decide: regenerar ou finalizar
"""

import logging
import os
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, START, END

from app.services.agents.state import AgentState
from app.services.agents.generator_agent import generator_node
from app.services.agents.reviewer_agent import reviewer_node
from app.services.agents.searcher_agent import searcher_node
from app.services.agents.quality_router import quality_router
from app.services.agents.image_pipeline_nodes import (
    image_router_decision,
    image_generator_node,
    image_validator_node,
    image_quality_router,
    increment_image_retry,
)
from app.schemas.question_schema import QuestionListSchema, QuestionSchema
from app.schemas.request_body_agent import RequestBodyAgentQuestion

logger = logging.getLogger(__name__)


def search_router(state: AgentState) -> Literal["searcher", "generator"]:
    """
    Router inicial que decide se deve buscar textos reais.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        "searcher" se use_real_text=True, senão "generator"
    """
    query = state["query"]
    
    if query.use_real_text:
        logger.info("🔀 Roteando para busca de textos reais")
        return "searcher"
    else:
        logger.info("🔀 Roteando direto para geração")
        return "generator"


def create_question_graph() -> StateGraph:
    """
    Cria e compila o grafo de geração de questões.
    
    Arquitetura:
    
        START → [search_router] → searcher → generator → reviewer → [quality_router]
                       ↓             ↑                         ↓
                  generator ←←←←←←←←←←← regenerate ←← ← ← ← ←
                                                               ↓
                                                    [image_router_decision]
                                                      ↓              ↓
                                              image_generator    finish → END
                                                    ↓
                                              image_validator
                                                    ↓
                                             [image_quality_router]
                                               ↓              ↓
                                         image_retry_inc   finish → END
                                               ↓
                                         image_generator (retry)
    
    Returns:
        Grafo compilado pronto para execução
    """
    logger.info("📊 Criando grafo LangGraph para geração de questões")
    
    # Cria o grafo com o tipo de estado
    graph = StateGraph(AgentState)
    
    # ── Nós de texto (existentes) ──
    graph.add_node("searcher", searcher_node)
    graph.add_node("generator", generator_node)
    graph.add_node("reviewer", reviewer_node)
    
    # ── Nós de imagem (novos) ──
    graph.add_node("image_generator", image_generator_node)
    graph.add_node("image_validator", image_validator_node)
    graph.add_node("image_retry_inc", increment_image_retry)
    
    # ── Arestas de texto ──
    # START → [search_router] → searcher OU generator
    graph.add_conditional_edges(
        START,
        search_router,
        {
            "searcher": "searcher",
            "generator": "generator"
        }
    )
    
    # searcher → generator
    graph.add_edge("searcher", "generator")
    
    # generator → reviewer
    graph.add_edge("generator", "reviewer")
    
    # reviewer → [quality_router] → regenerate OU image_router_decision
    graph.add_conditional_edges(
        "reviewer",
        quality_router,
        {
            "regenerate": "generator",
            "finish": "__image_decision__"
        }
    )
    
    # ── Decisão de imagem ──
    # Nó virtual para decidir se gera imagem ou finaliza
    graph.add_node("__image_decision__", lambda state: {})
    graph.add_conditional_edges(
        "__image_decision__",
        image_router_decision,
        {
            "image_generator": "image_generator",
            "__end__": END
        }
    )
    
    # ── Arestas de imagem ──
    # image_generator → image_validator
    graph.add_edge("image_generator", "image_validator")
    
    # image_validator → [image_quality_router] → retry OU finish
    graph.add_conditional_edges(
        "image_validator",
        image_quality_router,
        {
            "image_generator": "image_retry_inc",
            "__end__": END
        }
    )
    
    # image_retry_inc → image_generator
    graph.add_edge("image_retry_inc", "image_generator")
    
    # Compila o grafo
    compiled = graph.compile()
    logger.info("✅ Grafo LangGraph compilado com sucesso")
    
    return compiled


class LangGraphQuestionOrchestrator:
    """
    Orquestrador de geração de questões usando LangGraph.
    
    Encapsula o grafo e fornece uma interface simples para
    geração de questões com garantia de qualidade.
    """
    
    def __init__(self):
        """Inicializa o orquestrador com o grafo compilado."""
        self.graph = create_question_graph()
        logger.info("🚀 LangGraphQuestionOrchestrator inicializado")
    
    def generate(self, query: RequestBodyAgentQuestion) -> QuestionListSchema:
        """
        Gera questões usando o pipeline multi-agente.
        
        Args:
            query: Parâmetros da requisição
            
        Returns:
            Schema com lista de questões geradas e validadas
        """
        logger.info(
            f"🎯 Iniciando geração multi-agente | "
            f"Qtd: {query.count_questions} | "
            f"Habilidade: {query.skill[:40]}... | "
            f"Busca Real: {query.use_real_text}"
        )
        
        # Estado inicial
        initial_state: AgentState = {
            "query": query,
            "real_texts": None,
            "questions": [],
            "revision_feedback": None,
            "quality_score": None,
            "retry_count": 0,
            "error": None,
            "image_results": None,
            "image_retry_count": 0
        }
        
        # Executa o grafo
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extrai questões do estado final
            questions_data = final_state.get("questions", [])
            quality_score = final_state.get("quality_score", 0)
            retry_count = final_state.get("retry_count", 0)
            
            logger.info(
                f"🏁 Geração concluída | "
                f"Questões: {len(questions_data)} | "
                f"Score: {quality_score:.2f} | "
                f"Tentativas: {retry_count}"
            )
            
            # Converte para schema
            questions = [QuestionSchema(**q) for q in questions_data]
            
            return QuestionListSchema(questions=questions)
            
        except Exception as e:
            logger.error(f"❌ Erro na execução do grafo: {e}")
            raise

    def generate_with_progress(self, query: RequestBodyAgentQuestion, progress) -> QuestionListSchema:
        """
        Generates questions with real-time progress tracking.
        
        Sets the thread-local ProgressManager so each agent node can emit
        granular sub-step events. Uses graph.stream() for node-by-node execution.
        
        Args:
            query: Request parameters
            progress: ProgressManager for emitting events
            
        Returns:
            Schema with generated and validated questions
        """
        from app.services.progress_manager import set_current_progress
        
        logger.info(
            f"🎯 Starting generation with progress | "
            f"Qty: {query.count_questions} | "
            f"Skill: {query.skill[:40]}..."
        )
        
        # Set thread-local so agents can access progress
        set_current_progress(progress)
        
        # Routing phase
        progress.phase_start("routing", "Request Analysis", "🔀")
        if query.use_real_text:
            progress.log("routing", "Strategy: Real text search enabled", "", "📚")
        else:
            progress.log("routing", "Strategy: Direct generation (no text search)", "", "⚡")
        progress.phase_end("routing", "Route determined")
        
        # Initial state
        initial_state: AgentState = {
            "query": query,
            "real_texts": None,
            "questions": [],
            "revision_feedback": None,
            "quality_score": None,
            "retry_count": 0,
            "error": None,
            "image_results": None,
            "image_retry_count": 0
        }
        
        try:
            last_state = initial_state
            
            # Node-by-node phase labels (English)
            phase_labels = {
                "searcher": ("Text Search Agent", "📚"),
                "generator": ("Question Generator Agent", "✨"),
                "reviewer": ("Quality Review Agent", "📋"),
                "image_generator": ("Image Generation Agent", "🎨"),
                "image_validator": ("Image Validation Agent", "👁️"),
                "image_retry_inc": ("Image Retry", "🔄"),
                "__image_decision__": ("Image Decision", "🖼️"),
            }
            
            for event in self.graph.stream(initial_state):
                for node_name, node_output in event.items():
                    logger.info(f"📡 Node executed: {node_name}")
                    
                    label, icon = phase_labels.get(node_name, (node_name, "⏳"))
                    
                    # Emit phase boundaries
                    progress.phase_start(node_name, label, icon)
                    
                    # Node-specific summaries
                    if node_name == "searcher":
                        texts = node_output.get("real_texts")
                        count = len(texts) if texts else 0
                        progress.phase_end(node_name, f"{count} texts found")
                        
                    elif node_name == "generator":
                        questions = node_output.get("questions", [])
                        retry = node_output.get("retry_count", 0)
                        if retry > 1:
                            progress.retry(retry - 1, "Quality below threshold, regenerating...")
                        progress.phase_end(node_name, f"{len(questions)} questions generated")
                        
                    elif node_name == "reviewer":
                        score = node_output.get("quality_score", 0)
                        score_pct = f"{score * 100:.0f}%" if score else "N/A"
                        progress.phase_end(node_name, f"Quality score: {score_pct}")
                    
                    if node_output:
                        last_state = {**last_state, **node_output}
            
            # Final validation phase
            progress.phase_start("quality_gate", "Quality Gate", "✅")
            
            questions_data = last_state.get("questions", [])
            quality_score = last_state.get("quality_score", 0)
            retry_count = last_state.get("retry_count", 0)
            
            # Safely cast quality_score to float
            try:
                quality_score = float(quality_score) if quality_score else 0.0
            except (ValueError, TypeError):
                quality_score = 0.0
            
            score_pct = f"{quality_score * 100:.0f}%" if quality_score else "0%"
            progress.log("quality_gate", f"Final score: {score_pct}", "", "🎯")
            progress.log("quality_gate", f"Total attempts: {retry_count}", "", "🔄")
            progress.log("quality_gate", f"Questions delivered: {len(questions_data)}", "", "📝")
            progress.phase_end("quality_gate", f"Approved with {score_pct}")
            
            logger.info(
                f"🏁 Generation with progress completed | "
                f"Questions: {len(questions_data)} | "
                f"Score: {quality_score:.2f} | "
                f"Attempts: {retry_count}"
            )
            
            # Serialize questions for SSE — use raw dicts from state
            questions_serialized = []
            image_results = last_state.get("image_results") or []
            
            # Prepare image persistence
            import base64
            import uuid
            images_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
            os.makedirs(images_dir, exist_ok=True)
            
            for idx, q in enumerate(questions_data):
                if isinstance(q, dict):
                    q_dict = dict(q)
                elif hasattr(q, 'model_dump'):
                    q_dict = q.model_dump()
                else:
                    q_dict = q.__dict__ if hasattr(q, '__dict__') else {}
                
                # Attach image data from the pipeline
                img_result = next(
                    (r for r in image_results if r.get("question_index") == idx),
                    None
                )
                if img_result:
                    if img_result.get("validation_status") == "valid" and img_result.get("image_base64"):
                        img_b64 = img_result["image_base64"]
                        q_dict["image_base64"] = img_b64
                        q_dict["needs_manual_image"] = False
                        
                        # Save to disk so image persists via URL
                        try:
                            filename = f"question_{uuid.uuid4().hex[:12]}.png"
                            filepath = os.path.join(images_dir, filename)
                            with open(filepath, "wb") as f:
                                f.write(base64.b64decode(img_b64))
                            q_dict["image_url"] = f"/static/images/{filename}"
                            logger.info(f"💾 Image saved: /static/images/{filename}")
                        except Exception as save_err:
                            logger.warning(f"⚠️ Failed to save image to disk: {save_err}")
                    else:
                        q_dict["needs_manual_image"] = True
                        q_dict["image_validation_issues"] = img_result.get("validation_issues", [])
                
                questions_serialized.append(q_dict)
            
            # Emit finished event FIRST (so SSE stream gets it)
            progress.finish({
                "questions": questions_serialized,
                "quality_score": quality_score,
                "retry_count": retry_count
            })
            
            # Build result with serialized data (includes image_base64, image_url)
            try:
                questions = [QuestionSchema(**{k: v for k, v in q.items() if k in QuestionSchema.model_fields}) for q in questions_serialized]
                result = QuestionListSchema(questions=questions)
            except Exception as schema_err:
                logger.warning(f"⚠️ QuestionSchema construction failed: {schema_err}")
                result = QuestionListSchema(questions=[])
            
            # Attach serialized data with images for DB persistence
            result._serialized_questions = questions_serialized
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Error in progress execution: {e}")
            logger.error(traceback.format_exc())
            if not progress._finished:
                progress.error(str(e))
            raise
        finally:
            set_current_progress(None)


# Singleton para reutilização do grafo
_orchestrator_instance = None


def get_orchestrator() -> LangGraphQuestionOrchestrator:
    """
    Obtém a instância do orquestrador (singleton).
    
    Returns:
        Instância do LangGraphQuestionOrchestrator
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LangGraphQuestionOrchestrator()
    return _orchestrator_instance

