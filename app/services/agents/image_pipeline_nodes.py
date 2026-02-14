"""
Image Pipeline Nodes - Nós LangGraph para geração e validação de imagens.

Integra a geração de imagens diretamente no pipeline de questões,
com validação multimodal via Gemini Vision e retry automático.

🎯 Geração SEQUENCIAL: cada imagem é gerada após a questão completa,
garantindo fidelidade entre questão e imagem.

Fluxo:
    [quality_gate] → image_router_decision → image_generator → image_validator → image_quality_router
                                                   ↑                                      ↓
                                                   └── retry (max 2) ←←←←←←←←←←←←←←←←←←←┘
"""

import logging
from typing import Literal

from app.services.agents.state import AgentState
from app.services.progress_manager import get_current_progress
from app.schemas.question_schema import QuestionSchema

logger = logging.getLogger(__name__)

MAX_IMAGE_RETRIES = 2


def image_router_decision(state: AgentState) -> Literal["image_generator", "__end__"]:
    """
    Decide se deve gerar imagens ou finalizar.
    
    Rota para image_generator APENAS se image_dependency == "required".
    """
    query = state["query"]
    image_dep = getattr(query, "image_dependency", "none")
    
    if image_dep == "required":
        logger.info("🖼️ image_dependency=required → Roteando para geração de imagens")
        return "image_generator"
    else:
        logger.info(f"🖼️ image_dependency={image_dep} → Finalizando sem imagem")
        return "__end__"


def image_generator_node(state: AgentState) -> dict:
    """
    Gera imagens para TODAS as questões do batch SEQUENCIALMENTE.
    
    Cada imagem é gerada uma por uma, após a questão estar completa,
    garantindo que a imagem seja fidedigna ao conteúdo da questão.
    """
    from app.services.generate_image_agent_service import get_image_service
    
    progress = get_current_progress()
    questions = state.get("questions", [])
    existing_results = state.get("image_results") or []
    retry_count = state.get("image_retry_count", 0)
    
    if progress:
        progress.phase_start("image_generator", "Image Generation Agent", "🎨")
        progress.log("image_generator", f"Generating images for {len(questions)} questions", "", "🖼️")
    
    image_service = get_image_service()
    image_results = []
    
    for idx, q_data in enumerate(questions):
        # Verificar se já tem resultado válido de um ciclo anterior
        existing = next(
            (r for r in existing_results if r.get("question_index") == idx and r.get("validation_status") == "valid"),
            None
        )
        if existing:
            logger.info(f"✅ Questão {idx}: Imagem já válida, pulando")
            image_results.append(existing)
            continue
        
        # Verificar se há instruções de correção do validador
        corrections = None
        failed_result = next(
            (r for r in existing_results if r.get("question_index") == idx and r.get("validation_status") == "invalid"),
            None
        )
        if failed_result:
            corrections = failed_result.get("corrections", "")
        
        try:
            # Converter dict para QuestionSchema para o serviço
            if isinstance(q_data, dict):
                question = QuestionSchema(**q_data)
            else:
                question = q_data
            
            if progress:
                progress.log(
                    "image_generator",
                    f"Generating image {idx + 1}/{len(questions)}: {question.title[:40]}...",
                    corrections or "",
                    "🎨"
                )
            
            # Gerar com ou sem correções
            if corrections and retry_count > 0:
                logger.info(f"🔄 Regenerando imagem {idx} com correções: {corrections[:100]}...")
                result = image_service.generate_image_with_instructions(question, corrections)
            else:
                result = image_service.generate_image(question)
            
            image_results.append({
                "question_index": idx,
                "image_base64": result.image_base64,
                "validation_status": "pending",
                "attempts": retry_count + 1,
                "corrections": None
            })
            
            logger.info(f"✅ Imagem gerada para questão {idx}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar imagem para questão {idx}: {e}")
            image_results.append({
                "question_index": idx,
                "image_base64": None,
                "validation_status": "error",
                "attempts": retry_count + 1,
                "error": str(e),
                "corrections": None
            })
    
    if progress:
        generated_count = sum(1 for r in image_results if r.get("image_base64"))
        progress.phase_end("image_generator", f"{generated_count}/{len(questions)} images generated")
    
    return {
        "image_results": image_results,
        "image_retry_count": retry_count
    }


def image_validator_node(state: AgentState) -> dict:
    """
    Valida TODAS as imagens geradas usando Gemini Vision.
    
    Compara cada imagem com os dados da questão correspondente.
    """
    from app.services.agents.image_validator_agent import get_image_validator_agent
    
    progress = get_current_progress()
    questions = state.get("questions", [])
    image_results = state.get("image_results", [])
    
    if progress:
        progress.phase_start("image_validator", "Image Validation Agent", "👁️")
    
    validator = get_image_validator_agent()
    validated_results = []
    
    for result in image_results:
        idx = result.get("question_index", 0)
        
        # Pular resultados já válidos ou com erro
        if result.get("validation_status") in ("valid", "error"):
            validated_results.append(result)
            continue
        
        image_base64 = result.get("image_base64")
        if not image_base64:
            result["validation_status"] = "error"
            validated_results.append(result)
            continue
        
        # Obter dados da questão
        if idx < len(questions):
            q_data = questions[idx]
            if not isinstance(q_data, dict):
                q_data = q_data.model_dump() if hasattr(q_data, 'model_dump') else q_data.__dict__
        else:
            result["validation_status"] = "error"
            validated_results.append(result)
            continue
        
        if progress:
            title = q_data.get("title", "N/A")[:40]
            progress.log("image_validator", f"Validating image {idx + 1}: {title}...", "", "🔍")
        
        # Validar com Gemini Vision
        try:
            validation = validator.validate(q_data, image_base64)
            
            is_valid = validation.get("valid", False)
            score = validation.get("score", 0)
            
            result["validation_status"] = "valid" if is_valid else "invalid"
            result["validation_score"] = score
            result["validation_issues"] = validation.get("issues", [])
            result["corrections"] = validation.get("corrections", "") if not is_valid else None
            
            if progress:
                status_icon = "✅" if is_valid else "❌"
                progress.log(
                    "image_validator",
                    f"{status_icon} Image {idx + 1}: {'Approved' if is_valid else 'Rejected'} (score: {score})",
                    ", ".join(validation.get("issues", [])) if not is_valid else "",
                    status_icon
                )
                
        except Exception as e:
            logger.error(f"❌ Erro ao validar imagem {idx}: {e}")
            result["validation_status"] = "invalid"
            result["validation_issues"] = [str(e)]
            result["corrections"] = "Regenerar a imagem"
        
        validated_results.append(result)
    
    # Contar resultados
    valid_count = sum(1 for r in validated_results if r.get("validation_status") == "valid")
    invalid_count = sum(1 for r in validated_results if r.get("validation_status") == "invalid")
    
    if progress:
        progress.phase_end(
            "image_validator",
            f"✅ {valid_count} approved, ❌ {invalid_count} rejected"
        )
    
    return {"image_results": validated_results}


def image_quality_router(state: AgentState) -> Literal["image_generator", "__end__"]:
    """
    Decide se precisa regenerar imagens ou finalizar.
    
    - Se há imagens inválidas E attempts < MAX_IMAGE_RETRIES → retry
    - Caso contrário → finalizar (imagens inválidas ficam com needs_manual_image=true)
    """
    image_results = state.get("image_results", [])
    retry_count = state.get("image_retry_count", 0)
    
    invalid_images = [r for r in image_results if r.get("validation_status") == "invalid"]
    
    if invalid_images and retry_count < MAX_IMAGE_RETRIES:
        logger.info(
            f"🔄 Retry de imagem #{retry_count + 1}/{MAX_IMAGE_RETRIES}: "
            f"{len(invalid_images)} imagens inválidas"
        )
        
        progress = get_current_progress()
        if progress:
            progress.retry(
                retry_count + 1,
                f"{len(invalid_images)} images rejected, retrying..."
            )
        
        return "image_generator"
    
    if invalid_images:
        logger.warning(
            f"⚠️ {len(invalid_images)} imagens ainda inválidas após {retry_count} tentativas. "
            f"Habilitando geração manual."
        )
    else:
        logger.info("✅ Todas as imagens validadas com sucesso!")
    
    return "__end__"


def increment_image_retry(state: AgentState) -> dict:
    """Incrementa o contador de retry de imagem antes de regenerar."""
    return {"image_retry_count": state.get("image_retry_count", 0) + 1}
