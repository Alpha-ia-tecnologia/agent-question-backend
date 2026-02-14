"""
Rotas do Agente de Questões.

Endpoints para geração de questões educacionais e imagens ilustrativas.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from http import HTTPStatus
import logging
from sqlalchemy.orm import Session
import time
import asyncio
import threading

from app.schemas.response_agent_schema import ReponseAgentSchema
from app.schemas.request_body_agent import RequestBodyAgentQuestion
from app.services.generate_question_agent_service import GenerateQuestionAgentService
from app.services.generate_docx_service import GenerateDocxService
from app.schemas.question_schema import QuestionSchema
from app.services.generate_image_agent_service import GenerateImageAgentService
from app.schemas.image_response import ImageResponse
from app.core.llm_config import QuestionGenerationError, ImageGenerationError
from app.utils.connect_db import get_session
from app.repositories.question_repository import QuestionRepository

# Logger para este módulo
logger = logging.getLogger(__name__)

# Router e serviços
agent_router = APIRouter(prefix="/agent")
generate_question_agent_service = GenerateQuestionAgentService()
generate_image_agent_service = GenerateImageAgentService()
generate_docx_service = GenerateDocxService()


@agent_router.post(
    "/ask-agent",
    status_code=HTTPStatus.OK,
    response_model=ReponseAgentSchema,
    summary="Gerar questões educacionais",
    description="Gera questões de múltipla escolha usando IA baseado em habilidades e níveis de proficiência."
)
async def ask_agent(query: RequestBodyAgentQuestion, session: Session = Depends(get_session)):
    """
    Endpoint responsável por gerar questões.
    
    Campos:
     - count_questions: quantidade de questões a serem geradas
     - count_alternatives: quantidade de alternativas por questão
     - skill: habilidade a ser trabalhada nas questões
     - proficiency_level: nível de proficiência a ser trabalhado nas questões (número + descrição)
     - grade: ano letivo
     - authentic: true para gerar questões com textos de referências, false para textos gerados por IA

    As questões são automaticamente salvas no banco de dados, organizadas em grupos.
    """
    try:
        start_time = time.time()
        
        # Gera as questões
        generated_questions = generate_question_agent_service.generate_questions(query)
        
        processing_time = time.time() - start_time
        
        # Salva no banco de dados
        try:
            repo = QuestionRepository(session)
            
            # Converte questões para dicts
            questions_data = []
            for q in generated_questions.questions:
                q_dict = q.model_dump() if hasattr(q, 'model_dump') else q.__dict__
                # Adiciona metadados
                q_dict['skill'] = query.skill
                q_dict['proficiency_level'] = query.proficiency_level
                q_dict['grade'] = query.grade
                q_dict['model_evaluation_type'] = query.model_evaluation_type
                q_dict['curriculum_component'] = query.curriculum_component
                questions_data.append(q_dict)
            
            # Cria grupo com questões
            metadata = {
                'skill': query.skill,
                'proficiency_level': query.proficiency_level,
                'grade': query.grade,
                'model_evaluation_type': query.model_evaluation_type,
                'image_dependency': query.image_dependency,
                'curriculum_component': query.curriculum_component,
                'count_questions': query.count_questions,
                'processing_time': processing_time
            }
            
            group, saved_questions = repo.create_group_with_questions(
                questions_data=questions_data,
                metadata=metadata,
                user_id=None  # TODO: Obter do token de autenticação
            )
            
            logger.info(f"✅ Grupo #{group.id} criado com {len(saved_questions)} questões")
            
        except Exception as db_error:
            logger.warning(f"⚠️ Erro ao salvar no banco (questões ainda serão retornadas): {db_error}")
        
        return generated_questions
    
    except QuestionGenerationError as e:
        logger.error(f"Erro na geração de questões: {e}")
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Serviço de geração de questões temporariamente indisponível. {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro inesperado na geração de questões: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a requisição."
        )


@agent_router.post(
    "/ask-agent-stream",
    summary="Gerar questões com progresso em tempo real",
    description="Gera questões via SSE (Server-Sent Events) com acompanhamento etapa por etapa do pipeline LangGraph."
)
async def ask_agent_stream(query: RequestBodyAgentQuestion, session: Session = Depends(get_session)):
    """
    Endpoint SSE para geração de questões com progresso em tempo real.
    
    Emite eventos no formato:
    - init: lista de etapas do pipeline
    - step_started: etapa iniciada
    - step_completed: etapa concluída  
    - retry: regeneração por qualidade insuficiente
    - finished: resultado final com questões
    - error: erro durante a geração
    """
    from app.services.progress_manager import ProgressManager
    from app.services.langgraph_orchestrator import get_orchestrator
    
    progress = ProgressManager()
    
    # Capture event loop BEFORE starting thread (fixes race condition)
    loop = asyncio.get_event_loop()
    progress.set_loop(loop)
    
    def run_generation():
        """Executa a geração em thread separada."""
        try:
            start_time = time.time()
            orchestrator = get_orchestrator()
            result = orchestrator.generate_with_progress(query, progress)
            
            processing_time = time.time() - start_time
            
            # Salva no banco de dados (nova sessão para a thread)
            try:
                from app.utils.connect_db import engine
                thread_session = Session(bind=engine)
                try:
                    repo = QuestionRepository(thread_session)
                    
                    # Use serialized questions with image data when available
                    if hasattr(result, '_serialized_questions') and result._serialized_questions:
                        questions_data = []
                        for q_dict in result._serialized_questions:
                            q_dict = dict(q_dict)  # copy
                            q_dict['skill'] = query.skill
                            q_dict['proficiency_level'] = query.proficiency_level
                            q_dict['grade'] = query.grade
                            q_dict['model_evaluation_type'] = query.model_evaluation_type
                            q_dict['curriculum_component'] = query.curriculum_component
                            questions_data.append(q_dict)
                    else:
                        questions_data = []
                        for q in result.questions:
                            q_dict = q.model_dump() if hasattr(q, 'model_dump') else q.__dict__
                            q_dict['skill'] = query.skill
                            q_dict['proficiency_level'] = query.proficiency_level
                            q_dict['grade'] = query.grade
                            q_dict['model_evaluation_type'] = query.model_evaluation_type
                            q_dict['curriculum_component'] = query.curriculum_component
                            questions_data.append(q_dict)
                    
                    metadata = {
                        'skill': query.skill,
                        'proficiency_level': query.proficiency_level,
                        'grade': query.grade,
                        'model_evaluation_type': query.model_evaluation_type,
                        'image_dependency': query.image_dependency,
                        'curriculum_component': query.curriculum_component,
                        'count_questions': query.count_questions,
                        'processing_time': processing_time
                    }
                    
                    group, saved_questions = repo.create_group_with_questions(
                        questions_data=questions_data,
                        metadata=metadata,
                        user_id=None
                    )
                    logger.info(f"✅ Grupo #{group.id} criado com {len(saved_questions)} questões (stream)")
                finally:
                    thread_session.close()
            except Exception as db_error:
                logger.warning(f"⚠️ Erro ao salvar no banco (stream): {db_error}")
                
        except Exception as e:
            logger.error(f"❌ Erro na geração stream: {e}")
            import traceback
            traceback.print_exc()
            progress.error(str(e))
    
    # Inicia geração em thread separada
    thread = threading.Thread(target=run_generation, daemon=True)
    progress._thread = thread  # Allow stream to detect thread death
    thread.start()
    
    return StreamingResponse(
        progress.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@agent_router.post(
    "/ask-image",
    status_code=HTTPStatus.OK,
    response_model=ImageResponse,
    summary="Gerar imagem para questão",
    description="Gera uma imagem ilustrativa para uma questão educacional usando IA multimodal."
)
async def generate_image(question: QuestionSchema, session: Session = Depends(get_session)):
    """
    Endpoint responsável por gerar uma imagem a partir de uma questão.
    
    Recebe uma questão e retorna uma imagem ilustrativa em Base64.
    Se a questão já estiver salva no banco (tem ID), a imagem será salva em disco e vinculada.
    """
    try:
        logger.info(f"Recebida requisição de imagem para questão #{question.question_number}")
        
        # Gera a imagem
        image_response = generate_image_agent_service.generate_image(question)
        
        # Sempre salva a imagem em disco (com UUID para persistência)
        if image_response.image_base64:
            try:
                import uuid
                import base64
                import os
                
                # Cria diretório se não existir
                static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "static", "images")
                os.makedirs(static_dir, exist_ok=True)
                
                # Gera nome único para o arquivo
                image_id = str(uuid.uuid4())[:8]
                filename = f"question_{question.question_number}_{image_id}.png"
                filepath = os.path.join(static_dir, filename)
                
                # Salva a imagem
                image_data = base64.b64decode(image_response.image_base64)
                with open(filepath, "wb") as f:
                    f.write(image_data)
                
                # Retorna URL completa
                image_url = f"/static/images/{filename}"
                logger.info(f"✅ Imagem salva em {filepath}")
                
                # Retorna resposta com URL
                return ImageResponse(
                    image_base64=image_response.image_base64,
                    image_url=f"http://localhost:8000{image_url}"
                )
            except Exception as save_error:
                logger.warning(f"⚠️ Erro ao salvar imagem em disco: {save_error}")
        
        return image_response
    
    except ImageGenerationError as e:
        logger.error(f"Erro na geração de imagem: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Serviço de geração de imagens temporariamente indisponível. {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro inesperado na geração de imagem: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a requisição: {str(e)}"
        )


class ImageRegenerationRequest(BaseModel):
    """Schema para requisição de regeneração de imagem com instruções personalizadas."""
    question: QuestionSchema
    custom_instructions: str = Field(description="Instruções personalizadas para correção/melhoria da imagem")


@agent_router.post(
    "/regenerate-image",
    status_code=HTTPStatus.OK,
    response_model=ImageResponse,
    summary="Regenerar imagem com instruções",
    description="Regenera uma imagem para uma questão com instruções personalizadas de correção."
)
async def regenerate_image(request: ImageRegenerationRequest):
    """
    Endpoint para regenerar imagem com instruções de correção.
    
    Permite ao usuário fornecer instruções específicas para melhorar a imagem.
    """
    try:
        logger.info(f"Regenerando imagem para questão #{request.question.question_number}")
        logger.info(f"Instruções: {request.custom_instructions[:100]}...")
        return generate_image_agent_service.generate_image_with_instructions(
            request.question, 
            request.custom_instructions
        )
    
    except ImageGenerationError as e:
        logger.error(f"Erro na regeneração de imagem: {e}")
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Serviço de geração de imagens temporariamente indisponível. {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro inesperado na regeneração de imagem: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a requisição: {str(e)}"
        )
