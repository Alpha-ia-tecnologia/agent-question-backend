"""
Rotas de Questões.

Endpoints CRUD para gerenciamento de questões no banco de dados.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from app.utils.connect_db import get_session
from app.repositories.question_repository import QuestionRepository
from app.models.question_model import Question


# Schemas de resposta
class AlternativeResponse(BaseModel):
    """Schema de resposta para alternativa."""
    id: int
    letter: str
    text: str
    distractor: Optional[str] = None
    is_correct: bool


class QuestionResponse(BaseModel):
    """Schema de resposta para questão."""
    id: int
    question_number: Optional[int] = None
    id_skill: Optional[str] = None
    skill: Optional[str] = None
    proficiency_level: Optional[str] = None
    proficiency_description: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None
    source_author: Optional[str] = None
    question_statement: Optional[str] = None
    correct_answer: Optional[str] = None
    explanation_question: Optional[str] = None
    model_evaluation_type: Optional[str] = None
    grade: Optional[str] = None
    curriculum_component: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    quality_score: Optional[float] = None
    validated: bool = False
    observation: Optional[str] = None
    alternatives: List[AlternativeResponse] = []
    
    class Config:
        from_attributes = True


class QuestionListResponse(BaseModel):
    """Schema de lista de questões."""
    questions: List[QuestionResponse]
    total: int


class GenerationHistoryResponse(BaseModel):
    """Schema de resposta do histórico."""
    id: int
    skill: str
    proficiency_level: str
    grade: Optional[str]
    model_evaluation_type: Optional[str]
    count_questions: int
    questions_generated: int
    quality_score_avg: Optional[float]
    success: bool
    processing_time: Optional[float]
    
    class Config:
        from_attributes = True


# Router
question_router = APIRouter(prefix="/api/questions")


@question_router.get("/counts")
def get_question_counts(
    session: Session = Depends(get_session)
):
    """Retorna contadores de questões (total, validadas, pendentes)."""
    repo = QuestionRepository(session)
    total = repo.count_questions()
    validated = repo.count_questions(validated=True)
    return {
        "total": total,
        "validated": validated,
        "pending": total - validated
    }


@question_router.get("/", response_model=QuestionListResponse)
def list_questions(
    skill: Optional[str] = Query(None, description="Filtrar por habilidade"),
    proficiency_level: Optional[str] = Query(None, description="Filtrar por nível"),
    validated: Optional[bool] = Query(None, description="Filtrar por validação"),
    limit: int = Query(50, ge=1, le=200, description="Limite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginação"),
    session: Session = Depends(get_session)
):
    """
    Lista questões com filtros e paginação.
    
    - **skill**: Filtro parcial por habilidade
    - **proficiency_level**: Filtro exato por nível
    - **validated**: Filtro por status de validação (true/false)
    - **limit**: Máximo de resultados (1-200)
    - **offset**: Paginação
    """
    repo = QuestionRepository(session)
    try:
        questions = repo.get_questions(
            skill=skill,
            proficiency_level=proficiency_level,
            validated=validated,
            limit=limit,
            offset=offset
        )
        
        # Adiciona alternativas a cada questão
        result = []
        for q in questions:
            alternatives = repo.get_alternatives_by_question(q.id)
            q_dict = {
                "id": q.id,
                "question_number": q.question_number,
                "id_skill": q.id_skill,
                "skill": q.skill,
                "proficiency_level": q.proficiency_level,
                "proficiency_description": q.proficiency_description,
                "title": q.title,
                "text": q.text,
                "source": q.source,
                "source_author": q.source_author,
                "question_statement": q.question_statement,
                "correct_answer": q.correct_answer,
                "explanation_question": q.explanation_question,
                "model_evaluation_type": q.model_evaluation_type,
                "grade": q.grade,
                "curriculum_component": getattr(q, 'curriculum_component', None),
                "image_base64": q.image_base64,
                "image_url": q.image_url,
                "quality_score": q.quality_score,
                "validated": getattr(q, 'validated', False),
                "observation": getattr(q, 'observation', None),
                "alternatives": [
                    {
                        "id": alt.id,
                        "letter": alt.letter,
                        "text": alt.text,
                        "distractor": getattr(alt, 'distractor', None),
                        "is_correct": alt.is_correct
                    }
                    for alt in alternatives
                ]
            }
            result.append(q_dict)
        
        return QuestionListResponse(questions=result, total=len(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ValidationUpdate(BaseModel):
    """Schema para atualização de validação."""
    validated: bool


@question_router.patch("/{question_id}/validate")
def toggle_question_validation(
    question_id: int,
    data: ValidationUpdate,
    session: Session = Depends(get_session)
):
    """
    Atualiza o status de validação de uma questão.
    
    - **question_id**: ID da questão
    - **validated**: Status de validação (true/false)
    """
    repo = QuestionRepository(session)
    question = repo.update_question_validation(question_id, data.validated)
    
    if not question:
        raise HTTPException(status_code=404, detail="Questão não encontrada")
    
    return {
        "message": "Validação atualizada com sucesso",
        "id": question_id,
        "validated": question.validated
    }


class ObservationUpdate(BaseModel):
    """Schema para atualização de observação."""
    observation: Optional[str] = None


@question_router.patch("/{question_id}/observation")
def update_question_observation(
    question_id: int,
    data: ObservationUpdate,
    session: Session = Depends(get_session)
):
    """
    Atualiza a observação de uma questão.
    """
    repo = QuestionRepository(session)
    question = session.get(Question, question_id)
    
    if not question:
        raise HTTPException(status_code=404, detail="Questão não encontrada")
    
    question.observation = data.observation
    session.commit()
    session.refresh(question)
    
    return {
        "message": "Observação atualizada com sucesso",
        "id": question_id,
        "observation": question.observation
    }


@question_router.get("/{question_id}", response_model=QuestionResponse)
def get_question(
    question_id: int,
    session: Session = Depends(get_session)
):
    """Busca uma questão por ID."""
    repo = QuestionRepository(session)
    question = repo.get_question_by_id(question_id)
    
    if not question:
        raise HTTPException(status_code=404, detail="Questão não encontrada")
    
    alternatives = repo.get_alternatives_by_question(question_id)
    
    return QuestionResponse(
        id=question.id,
        question_number=question.question_number,
        id_skill=question.id_skill,
        skill=question.skill,
        proficiency_level=question.proficiency_level,
        proficiency_description=question.proficiency_description,
        title=question.title,
        text=question.text,
        source=question.source,
        source_author=question.source_author,
        question_statement=question.question_statement,
        correct_answer=question.correct_answer,
        explanation_question=question.explanation_question,
        model_evaluation_type=question.model_evaluation_type,
        grade=question.grade,
        curriculum_component=getattr(question, 'curriculum_component', None),
        image_base64=question.image_base64,
        image_url=question.image_url,
        quality_score=question.quality_score,
        alternatives=[
            AlternativeResponse(
                id=alt.id,
                letter=alt.letter,
                text=alt.text,
                distractor=getattr(alt, 'distractor', None),
                is_correct=alt.is_correct
            )
            for alt in alternatives
        ]
    )


@question_router.delete("/{question_id}")
def delete_question(
    question_id: int,
    session: Session = Depends(get_session)
):
    """Remove uma questão e suas alternativas."""
    repo = QuestionRepository(session)
    success = repo.delete_question(question_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Questão não encontrada")
    
    return {"message": "Questão removida com sucesso", "id": question_id}


@question_router.get("/history/", response_model=List[GenerationHistoryResponse])
def get_generation_history(
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session)
):
    """Lista o histórico de gerações."""
    repo = QuestionRepository(session)
    history = repo.get_generation_history(limit=limit)
    
    return [
        GenerationHistoryResponse(
            id=h.id,
            skill=h.skill,
            proficiency_level=h.proficiency_level,
            grade=h.grade,
            model_evaluation_type=h.model_evaluation_type,
            count_questions=h.count_questions,
            questions_generated=h.questions_generated,
            quality_score_avg=h.quality_score_avg,
            success=h.success,
            processing_time=h.processing_time
        )
        for h in history
    ]


# ========== SCHEMAS E ROTAS DE GRUPOS ==========

class QuestionGroupResponse(BaseModel):
    """Schema de resposta para grupo de questões."""
    id: int
    name: str
    description: Optional[str]
    skill: str
    proficiency_level: str
    grade: Optional[str]
    model_evaluation_type: Optional[str]
    image_dependency: Optional[str]
    curriculum_component: Optional[str] = None
    total_questions: int
    questions_with_image: int
    quality_score_avg: Optional[float]
    created_at: str
    
    class Config:
        from_attributes = True


class GroupWithQuestionsResponse(BaseModel):
    """Schema de grupo com suas questões."""
    group: QuestionGroupResponse
    questions: List[QuestionResponse]


# Router para grupos
group_router = APIRouter(prefix="/api/groups")


@group_router.get("/", response_model=List[QuestionGroupResponse])
def list_groups(
    limit: int = Query(50, ge=1, le=200, description="Limite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginação"),
    session: Session = Depends(get_session)
):
    """Lista todos os grupos de questões."""
    repo = QuestionRepository(session)
    groups = repo.get_groups(limit=limit, offset=offset)
    
    return [
        QuestionGroupResponse(
            id=g.id,
            name=g.name,
            description=g.description,
            skill=g.skill,
            proficiency_level=g.proficiency_level,
            grade=g.grade,
            model_evaluation_type=g.model_evaluation_type,
            image_dependency=g.image_dependency,
            curriculum_component=getattr(g, 'curriculum_component', None),
            total_questions=g.total_questions,
            questions_with_image=g.questions_with_image,
            quality_score_avg=g.quality_score_avg,
            created_at=g.created_at.isoformat() if g.created_at else ""
        )
        for g in groups
    ]


@group_router.get("/{group_id}", response_model=GroupWithQuestionsResponse)
def get_group_with_questions(
    group_id: int,
    session: Session = Depends(get_session)
):
    """Busca um grupo com todas suas questões."""
    repo = QuestionRepository(session)
    group = repo.get_group_by_id(group_id)
    
    if not group:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
    
    questions = repo.get_questions_by_group(group_id)
    
    # Monta resposta com questões
    questions_list = []
    for q in questions:
        alternatives = repo.get_alternatives_by_question(q.id)
        questions_list.append(
            QuestionResponse(
                id=q.id,
                question_number=q.question_number,
                id_skill=q.id_skill,
                skill=q.skill,
                proficiency_level=q.proficiency_level,
                proficiency_description=q.proficiency_description,
                title=q.title,
                text=q.text,
                source=q.source,
                source_author=q.source_author,
                question_statement=q.question_statement,
                correct_answer=q.correct_answer,
                explanation_question=q.explanation_question,
                model_evaluation_type=q.model_evaluation_type,
                grade=q.grade,
                image_base64=q.image_base64,
                image_url=q.image_url,
                quality_score=q.quality_score,
                alternatives=[
                    AlternativeResponse(
                        id=alt.id,
                        letter=alt.letter,
                        text=alt.text,
                        distractor=getattr(alt, 'distractor', None),
                        is_correct=alt.is_correct
                    )
                    for alt in alternatives
                ]
            )
        )
    
    return GroupWithQuestionsResponse(
        group=QuestionGroupResponse(
            id=group.id,
            name=group.name,
            description=group.description,
            skill=group.skill,
            proficiency_level=group.proficiency_level,
            grade=group.grade,
            model_evaluation_type=group.model_evaluation_type,
            image_dependency=group.image_dependency,
            curriculum_component=getattr(group, 'curriculum_component', None),
            total_questions=group.total_questions,
            questions_with_image=group.questions_with_image,
            quality_score_avg=group.quality_score_avg,
            created_at=group.created_at.isoformat() if group.created_at else ""
        ),
        questions=questions_list
    )


@group_router.delete("/{group_id}")
def delete_group(
    group_id: int,
    session: Session = Depends(get_session)
):
    """Remove um grupo e todas suas questões."""
    repo = QuestionRepository(session)
    success = repo.delete_group(group_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
    
    return {"message": "Grupo removido com sucesso", "id": group_id}

