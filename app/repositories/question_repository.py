"""
Repositório de Questões.

Operações CRUD para questões no banco de dados.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func as sa_func
from typing import List, Optional
from datetime import datetime
import os
import base64
import uuid

from app.models.question_model import Question, Alternative, GenerationHistory, QuestionGroup
from app.schemas.question_schema import QuestionSchema


# Diretório para salvar imagens
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


class QuestionRepository:
    """Repositório para operações com questões."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_question(self, question_data: dict, user_id: Optional[int] = None) -> Question:
        """
        Cria uma nova questão no banco.
        
        Args:
            question_data: Dicionário com dados da questão
            user_id: ID do usuário que criou (opcional)
            
        Returns:
            Questão criada
        """
        # Cria a questão
        question = Question(
            question_number=question_data.get("question_number", 1),
            id_skill=question_data.get("id_skill"),
            skill=question_data.get("skill", ""),
            proficiency_level=question_data.get("proficiency_level", ""),
            proficiency_description=question_data.get("proficiency_description"),
            title=question_data.get("title", ""),
            text=question_data.get("text", ""),
            source=question_data.get("source"),
            source_url=question_data.get("source_url"),
            source_author=question_data.get("source_author"),
            question_statement=question_data.get("question_statement", ""),
            correct_answer=question_data.get("correct_answer", ""),
            explanation_question=question_data.get("explanation_question"),
            model_evaluation_type=question_data.get("model_evaluation_type"),
            grade=question_data.get("grade"),
            image_base64=question_data.get("image_base64"),
            image_url=question_data.get("image_url"),
            quality_score=question_data.get("quality_score"),
            curriculum_component=question_data.get("curriculum_component"),
            user_id=user_id
        )
        
        self.session.add(question)
        self.session.commit()
        self.session.refresh(question)
        
        # Cria as alternativas
        alternatives = question_data.get("alternatives", [])
        for alt_data in alternatives:
            alt = Alternative(
                question_id=question.id,
                letter=alt_data.get("letter", ""),
                text=alt_data.get("text", ""),
                distractor=alt_data.get("distractor"),
                is_correct=(alt_data.get("letter") == question.correct_answer)
            )
            self.session.add(alt)
        
        self.session.commit()
        return question
    
    def create_questions_batch(
        self, 
        questions_data: List[dict], 
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> List[Question]:
        """
        Cria múltiplas questões de uma vez.
        
        Args:
            questions_data: Lista de dicionários com dados das questões
            user_id: ID do usuário
            metadata: Metadados da geração (para histórico)
            
        Returns:
            Lista de questões criadas
        """
        created_questions = []
        
        for q_data in questions_data:
            question = self.create_question(q_data, user_id)
            created_questions.append(question)
        
        # Registra no histórico
        if metadata:
            history = GenerationHistory(
                user_id=user_id,
                skill=metadata.get("skill", ""),
                proficiency_level=metadata.get("proficiency_level", ""),
                grade=metadata.get("grade"),
                model_evaluation_type=metadata.get("model_evaluation_type"),
                count_questions=metadata.get("count_questions", 1),
                image_dependency=metadata.get("image_dependency"),
                questions_generated=len(created_questions),
                quality_score_avg=metadata.get("quality_score_avg"),
                retry_count=metadata.get("retry_count", 0),
                success=True,
                processing_time=metadata.get("processing_time")
            )
            self.session.add(history)
            self.session.commit()
        
        return created_questions
    
    def get_question_by_id(self, question_id: int) -> Optional[Question]:
        """Busca questão por ID."""
        return self.session.query(Question).filter(Question.id == question_id).first()
    
    def get_questions(
        self,
        user_id: Optional[int] = None,
        skill: Optional[str] = None,
        proficiency_level: Optional[str] = None,
        validated: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Question]:
        """
        Lista questões com filtros.
        
        Args:
            user_id: Filtrar por usuário
            skill: Filtrar por habilidade
            proficiency_level: Filtrar por nível
            validated: Filtrar por status de validação
            limit: Limite de resultados
            offset: Offset para paginação
            
        Returns:
            Lista de questões
        """
        query = self.session.query(Question)
        
        if user_id:
            query = query.filter(Question.user_id == user_id)
        if skill:
            query = query.filter(Question.skill.ilike(f"%{skill}%"))
        if proficiency_level:
            query = query.filter(Question.proficiency_level == proficiency_level)
        if validated is not None:
            query = query.filter(Question.validated == validated)
        
        return query.order_by(desc(Question.id)).offset(offset).limit(limit).all()
    
    def count_questions(
        self,
        validated: Optional[bool] = None
    ) -> int:
        """Conta questões com filtro opcional de validação."""
        query = self.session.query(sa_func.count(Question.id))
        if validated is not None:
            query = query.filter(Question.validated == validated)
        return query.scalar() or 0
    
    def get_alternatives_by_question(self, question_id: int) -> List[Alternative]:
        """Busca alternativas de uma questão."""
        return self.session.query(Alternative).filter(
            Alternative.question_id == question_id
        ).order_by(Alternative.letter).all()
    
    def delete_question(self, question_id: int) -> bool:
        """
        Remove uma questão e suas alternativas.
        
        Args:
            question_id: ID da questão
            
        Returns:
            True se removida, False se não encontrada
        """
        question = self.get_question_by_id(question_id)
        if not question:
            return False
        
        # Remove alternativas primeiro
        self.session.query(Alternative).filter(
            Alternative.question_id == question_id
        ).delete()
        
        # Remove a questão
        self.session.delete(question)
        self.session.commit()
        
        return True
    
    def update_question_image(
        self, 
        question_id: int, 
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> Optional[Question]:
        """
        Atualiza a imagem de uma questão.
        
        Args:
            question_id: ID da questão
            image_base64: Imagem em Base64
            image_url: URL da imagem
            
        Returns:
            Questão atualizada ou None
        """
        question = self.get_question_by_id(question_id)
        if not question:
            return None
        
        if image_base64:
            question.image_base64 = image_base64
        if image_url:
            question.image_url = image_url
        
        self.session.commit()
        self.session.refresh(question)
        
        return question
    
    def get_generation_history(
        self,
        user_id: Optional[int] = None,
        limit: int = 50
    ) -> List[GenerationHistory]:
        """Busca histórico de gerações."""
        query = self.session.query(GenerationHistory)
        
        if user_id:
            query = query.filter(GenerationHistory.user_id == user_id)
        
        return query.order_by(desc(GenerationHistory.created_at)).limit(limit).all()
    
    def update_question_validation(
        self, 
        question_id: int, 
        validated: bool
    ) -> Optional[Question]:
        """
        Atualiza o status de validação de uma questão.
        
        Args:
            question_id: ID da questão
            validated: Status de validação
            
        Returns:
            Questão atualizada ou None
        """
        question = self.get_question_by_id(question_id)
        if not question:
            return None
        
        question.validated = validated
        self.session.commit()
        self.session.refresh(question)
        
        return question
    
    # ===== MÉTODOS PARA GRUPOS DE QUESTÕES =====
    
    def create_group(
        self,
        name: str,
        skill: str,
        proficiency_level: str,
        grade: Optional[str] = None,
        model_evaluation_type: Optional[str] = None,
        image_dependency: Optional[str] = None,
        curriculum_component: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> QuestionGroup:
        """
        Cria um novo grupo de questões.
        
        Args:
            name: Nome do grupo
            skill: Habilidade das questões
            proficiency_level: Nível de proficiência
            grade: Ano escolar
            model_evaluation_type: Tipo de avaliação
            image_dependency: Dependência de imagem
            user_id: ID do usuário
            
        Returns:
            Grupo criado
        """
        group = QuestionGroup()
        group.name = name
        group.skill = skill
        group.proficiency_level = proficiency_level
        group.grade = grade
        group.model_evaluation_type = model_evaluation_type
        group.image_dependency = image_dependency
        group.curriculum_component = curriculum_component
        group.user_id = user_id
        group.total_questions = 0
        group.questions_with_image = 0
        
        self.session.add(group)
        self.session.commit()
        self.session.refresh(group)
        
        return group
    
    def save_image_to_disk(self, image_base64: str, question_id: int) -> str:
        """
        Salva imagem Base64 em disco e retorna o caminho.
        
        Args:
            image_base64: Imagem codificada em Base64
            question_id: ID da questão
            
        Returns:
            Caminho relativo da imagem salva
        """
        # Gera nome único
        filename = f"question_{question_id}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Decodifica e salva
        try:
            image_data = base64.b64decode(image_base64)
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            # Retorna caminho relativo para URL
            return f"/static/images/{filename}"
        except Exception as e:
            raise ValueError(f"Erro ao salvar imagem: {e}")
    
    def create_group_with_questions(
        self,
        questions_data: List[dict],
        metadata: dict,
        user_id: Optional[int] = None
    ) -> tuple[QuestionGroup, List[Question]]:
        """
        Cria um grupo e suas questões de uma vez.
        
        Args:
            questions_data: Lista de dicts com dados das questões
            metadata: Metadados da geração (skill, grade, etc.)
            user_id: ID do usuário
            
        Returns:
            Tupla (grupo criado, lista de questões)
        """
        # Gera nome do grupo
        skill_short = metadata.get("skill", "Questões")[:50]
        group_name = f"{skill_short} - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        
        # Cria o grupo
        group = self.create_group(
            name=group_name,
            skill=metadata.get("skill", ""),
            proficiency_level=metadata.get("proficiency_level", ""),
            grade=metadata.get("grade"),
            model_evaluation_type=metadata.get("model_evaluation_type"),
            image_dependency=metadata.get("image_dependency"),
            curriculum_component=metadata.get("curriculum_component"),
            user_id=user_id
        )
        
        # Cria as questões vinculadas ao grupo
        created_questions = []
        images_count = 0
        
        for q_data in questions_data:
            # Adiciona o group_id
            q_data["group_id"] = group.id
            
            # Se tem imagem Base64, salva em disco
            if q_data.get("image_base64"):
                try:
                    # Primeiro cria a questão para ter o ID
                    question = self._create_question_in_group(q_data, group.id, user_id)
                    
                    # Depois salva a imagem
                    image_path = self.save_image_to_disk(q_data["image_base64"], question.id)
                    question.image_path = image_path
                    question.image_url = image_path  # URL = path para servir estático
                    
                    # Limpa o Base64 para não ocupar espaço no banco
                    question.image_base64 = None
                    
                    images_count += 1
                    created_questions.append(question)
                except Exception as e:
                    # Se falhar, cria sem imagem
                    question = self._create_question_in_group(q_data, group.id, user_id)
                    created_questions.append(question)
            else:
                question = self._create_question_in_group(q_data, group.id, user_id)
                created_questions.append(question)
        
        # Atualiza contadores do grupo
        group.total_questions = len(created_questions)
        group.questions_with_image = images_count
        
        self.session.commit()
        
        return group, created_questions
    
    def _create_question_in_group(
        self,
        question_data: dict,
        group_id: int,
        user_id: Optional[int]
    ) -> Question:
        """Cria questão dentro de um grupo (método interno)."""
        question = Question()
        question.group_id = group_id
        question.question_number = question_data.get("question_number", 1)
        question.id_skill = question_data.get("id_skill")
        question.skill = question_data.get("skill", "")
        question.proficiency_level = question_data.get("proficiency_level", "")
        question.proficiency_description = question_data.get("proficiency_description")
        question.title = question_data.get("title", "")
        question.text = question_data.get("text", "")
        question.source = question_data.get("source")
        question.source_url = question_data.get("source_url")
        question.source_author = question_data.get("source_author")
        question.question_statement = question_data.get("question_statement", "")
        question.correct_answer = question_data.get("correct_answer", "")
        question.explanation_question = question_data.get("explanation_question")
        question.model_evaluation_type = question_data.get("model_evaluation_type")
        question.grade = question_data.get("grade")
        question.curriculum_component = question_data.get("curriculum_component")
        question.quality_score = question_data.get("quality_score")
        question.user_id = user_id
        
        self.session.add(question)
        self.session.commit()
        self.session.refresh(question)
        
        # Cria alternativas
        alternatives = question_data.get("alternatives", [])
        for alt_data in alternatives:
            alt = Alternative()
            alt.question_id = question.id
            alt.letter = alt_data.get("letter", "")
            alt.text = alt_data.get("text", "")
            alt.distractor = alt_data.get("distractor")
            alt.is_correct = (alt_data.get("letter") == question.correct_answer)
            self.session.add(alt)
        
        self.session.commit()
        return question
    
    def get_groups(
        self,
        user_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[QuestionGroup]:
        """Lista grupos de questões com paginação."""
        query = self.session.query(QuestionGroup)
        
        if user_id:
            query = query.filter(QuestionGroup.user_id == user_id)
        
        return query.order_by(desc(QuestionGroup.created_at)).offset(offset).limit(limit).all()
    
    def get_group_by_id(self, group_id: int) -> Optional[QuestionGroup]:
        """Busca grupo por ID."""
        return self.session.query(QuestionGroup).filter(QuestionGroup.id == group_id).first()
    
    def get_questions_by_group(self, group_id: int) -> List[Question]:
        """Busca todas as questões de um grupo."""
        return self.session.query(Question).filter(
            Question.group_id == group_id
        ).order_by(Question.question_number).all()
    
    def delete_group(self, group_id: int) -> bool:
        """Remove um grupo e todas suas questões."""
        group = self.get_group_by_id(group_id)
        if not group:
            return False
        
        # Busca questões do grupo
        questions = self.get_questions_by_group(group_id)
        
        for q in questions:
            # Remove alternativas
            self.session.query(Alternative).filter(
                Alternative.question_id == q.id
            ).delete()
            
            # Remove imagem do disco se existir
            if q.image_path:
                try:
                    full_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        q.image_path.lstrip("/")
                    )
                    if os.path.exists(full_path):
                        os.remove(full_path)
                except Exception:
                    pass
            
            # Remove questão
            self.session.delete(q)
        
        # Remove o grupo
        self.session.delete(group)
        self.session.commit()
        
        return True
