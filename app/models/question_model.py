"""
Modelo de Questão Educacional.

Define a estrutura de persistência para questões geradas pelo sistema.
Usa DeclarativeBase padrão (sem dataclass) para compatibilidade.
"""

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func, String, Text, ForeignKey, Integer, Float, Boolean
from datetime import datetime
from typing import Optional
from app.models.table_resgitry import table_registry


@table_registry.mapped
class QuestionGroup:
    """
    Grupo de questões geradas em uma mesma sessão.
    
    Agrupa todas as questões geradas em uma única requisição,
    permitindo visualização e gerenciamento conjunto.
    """
    __tablename__ = "question_groups"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Nome do grupo (gerado automaticamente ou definido pelo usuário)
    name: Mapped[str] = mapped_column(String(255), default="Grupo de Questões")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Parâmetros da geração
    skill: Mapped[str] = mapped_column(Text)
    proficiency_level: Mapped[str] = mapped_column(String(100))
    grade: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_evaluation_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    image_dependency: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    curriculum_component: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Contadores
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    questions_with_image: Mapped[int] = mapped_column(Integer, default=0)
    
    # Score médio de qualidade
    quality_score_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Usuário que criou
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


@table_registry.mapped
class Question:
    """
    Modelo de questão educacional.
    
    Armazena questões geradas com seus metadados de habilidade,
    proficiência e conteúdo.
    """
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Relacionamento com grupo
    group_id: Mapped[Optional[int]] = mapped_column(ForeignKey("question_groups.id"), nullable=True)
    
    # Identificação da questão
    question_number: Mapped[int] = mapped_column(Integer, default=1)
    id_skill: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    skill: Mapped[str] = mapped_column(Text)
    
    # Nível de proficiência
    proficiency_level: Mapped[str] = mapped_column(String(100))
    proficiency_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Conteúdo do texto base
    title: Mapped[str] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    source_author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Enunciado e resposta
    question_statement: Mapped[str] = mapped_column(Text)
    correct_answer: Mapped[str] = mapped_column(String(5))
    explanation_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metadados da geração
    model_evaluation_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    grade: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    curriculum_component: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Imagem (Base64 ou URL - URL preferível para economia de espaço)
    image_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Caminho local
    
    # Score de qualidade (do revisor)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Status de validação (revisão manual)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Observação do revisor
    observation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relacionamentos
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


@table_registry.mapped
class Alternative:
    """
    Modelo de alternativa de questão.
    
    Cada questão tem múltiplas alternativas (A, B, C, D, E).
    """
    __tablename__ = "alternatives"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Chave estrangeira para questão
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    
    # Identificação da alternativa
    letter: Mapped[str] = mapped_column(String(5))
    text: Mapped[str] = mapped_column(Text)
    
    # Explicação do distrator (por que está certa/errada)
    distractor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Indica se é a resposta correta
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


@table_registry.mapped
class GenerationHistory:
    """
    Histórico de gerações de questões.
    
    Armazena metadados de cada sessão de geração para análise.
    """
    __tablename__ = "generation_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Usuário que solicitou
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Parâmetros da requisição
    skill: Mapped[str] = mapped_column(Text)
    proficiency_level: Mapped[str] = mapped_column(String(100))
    grade: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_evaluation_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    count_questions: Mapped[int] = mapped_column(Integer, default=1)
    image_dependency: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Resultado
    questions_generated: Mapped[int] = mapped_column(Integer, default=0)
    quality_score_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Tempo de processamento (em segundos)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
