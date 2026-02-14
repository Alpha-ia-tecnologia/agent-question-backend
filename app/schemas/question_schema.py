from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class AlternativeSchema(BaseModel):
    letter: str = Field(description="Letra da alternativa")
    text: str = Field(description="Texto da alternativa")
    distractor: Optional[str] = Field(default=None, description="Explicação do distrator: por que esta alternativa é correta ou incorreta")
    

class QuestionSchema(BaseModel):
    question_number: int = Field(description="Número da questão")
    id_skill: str = Field(description="ID da habilidade")
    skill: str = Field(description="Habilidade da questão")
    proficiency_level: str = Field(description="Nível de proficiência")
    proficiency_description: str = Field(description="Descrição da proficiência")
    title: str = Field(description="Título do texto")
    text: str = Field(description="Texto da questão")
    source: str = Field(description="Fonte do texto utilizado")
    source_url: Optional[str] = Field(default=None, description="URL da fonte original do texto")
    source_author: Optional[str] = Field(default=None, description="Autor original do texto")
    question_statement: str = Field(description="Enunciado da questão")
    alternatives: List[AlternativeSchema] = Field(description="Alternativas da questão")
    correct_answer: str = Field(description="Letra da alternativa correta")
    explanation_question: str = Field(description="Explicação das alternativas")
    image_data: Optional[Dict[str, Any]] = Field(default=None, description="Dados estruturados para geração da imagem (tipo, valores, rótulos, medidas)")
    

class QuestionListSchema(BaseModel):
    questions: List[QuestionSchema] = Field(description="Lista de questões geradas pelo agente")


class QuestionWithImageSchema(QuestionSchema):
    image_base64: Optional[str] = Field(default=None, description="Imagem em base64")
    image_url: Optional[str] = Field(default=None, description="URL da imagem no servidor")

    class Config:
        str_max_length = 1000000000