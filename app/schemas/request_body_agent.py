from pydantic import BaseModel, Field
from app.enums.model_evaluation_type import ModelEvaluationType
from typing import Literal

class RequestBodyAgentQuestion(BaseModel):
    count_questions: int
    count_alternatives: int
    skill: str
    proficiency_level: str
    grade: str
    curriculum_component: str = ""
    authentic: bool = False
    use_real_text: bool = False  # Se True, busca textos reais da internet
    image_dependency: Literal["none", "optional", "required"] = Field(
        default="none",
        description="Dependência de imagem: 'none' = sem imagem, 'optional' = imagem opcional, 'required' = resolução depende da imagem"
    )
    model_evaluation_type: ModelEvaluationType = ModelEvaluationType.SAEB
