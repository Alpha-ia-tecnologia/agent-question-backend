from pydantic import BaseModel, Field
from app.enums.model_evaluation_type import ModelEvaluationType
from typing import Literal, Optional, List


class CombinedSkillItem(BaseModel):
    """Item de habilidade para combinação interdisciplinar."""
    skill: str
    curriculum_component: str


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
    combined_skills: Optional[List[CombinedSkillItem]] = Field(
        default=None,
        description="Lista de habilidades de componentes curriculares diferentes para questões interdisciplinares. Quando presente (>=2 itens), ativa o modo interdisciplinar."
    )
    context_theme: Optional[str] = Field(
        default=None,
        description="Tema ou assunto complementar que direciona a geração da questão. Ex: 'Canção do Exílio', 'Semana da Água'. Quando presente, o gerador deve contextualizar a questão com base nesse tema."
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Override do modelo de LLM usado na geração (ex: 'gemini-pro-latest', 'gemini-2.5-flash', 'gpt-4o-mini', 'deepseek-chat'). Se omitido, usa o padrão configurado no servidor."
    )

    @property
    def is_interdisciplinary(self) -> bool:
        """Verifica se a requisição é interdisciplinar."""
        return bool(self.combined_skills and len(self.combined_skills) >= 2)
