from pydantic import BaseModel, Field
from typing import Optional, List


class UpdatedAlternative(BaseModel):
    """Alternativa potencialmente atualizada após validação com imagem."""
    letter: str = Field(description="Letra da alternativa")
    text: str = Field(description="Texto da alternativa (original ou corrigido)")
    distractor: Optional[str] = Field(default=None, description="Distrator atualizado ou original")
    modified: bool = Field(default=False, description="Se algo foi alterado nesta alternativa")
    text_modified: bool = Field(default=False, description="Se o TEXTO da alternativa foi alterado")


class ImageResponse(BaseModel):
    image_base64: str
    image_url: Optional[str] = None
    alternatives: Optional[List[UpdatedAlternative]] = None
    distractors_updated: bool = Field(default=False, description="Flag geral: algo foi alterado nas alternativas")
    correct_answer: Optional[str] = Field(default=None, description="Letra da resposta correta (pode ter mudado)")

