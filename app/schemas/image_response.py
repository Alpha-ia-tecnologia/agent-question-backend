from pydantic import BaseModel, Field
from typing import Optional, List


class UpdatedAlternative(BaseModel):
    """Alternativa com distrator potencialmente atualizado."""
    letter: str = Field(description="Letra da alternativa")
    text: str = Field(description="Texto original da alternativa")
    distractor: Optional[str] = Field(default=None, description="Distrator atualizado ou original")
    modified: bool = Field(default=False, description="Se o distrator foi alterado")


class ImageResponse(BaseModel):
    image_base64: str
    image_url: Optional[str] = None  # URL persistente da imagem salva em disco
    alternatives: Optional[List[UpdatedAlternative]] = None  # Alternativas com distratores atualizados
    distractors_updated: bool = Field(default=False, description="Flag geral: algum distrator foi alterado")
