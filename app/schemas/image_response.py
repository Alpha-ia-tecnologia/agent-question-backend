from pydantic import BaseModel
from typing import Optional


class ImageResponse(BaseModel):
    image_base64: str
    image_url: Optional[str] = None  # URL persistente da imagem salva em disco
