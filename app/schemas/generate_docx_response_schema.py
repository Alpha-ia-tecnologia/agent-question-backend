from pydantic import BaseModel

class GenerateDocxResponseSchema(BaseModel):
    message: str
    link: str