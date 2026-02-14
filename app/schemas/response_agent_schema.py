from pydantic import BaseModel 
from typing import List
from app.schemas.question_schema import QuestionSchema


class ReponseAgentSchema(BaseModel):
    questions: List[QuestionSchema]
    

    


