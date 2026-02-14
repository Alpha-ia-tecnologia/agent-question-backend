from langchain_core.prompts import PromptTemplate  
from enum import Enum
import os


class AgentPromptTemplates(Enum):
    SOURCE_PT_TEMPLATE = "source_pt_template"
    AUTHENTIC_PT_TEMPLATE = "authentic_pt_template"
    REAL_TEXT_PT_TEMPLATE = "real_text_pt_template"
    AI_RETRIEVAL_PT_TEMPLATE = "ai_retrieval_pt_template"
    GENERATE_IMAGE_TEMPLATE = "generate_image_template"



def get_prompt(key: AgentPromptTemplates):
    path = os.path.abspath(f"app/prompts/{key.value}.txt")
    with open(path, "r", encoding="utf-8") as file:
        template = file.read()

    return template

    
