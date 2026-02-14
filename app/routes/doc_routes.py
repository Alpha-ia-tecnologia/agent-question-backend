from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from http import HTTPStatus
import os
from app.schemas.question_schema import QuestionSchema, QuestionWithImageSchema
from app.schemas.generate_docx_response_schema import GenerateDocxResponseSchema
from app.services.generate_docx_service import GenerateDocxService

doc_router = APIRouter(prefix="/doc")

@doc_router.get("/download/{file_name}", status_code=HTTPStatus.OK, response_class=FileResponse)
async def download_file(file_name: str):
    # Evita path traversal
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid file name.")

    file_path = os.path.abspath(f"export/{file_name}.docx")
    print(file_path)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="File not found.")

    return FileResponse(
        file_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename=f"{file_name}"
    )


@doc_router.post("/generate-docx", status_code=HTTPStatus.OK, response_model=GenerateDocxResponseSchema)
def export_docx(questions: list[QuestionSchema | QuestionWithImageSchema], file_name: str):
    """
        Endpoint responsável por gera um docx de questões.\n
        Recebe uma lista de questões.\n
        Retorna um link para download do arquivo docx.

        Aceita umas lista de questões com imagens, sem imagens e ambas
        
    """
    if not questions:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Questions list cannot be empty.")
    if not file_name:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="File name cannot be empty.")
    try:
        GenerateDocxService.generate_docx(questions=questions, file_name=file_name)
        return {
        "message": "Document generated successfully",
        "link": f"doc/download/{file_name}"
    }
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))
    
    