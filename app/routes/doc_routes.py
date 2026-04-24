from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from http import HTTPStatus
from typing import Any, Dict, List
import os
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


_EXPORT_STR_DEFAULTS = (
    "title", "text", "source", "source_url", "source_author",
    "proficiency_description", "explanation_question",
    "id_skill", "skill", "proficiency_level", "question_statement",
    "correct_answer",
)


def _sanitize_export_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza uma questão recebida do cliente para a exportação:
    - campos de texto None → ""
    - question_number None → 0
    - alternatives: garante list[dict] com pelo menos letter/text
    """
    clean = dict(q or {})
    for field in _EXPORT_STR_DEFAULTS:
        if clean.get(field) is None:
            clean[field] = ""
    if clean.get("question_number") is None:
        clean["question_number"] = 0

    alts = clean.get("alternatives") or []
    clean_alts = []
    for alt in alts:
        if not isinstance(alt, dict):
            continue
        clean_alts.append({
            "letter": alt.get("letter") or "",
            "text": alt.get("text") or "",
            "distractor": alt.get("distractor"),
        })
    clean["alternatives"] = clean_alts
    return clean


@doc_router.post("/generate-docx", status_code=HTTPStatus.OK, response_model=GenerateDocxResponseSchema)
def export_docx(
    questions: List[Dict[str, Any]],
    file_name: str,
    version: str = "teacher",
):
    """
        Endpoint responsável por gerar um docx de questões.\n
        Recebe uma lista de questões (dicts flexíveis — campos nulos são
        tolerados, já que questões persistidas no banco podem ter colunas
        NULL como `source` ou `proficiency_description`).\n
        Retorna um link para download do arquivo docx.

        `version`:
        - `teacher`: inclui gabarito, explicação e distratores (padrão).
        - `student`: apenas enunciado e alternativas, sem gabarito.
    """
    if not questions:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Questions list cannot be empty.")
    if not file_name:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="File name cannot be empty.")
    if version not in ("student", "teacher"):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid version. Use 'student' or 'teacher'.")
    try:
        sanitized = [_sanitize_export_question(q) for q in questions]
        GenerateDocxService.generate_docx(questions=sanitized, file_name=file_name, version=version)
        return {
            "message": "Document generated successfully",
            "link": f"doc/download/{file_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))
    
    