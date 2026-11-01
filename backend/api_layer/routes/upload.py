import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from ai_layer.rag.rag import add_document_to_rag
from database.repositories.conversation_repository import create_or_update_conversation
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_document(
    file: Annotated[UploadFile, File()],
    thread_id: Annotated[str, Form()],
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")

        upload_dir = Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / f"{file_id}_{safe_filename}"

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks."
        })

    except (ValueError, KeyError, TypeError) as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )