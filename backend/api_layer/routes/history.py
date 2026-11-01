from database.repositories.message_repository import get_chat_history
from fastapi import APIRouter

router = APIRouter(prefix="/history/{thread_id}", tags=["history_thread_id"])

@router.get("/")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }