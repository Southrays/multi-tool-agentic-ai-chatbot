from uuid import uuid4

from ai_layer.agent.agent import get_agent
from database.repositories.conversation_repository import (
    create_or_update_conversation,
)
from database.repositories.message_repository import save_chat_message
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from api_layer.monitoring.logger import get_logger
from api_layer.schemas.chat import ChatRequest, ChatStreamEvent

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def sse_data(event: ChatStreamEvent) -> str:
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    return not additional_kwargs.get("tool_calls")


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])

                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""


@router.post("/", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid4())

    agent = get_agent(request.model)

    create_or_update_conversation(
        thread_id,
        request.message,
    )

    save_chat_message(
        thread_id,
        "user",
        request.message,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    def event_generator():
        final_answer = ""

        # Tell the frontend which conversation this stream belongs to.
        yield sse_data(
            ChatStreamEvent(thread_id=thread_id)
        )

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=request.message)
                ]
            }

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                context={"thread_id": thread_id},
                stream_mode="messages",
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token

                    yield sse_data(
                        ChatStreamEvent(token=token)
                    )

            if final_answer.strip():
                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer,
                )

            yield sse_data(
                ChatStreamEvent(done=True)
            )

        except Exception:
            logger.exception(
                "Chat streaming failed for thread %s",
                thread_id,
            )

            yield sse_data(
                ChatStreamEvent(
                    error="The AI service could not complete your request."
                )
            )

            yield sse_data(
                ChatStreamEvent(done=True)
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
