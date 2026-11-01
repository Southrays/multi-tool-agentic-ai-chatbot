import math

from database.repositories.memory_repository import save_memory, search_memory
from dotenv import load_dotenv
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch

from ai_layer.schemas.context import AgentContext

load_dotenv()

def get_thread_id(runtime: ToolRuntime[AgentContext]) -> str:
    if runtime.context is None:
        raise ValueError("Agent context was not provided.")

    return runtime.context["thread_id"]


def remember_this_impl(memory: str, thread_id: str) -> str:
    return save_memory(
        thread_id=thread_id,
        memory=memory,
    )


def recall_memory_impl(query: str, thread_id: str) -> str:
    return search_memory(
        thread_id=thread_id,
        query=query,
    )


def search_uploaded_documents_impl(
    query: str,
    thread_id: str,
) -> str:
    from ai_layer.rag.rag import retrieve_from_rag

    return retrieve_from_rag(
        query=query,
        thread_id=thread_id,
    )



web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except (ValueError, ZeroDivisionError, TypeError) as e:
        return f"Calculation error: {e!s}"


@tool
def search_uploaded_documents(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return search_uploaded_documents_impl(
        query=query,
        thread_id=get_thread_id(runtime),
    )


@tool
def remember_this(memory: str, runtime: ToolRuntime[AgentContext]) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return remember_this_impl(
        memory=memory,
        thread_id=get_thread_id(runtime),
    )

@tool
def recall_memory(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return recall_memory_impl(
        query=query,
        thread_id=get_thread_id(runtime),
    )


tools = [
    calculator,
    recall_memory,
    remember_this,
    search_uploaded_documents,
    web_search
]