import json

import pytest
from ai_layer.tools import tools as tools_module
from api_layer.routes import chat as chat_module
from langchain_core.messages import AIMessageChunk


class FakeRAGAgent:
    """Simulate an agent that invokes the actual document retrieval tool."""

    def stream(self, inputs, config=None, context=None, stream_mode=None):
        if context is None:
            raise ValueError(
                "Agent context was not provided."
            )

        thread_id = context["thread_id"]

        user_message = inputs["messages"][0].content

        retrieved_content = (
            tools_module.search_uploaded_documents_impl(
                query=user_message,
                thread_id=thread_id,
            )
        )

        response = (
            "Here is what I found in your uploaded document:\n"
            f"{retrieved_content}"
        )

        yield (
            AIMessageChunk(content=response),
            {"langgraph_node": "chatbot"},
        )



@pytest.fixture
def mock_rag_agent(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: FakeRAGAgent(),
    )


def parse_sse_events(response_text):
    events = []

    for line in response_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    return events


def test_complete_rag_question_answering_workflow(
    client,
    fake_vectorstore,
    mock_rag_agent,
):
    thread_id = "rag-e2e-conversation"

    document_content = (
        "FastAPI is a modern Python framework for building APIs. "
        "It supports asynchronous endpoints, automatic validation, "
        "and interactive API documentation through Swagger UI."
    )

    # 1. Upload a document to the conversation.
    upload_response = client.post(
        "/upload/",
        data={"thread_id": thread_id},
        files={
            "file": (
                "fastapi_guide.txt",
                document_content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["success"] is True

    # 2. Confirm the document entered the vector store.
    assert len(fake_vectorstore.documents) > 0

    # 3. Ask a question through the actual chat endpoint.
    chat_response = client.post(
        "/chat/",
        json={
            "message": "What does the document say about FastAPI?",
            "thread_id": thread_id,
        },
    )

    assert chat_response.status_code == 200
    assert chat_response.headers["content-type"].startswith(
        "text/event-stream"
    )

    events = parse_sse_events(chat_response.text)

    # 4. Confirm the chat belongs to the uploaded document's conversation.
    assert events[0]["thread_id"] == thread_id

    # 5. Verify that the streamed answer contains retrieved document content.
    tokens = [
        event["token"]
        for event in events
        if "token" in event
    ]

    answer = "".join(tokens)

    assert "FastAPI" in answer
    assert "asynchronous endpoints" in answer
    assert "Swagger UI" in answer

    # 6. Verify the stream completed successfully.
    assert events[-1]["done"] is True
    assert not any("error" in event for event in events)

    # 7. Verify that the user question and assistant answer were persisted.
    history_response = client.get(
        f"/history/{thread_id}/"
    )

    assert history_response.status_code == 200

    messages = history_response.json()["messages"]

    assert len(messages) == 2

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == (
        "What does the document say about FastAPI?"
    )

    assert messages[1]["role"] == "assistant"
    assert "Swagger UI" in messages[1]["content"]

    # 8. Verify that another conversation cannot retrieve this document.
    other_results = fake_vectorstore.similarity_search(
        "FastAPI",
        filter={"thread_id": "another-conversation"},
    )

    assert other_results == []