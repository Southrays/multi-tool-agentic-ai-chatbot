
from unittest.mock import MagicMock

import pytest
from api_layer.main import app
from api_layer.routes import chat
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

client = TestClient(app)


@pytest.fixture
def mock_agent(monkeypatch):
    agent = MagicMock()

    def fake_stream(*args, **kwargs):
        yield (
            AIMessageChunk(content="Hello"),
            {"langgraph_node": "model"},
        )

        yield (
            AIMessageChunk(content=" there!"),
            {"langgraph_node": "model"},
        )

    agent.stream = fake_stream

    monkeypatch.setattr(
        chat,
        "get_agent",
        lambda model_name: agent,
    )

    return agent


def test_chat_creates_new_thread_id(mock_agent):
    response = client.post(
        "/chat/",
        json={"message": "Hello"},
    )

    assert response.status_code == 200

    assert '"thread_id"' in response.text
    assert '"token":"Hello"' in response.text
    assert '"token":" there!"' in response.text
    assert '"done":true' in response.text


def test_chat_saves_user_and_assistant_messages(mock_agent, monkeypatch):
    saved_messages = []

    monkeypatch.setattr(
        chat,
        "save_chat_message",
        lambda thread_id, role, content: saved_messages.append(
            {
                "thread_id": thread_id,
                "role": role,
                "content": content,
            }
        ),
    )

    response = client.post(
        "/chat/",
        json={"message": "Hello"},
    )

    assert response.status_code == 200

    assert len(saved_messages) == 2

    assert saved_messages[0]["role"] == "user"
    assert saved_messages[0]["content"] == "Hello"

    assert saved_messages[1]["role"] == "assistant"
    assert saved_messages[1]["content"] == "Hello there!"



def test_chat_handles_agent_failure(mock_agent):
    def failing_stream(*args, **kwargs):
        raise RuntimeError("Gemini API unavailable")
        yield

    mock_agent.stream = failing_stream

    response = client.post(
        "/chat/",
        json={"message": "Hello"},
    )

    assert response.status_code == 200

    assert (
        "The AI service could not complete your request."
        in response.text
    )

    assert '"done":true' in response.text

    assert "Gemini API unavailable" not in response.text

