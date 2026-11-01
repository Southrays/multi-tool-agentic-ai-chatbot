from unittest.mock import MagicMock

import pytest
from api_layer.main import app
from api_layer.routes import history
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def mock_history(monkeypatch):
    messages = [
        MagicMock(role="user", content="Hello"),
        MagicMock(role="assistant", content="Hello there!"),
        MagicMock(role="user", content="How are you?"),
        MagicMock(role="assistant", content="I'm doing well."),
    ]

    repository_mock = MagicMock(return_value=messages)

    monkeypatch.setattr(
        history,
        "get_chat_history",
        repository_mock,
    )

    return repository_mock


def test_get_chat_history(mock_history):
    thread_id = "test-thread-123"

    response = client.get(f"/history/{thread_id}/")

    assert response.status_code == 200

    data = response.json()

    assert "messages" in data
    assert len(data["messages"]) == 4

    assert data["messages"][0] == {
        "role": "user",
        "content": "Hello",
    }

    assert data["messages"][1] == {
        "role": "assistant",
        "content": "Hello there!",
    }

    assert data["messages"][2] == {
        "role": "user",
        "content": "How are you?",
    }

    assert data["messages"][3] == {
        "role": "assistant",
        "content": "I'm doing well.",
    }

    mock_history.assert_called_once_with(thread_id)


def test_get_chat_history_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        history,
        "get_chat_history",
        lambda thread_id: [],
    )

    response = client.get("/history/empty-thread/")

    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_get_chat_history_uses_requested_thread_id(monkeypatch):
    requested_thread_ids = []

    def fake_get_history(thread_id):
        requested_thread_ids.append(thread_id)
        return []

    monkeypatch.setattr(
        history,
        "get_chat_history",
        fake_get_history,
    )

    client.get("/history/conversation-456/")

    assert requested_thread_ids == ["conversation-456"]


def test_get_chat_history_handles_repository_error(monkeypatch):
    def failing_history(thread_id):
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(
        history,
        "get_chat_history",
        failing_history,
    )

    with pytest.raises(RuntimeError, match="Database unavailable"):
        client.get("/history/test-thread/")
