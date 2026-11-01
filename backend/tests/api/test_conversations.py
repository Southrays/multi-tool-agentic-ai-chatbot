from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from api_layer.main import app
from api_layer.routes import conversations
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def mock_conversations(monkeypatch):
    items = [
        MagicMock(
            thread_id="thread-1",
            title="First conversation",
            created_at=datetime(2026, 10, 30, 10, 0, tzinfo=UTC),
            updated_at=datetime(2026, 10, 30, 11, 0, tzinfo=UTC),
        ),
        MagicMock(
            thread_id="thread-2",
            title="Second conversation",
            created_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 10, 31, 13, 0, tzinfo=UTC),
        ),
    ]

    repository_mock = MagicMock(return_value=items)

    monkeypatch.setattr(
        conversations,
        "list_conversations",
        repository_mock,
    )

    return repository_mock


def test_get_conversations(mock_conversations):
    response = client.get("/conversations/")

    assert response.status_code == 200

    data = response.json()

    assert "conversations" in data
    assert len(data["conversations"]) == 2

    first = data["conversations"][0]

    assert first["thread_id"] == "thread-1"
    assert first["title"] == "First conversation"
    assert first["created_at"] == "2026-10-30T10:00:00+00:00"
    assert first["updated_at"] == "2026-10-30T11:00:00+00:00"

    second = data["conversations"][1]

    assert second["thread_id"] == "thread-2"
    assert second["title"] == "Second conversation"

    mock_conversations.assert_called_once_with()


def test_get_conversations_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        conversations,
        "list_conversations",
        list,
    )

    response = client.get("/conversations/")

    assert response.status_code == 200
    assert response.json() == {"conversations": []}


def test_get_conversations_handles_repository_error(monkeypatch):
    def failing_repository():
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(
        conversations,
        "list_conversations",
        failing_repository,
    )

    with pytest.raises(RuntimeError, match="Database unavailable"):
        client.get("/conversations/")
