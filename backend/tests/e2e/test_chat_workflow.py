import json

import pytest
from api_layer.routes import chat as chat_module
from langchain_core.messages import AIMessageChunk


class FakeAgent:
    """Predictable substitute for the real LangGraph agent."""

    def stream(self, inputs, config=None, context=None, stream_mode=None):
        message = inputs["messages"][0]
        user_message = message.content

        response = f"Test response to: {user_message}"

        yield (
            AIMessageChunk(content=response),
            {"langgraph_node": "chatbot"},
        )


@pytest.fixture
def mock_agent(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: FakeAgent(),
    )


def parse_sse_events(response_text):
    """Convert SSE data events into Python dictionaries."""

    events = []

    for line in response_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    return events


def test_complete_chat_workflow(client, mock_agent):
    # 1. Start a new conversation.
    first_response = client.post(
        "/chat/",
        json={
            "message": "Hello, chatbot!"
        },
    )

    assert first_response.status_code == 200
    assert first_response.headers["content-type"].startswith(
        "text/event-stream"
    )

    first_events = parse_sse_events(first_response.text)

    # 2. Verify the conversation ID was returned.
    thread_id = first_events[0]["thread_id"]

    assert thread_id

    # 3. Verify the streamed AI response.
    first_tokens = [
        event["token"]
        for event in first_events
        if "token" in event
    ]

    assert "".join(first_tokens) == (
        "Test response to: Hello, chatbot!"
    )

    assert first_events[-1]["done"] is True

    # 4. Continue the same conversation.
    second_response = client.post(
        "/chat/",
        json={
            "message": "What did I just say?",
            "thread_id": thread_id,
        },
    )

    assert second_response.status_code == 200

    second_events = parse_sse_events(second_response.text)

    assert second_events[0]["thread_id"] == thread_id

    # 5. Retrieve conversation history.
    history_response = client.get(
        f"/history/{thread_id}/"
    )

    assert history_response.status_code == 200

    history = history_response.json()["messages"]

    assert len(history) == 4

    assert history[0] == {
        "role": "user",
        "content": "Hello, chatbot!",
    }

    assert history[1] == {
        "role": "assistant",
        "content": "Test response to: Hello, chatbot!",
    }

    assert history[2] == {
        "role": "user",
        "content": "What did I just say?",
    }

    assert history[3] == {
        "role": "assistant",
        "content": (
            "Test response to: What did I just say?"
        ),
    }

    # 6. Retrieve the conversation list.
    conversations_response = client.get(
        "/conversations/"
    )

    assert conversations_response.status_code == 200

    conversations = conversations_response.json()["conversations"]

    assert len(conversations) == 1
    assert conversations[0]["thread_id"] == thread_id
    assert conversations[0]["title"] == "Hello, chatbot!"


def test_chat_rejects_empty_message(client, mock_agent):
    response = client.post(
        "/chat/",
        json={
            "message": "   "
        },
    )

    assert response.status_code == 422


def test_chat_rejects_message_over_10000_characters(
    client,
    mock_agent,
):
    response = client.post(
        "/chat/",
        json={
            "message": "a" * 10001
        },
    )

    assert response.status_code == 422