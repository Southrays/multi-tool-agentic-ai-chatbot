import json

from api_layer.routes import chat as chat_module
from langchain_core.messages import AIMessageChunk


class FakeAgent:
    def stream(self, inputs, config, context, stream_mode):
        yield (
            AIMessageChunk(content="Hello "),
            {"langgraph_node": "chatbot"},
        )

        yield (
            AIMessageChunk(content="from the integration test."),
            {"langgraph_node": "chatbot"},
        )


def parse_sse_events(response_text):
    events = []

    for block in response_text.strip().split("\n\n"):
        if block.startswith("data: "):
            events.append(json.loads(block[6:]))

    return events


def test_chat_persists_messages_and_returns_history(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: FakeAgent(),
    )

    response = client.post(
        "/chat/",
        json={
            "message": "Hello, chatbot!",
            "model": "gemini-3.8-flash",
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = parse_sse_events(response.text)

    thread_event = next(
        event for event in events if event.get("thread_id")
    )

    thread_id = thread_event["thread_id"]

    tokens = [
        event["token"]
        for event in events
        if event.get("token")
    ]

    assert "".join(tokens) == (
        "Hello from the integration test."
    )

    assert any(event.get("done") for event in events)

    history_response = client.get(
        f"/history/{thread_id}/"
    )

    assert history_response.status_code == 200

    messages = history_response.json()["messages"]

    assert len(messages) == 2

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, chatbot!"

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == (
        "Hello from the integration test."
    )

    conversations_response = client.get("/conversations/")

    assert conversations_response.status_code == 200

    conversations = conversations_response.json()["conversations"]

    assert len(conversations) == 1
    assert conversations[0]["thread_id"] == thread_id
    assert conversations[0]["title"] == "Hello, chatbot!"


def test_chat_reuses_existing_conversation(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: FakeAgent(),
    )

    thread_id = "existing-thread-123"

    first_response = client.post(
        "/chat/",
        json={
            "message": "First message",
            "thread_id": thread_id,
        },
    )

    second_response = client.post(
        "/chat/",
        json={
            "message": "Second message",
            "thread_id": thread_id,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    history_response = client.get(
        f"/history/{thread_id}/"
    )

    messages = history_response.json()["messages"]

    assert len(messages) == 4

    assert messages[0]["content"] == "First message"
    assert messages[1]["role"] == "assistant"

    assert messages[2]["content"] == "Second message"
    assert messages[3]["role"] == "assistant"

    conversations = client.get("/conversations/").json()[
        "conversations"
    ]

    assert len(conversations) == 1
