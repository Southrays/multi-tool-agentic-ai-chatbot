import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from api_layer.routes import chat as chat_module
from langchain_core.messages import AIMessageChunk


class ConcurrentFakeAgent:
    """Fake agent that identifies the conversation producing each response."""

    def stream(
        self,
        inputs,
        config=None,
        context=None,
        stream_mode=None,
    ):
        if context is None:
            raise ValueError("Agent context was not provided.")

        thread_id = context["thread_id"]
        user_message = inputs["messages"][0].content

        response = f"Response for {thread_id}: {user_message}"

        yield (
            AIMessageChunk(content=response),
            {"langgraph_node": "chatbot"},
        )


@pytest.fixture
def concurrent_fake_agent(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: ConcurrentFakeAgent(),
    )


def parse_sse_events(response_text):
    return [
        json.loads(line[6:])
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def send_chat_request(client, thread_id, message):
    return client.post(
        "/chat/",
        json={
            "message": message,
            "thread_id": thread_id,
        },
    )


def test_concurrent_chat_requests_are_isolated(
    client,
    concurrent_fake_agent,
):
    conversations = [
        ("conversation-A", "Message from A"),
        ("conversation-B", "Message from B"),
        ("conversation-C", "Message from C"),
        ("conversation-D", "Message from D"),
        ("conversation-E", "Message from E"),
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                send_chat_request,
                client,
                thread_id,
                message,
            )
            for thread_id, message in conversations
        ]

        responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)

    for response, (thread_id, message) in zip(
        responses,
        conversations,
    ):
        events = parse_sse_events(response.text)

        assert events[0]["thread_id"] == thread_id

        answer = "".join(
            event["token"]
            for event in events
            if "token" in event
        )

        assert answer == f"Response for {thread_id}: {message}"
        assert events[-1]["done"] is True
        assert not any("error" in event for event in events)

        history_response = client.get(f"/history/{thread_id}/")

        assert history_response.status_code == 200

        history = history_response.json()["messages"]

        assert len(history) == 2
        assert history[0]["content"] == message
        assert history[1]["content"] == answer
