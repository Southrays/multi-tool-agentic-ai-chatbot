import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from ai_layer.tools import tools as tools_module
from api_layer.routes import chat as chat_module
from langchain_core.messages import AIMessageChunk


class ConcurrentRAGFakeAgent:
    """Fake agent that uses the actual document retrieval implementation."""

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

        retrieved_content = (
            tools_module.search_uploaded_documents_impl(
                query=user_message,
                thread_id=thread_id,
            )
        )

        response = (
            f"Response for {thread_id}:\n"
            f"{retrieved_content}"
        )

        yield (
            AIMessageChunk(content=response),
            {"langgraph_node": "chatbot"},
        )


@pytest.fixture
def concurrent_rag_agent(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_agent",
        lambda model=None: ConcurrentRAGFakeAgent(),
    )


def parse_sse_events(response_text):
    return [
        json.loads(line[6:])
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def upload_document(client, thread_id, filename, content):
    return client.post(
        "/upload/",
        data={"thread_id": thread_id},
        files={
            "file": (
                filename,
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )


def send_rag_question(client, thread_id, question):
    return client.post(
        "/chat/",
        json={
            "message": question,
            "thread_id": thread_id,
        },
    )


def test_concurrent_rag_requests_are_isolated(
    client,
    fake_vectorstore,
    concurrent_rag_agent,
):
    documents = [
        (
            "rag-conversation-A",
            "project_alpha.txt",
            "Project Alpha uses PostgreSQL for its database.",
            "What database does Project Alpha use?",
            "PostgreSQL",
            "Project Beta",
        ),
        (
            "rag-conversation-B",
            "project_beta.txt",
            "Project Beta uses MongoDB for its database.",
            "What database does Project Beta use?",
            "MongoDB",
            "Project Alpha",
        ),
    ]

    # Upload each document into its own conversation.
    for thread_id, filename, content, *_ in documents:
        response = upload_document(
            client,
            thread_id,
            filename,
            content,
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    # Ask both questions concurrently.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                send_rag_question,
                client,
                thread_id,
                question,
            )
            for (
                thread_id,
                _,
                _,
                question,
                _,
                _,
            ) in documents
        ]

        responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)

    for response, document in zip(responses, documents):
        thread_id = document[0]
        expected_content = document[4]
        forbidden_content = document[5]

        events = parse_sse_events(response.text)

        assert events[0]["thread_id"] == thread_id

        answer = "".join(
            event["token"]
            for event in events
            if "token" in event
        )

        assert expected_content in answer
        assert forbidden_content not in answer

        assert events[-1]["done"] is True
        assert not any("error" in event for event in events)
