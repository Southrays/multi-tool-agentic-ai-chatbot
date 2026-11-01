from ai_layer.rag import rag


def test_upload_document_and_retrieve_from_rag(
    client,
    fake_vectorstore,
):
    thread_id = "rag-integration-thread"

    document_content = (
        "FastAPI is a Python framework for building APIs. "
        "LangGraph is used to build stateful agent workflows. "
        "Retrieval augmented generation allows an AI system "
        "to retrieve relevant information from external documents."
    )

    response = client.post(
        "/upload/",
        data={"thread_id": thread_id},
        files={
            "file": (
                "integration_document.txt",
                document_content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["success"] is True
    assert "integration_document.txt" in result["message"]
    assert "chunks" in result["message"]

    # Confirm the actual document was added to the vector store.
    assert len(fake_vectorstore.documents) > 0

    uploaded_documents = [
        doc
        for doc in fake_vectorstore.documents
        if doc.metadata["thread_id"] == thread_id
    ]

    assert len(uploaded_documents) > 0

    # Confirm the real retrieval function can find the content.
    retrieved_content = rag.retrieve_from_rag(
        query="What is FastAPI?",
        thread_id=thread_id,
    )

    assert "FastAPI" in retrieved_content
    assert "integration_document.txt" in retrieved_content


def test_rag_does_not_return_another_threads_documents(
    client,
    fake_vectorstore,
):
    content = (
        "This document contains confidential project information "
        "about the architecture of an AI chatbot."
    )

    client.post(
        "/upload/",
        data={"thread_id": "thread-one"},
        files={
            "file": (
                "project.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    result = rag.retrieve_from_rag(
        query="AI chatbot architecture",
        thread_id="thread-two",
    )

    assert result == "No relevant uploaded document content found."


def test_upload_creates_conversation(
    client,
):
    thread_id = "upload-conversation-thread"

    response = client.post(
        "/upload/",
        data={"thread_id": thread_id},
        files={
            "file": (
                "notes.txt",
                b"Some integration test notes.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    conversations = client.get("/conversations/").json()[
        "conversations"
    ]

    matching_conversation = next(
        item
        for item in conversations
        if item["thread_id"] == thread_id
    )

    assert matching_conversation["title"] == "Uploaded document"
