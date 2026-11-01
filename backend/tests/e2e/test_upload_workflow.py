def test_complete_document_upload_workflow(
    client,
    fake_vectorstore,
):
    thread_id = "e2e-document-conversation"

    document_content = (
        "FastAPI is a Python framework for building APIs. "
        "It supports asynchronous endpoints and automatic "
        "API documentation."
    )

    # 1. Upload a document through the actual HTTP endpoint.
    upload_response = client.post(
        "/upload/",
        data={
            "thread_id": thread_id,
        },
        files={
            "file": (
                "fastapi_notes.txt",
                document_content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    upload_data = upload_response.json()

    assert upload_data["success"] is True
    assert "fastapi_notes.txt" in upload_data["message"]
    assert "chunks" in upload_data["message"]

    # 2. Verify that the document was added to the vector store.
    assert len(fake_vectorstore.documents) > 0

    uploaded_document = fake_vectorstore.documents[0]

    assert uploaded_document.page_content
    assert uploaded_document.metadata["thread_id"] == thread_id
    assert uploaded_document.metadata["source"].endswith("_fastapi_notes.txt")

    # 3. Verify that the uploaded document created a conversation.
    conversations_response = client.get("/conversations/")

    assert conversations_response.status_code == 200

    conversations = conversations_response.json()["conversations"]

    uploaded_conversation = next(
        (
            conversation
            for conversation in conversations
            if conversation["thread_id"] == thread_id
        ),
        None,
    )

    assert uploaded_conversation is not None
    assert uploaded_conversation["title"] == "Uploaded document"

    # 4. Verify that the conversation has no chat messages yet.
    history_response = client.get(
        f"/history/{thread_id}/"
    )

    assert history_response.status_code == 200
    assert history_response.json()["messages"] == []


def test_document_upload_rejects_unsupported_file_type(client):
    response = client.post(
        "/upload/",
        data={
            "thread_id": "unsupported-file-test",
        },
        files={
            "file": (
                "malicious.exe",
                b"test content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    assert response.json()["success"] is False
    assert "Unsupported file type" in response.json()["message"]


def test_uploaded_documents_are_isolated_by_conversation(
    client,
    fake_vectorstore,
):
    first_thread = "conversation-one"
    second_thread = "conversation-two"

    # Upload a document to the first conversation.
    first_upload = client.post(
        "/upload/",
        data={"thread_id": first_thread},
        files={
            "file": (
                "first.txt",
                b"Content belonging to conversation one.",
                "text/plain",
            )
        },
    )

    assert first_upload.status_code == 200

    # Upload another document to the second conversation.
    second_upload = client.post(
        "/upload/",
        data={"thread_id": second_thread},
        files={
            "file": (
                "second.txt",
                b"Content belonging to conversation two.",
                "text/plain",
            )
        },
    )

    assert second_upload.status_code == 200

    # Verify the vector store keeps both documents.
    assert len(fake_vectorstore.documents) == 2

    # Verify retrieval filtering by thread_id.
    first_results = fake_vectorstore.similarity_search(
        "content",
        filter={"thread_id": first_thread},
    )

    second_results = fake_vectorstore.similarity_search(
        "content",
        filter={"thread_id": second_thread},
    )

    assert len(first_results) == 1
    assert len(second_results) == 1

    assert first_results[0].metadata["source"].endswith("_first.txt")
    assert second_results[0].metadata["source"].endswith("_second.txt")