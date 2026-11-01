from unittest.mock import MagicMock

import pytest
from api_layer.main import app
from api_layer.routes import upload
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def mock_upload_dependencies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Create the uploads directory inside the temporary directory.
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    rag_mock = MagicMock(
        return_value={
            "filename": "document.txt",
            "chunks": 3,
        }
    )

    conversation_mock = MagicMock()

    monkeypatch.setattr(
        upload,
        "add_document_to_rag",
        rag_mock,
    )

    monkeypatch.setattr(
        upload,
        "create_or_update_conversation",
        conversation_mock,
    )

    return {
        "rag": rag_mock,
        "conversation": conversation_mock,
    }


def test_upload_document_success(mock_upload_dependencies):
    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
        files={
            "file": (
                "document.txt",
                b"This is a test document.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "document.txt" in data["message"]
    assert "3 chunks" in data["message"]

    mock_upload_dependencies["conversation"].assert_called_once_with(
        "thread-123",
        "Uploaded document",
    )

    mock_upload_dependencies["rag"].assert_called_once()

    rag_args = mock_upload_dependencies["rag"].call_args.kwargs

    assert rag_args["thread_id"] == "thread-123"


def test_upload_document_unsupported_extension(
    mock_upload_dependencies,
):
    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
        files={
            "file": (
                "document.exe",
                b"Invalid file",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert "Unsupported file type" in data["message"]

    mock_upload_dependencies["rag"].assert_not_called()
    mock_upload_dependencies["conversation"].assert_not_called()


def test_upload_document_without_thread_id():
    response = client.post(
        "/upload/",
        files={
            "file": (
                "document.txt",
                b"Test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


def test_upload_document_without_file():
    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
    )

    assert response.status_code == 422


def test_upload_document_rag_failure(mock_upload_dependencies):
    mock_upload_dependencies["rag"].side_effect = ValueError(
        "No text could be extracted from this file."
    )

    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
        files={
            "file": (
                "document.txt",
                b"Test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["success"] is False
    assert "No text could be extracted" in data["message"]


def test_upload_document_handles_rag_key_error(
    mock_upload_dependencies,
):
    mock_upload_dependencies["rag"].side_effect = KeyError(
        "filename"
    )

    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
        files={
            "file": (
                "document.txt",
                b"Test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["success"] is False


def test_upload_document_handles_rag_type_error(
    mock_upload_dependencies,
):
    mock_upload_dependencies["rag"].side_effect = TypeError(
        "Invalid document data"
    )

    response = client.post(
        "/upload/",
        data={"thread_id": "thread-123"},
        files={
            "file": (
                "document.txt",
                b"Test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["success"] is False
