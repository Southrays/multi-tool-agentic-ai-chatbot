from unittest.mock import MagicMock

import pytest
from ai_layer.rag import rag
from langchain_core.documents import Document


def test_read_txt_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a test document.", encoding="utf-8")

    result = rag.read_file_text(str(file_path))

    assert result == "This is a test document."


def test_read_markdown_file(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Test heading", encoding="utf-8")

    result = rag.read_file_text(str(file_path))

    assert result == "# Test heading"


def test_read_python_file(tmp_path):
    file_path = tmp_path / "script.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    result = rag.read_file_text(str(file_path))

    assert result == "print('hello')"


def test_read_csv_file(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("name,age\nJohn,25", encoding="utf-8")

    result = rag.read_file_text(str(file_path))

    assert result == "name,age\nJohn,25"


def test_read_docx_file(monkeypatch, tmp_path):
    file_path = tmp_path / "document.docx"
    file_path.touch()

    monkeypatch.setattr(
        rag.docx2txt,
        "process",
        lambda path: "DOCX document content",
    )

    result = rag.read_file_text(str(file_path))

    assert result == "DOCX document content"


def test_read_pdf_file(monkeypatch, tmp_path):
    file_path = tmp_path / "document.pdf"
    file_path.touch()

    class FakePage:
        def extract_text(self):
            return "PDF page content"

    class FakePdfReader:
        def __init__(self, path):
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr(rag, "PdfReader", FakePdfReader)

    result = rag.read_file_text(str(file_path))

    assert result == "PDF page content\nPDF page content\n"


def test_reject_unsupported_file_type(tmp_path):
    file_path = tmp_path / "document.exe"
    file_path.touch()

    with pytest.raises(ValueError, match="Unsupported file type"):
        rag.read_file_text(str(file_path))


def test_add_document_to_rag(monkeypatch, tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a test document.", encoding="utf-8")

    fake_vectorstore = MagicMock()

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    result = rag.add_document_to_rag(
        file_path=str(file_path),
        thread_id="thread-123",
    )

    assert result["filename"] == "notes.txt"
    assert result["chunks"] == 1

    fake_vectorstore.add_documents.assert_called_once()

    saved_docs = fake_vectorstore.add_documents.call_args.args[0]

    assert len(saved_docs) == 1
    assert isinstance(saved_docs[0], Document)
    assert saved_docs[0].page_content == "This is a test document."
    assert saved_docs[0].metadata["thread_id"] == "thread-123"
    assert saved_docs[0].metadata["source"] == "notes.txt"


def test_add_document_rejects_empty_file(monkeypatch, tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    fake_vectorstore = MagicMock()

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    with pytest.raises(ValueError, match="No text could be extracted"):
        rag.add_document_to_rag(
            file_path=str(file_path),
            thread_id="thread-123",
        )

    fake_vectorstore.add_documents.assert_not_called()


def test_retrieve_from_rag(monkeypatch):
    fake_vectorstore = MagicMock()

    fake_vectorstore.similarity_search.return_value = [
        Document(
            page_content="Relevant document content.",
            metadata={
                "thread_id": "thread-123",
                "source": "notes.pdf",
            },
        )
    ]

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    result = rag.retrieve_from_rag(
        query="What is in the document?",
        thread_id="thread-123",
    )

    assert "[Source 1: notes.pdf]" in result
    assert "Relevant document content." in result

    fake_vectorstore.similarity_search.assert_called_once_with(
        "What is in the document?",
        k=4,
        filter={"thread_id": "thread-123"},
    )


def test_retrieve_from_rag_returns_empty_message(monkeypatch):
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search.return_value = []

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    result = rag.retrieve_from_rag(
        query="Missing information",
        thread_id="thread-123",
    )

    assert result == "No relevant uploaded document content found."


def test_retrieve_from_rag_respects_custom_k(monkeypatch):
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search.return_value = []

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    rag.retrieve_from_rag(
        query="Search query",
        thread_id="thread-123",
        k=2,
    )

    fake_vectorstore.similarity_search.assert_called_once_with(
        "Search query",
        k=2,
        filter={"thread_id": "thread-123"},
    )
