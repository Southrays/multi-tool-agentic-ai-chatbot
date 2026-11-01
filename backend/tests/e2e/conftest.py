import pytest
from fastapi.testclient import TestClient


class FakeVectorStore:
    """In-memory replacement for Chroma during E2E tests."""

    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search(self, query, k=4, filter=None):
        results = self.documents

        if filter:
            results = [
                doc
                for doc in results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]

        return results[:k]


@pytest.fixture
def fake_vectorstore():
    return FakeVectorStore()


@pytest.fixture
def client(test_db, monkeypatch, tmp_path, fake_vectorstore):
    """Run the actual FastAPI application with isolated test dependencies."""

    monkeypatch.chdir(tmp_path)

    from ai_layer.rag import rag
    from api_layer.main import app

    monkeypatch.setattr(
        rag,
        "get_vectorstore",
        lambda: fake_vectorstore,
    )

    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)

    with TestClient(app) as test_client:
        yield test_client