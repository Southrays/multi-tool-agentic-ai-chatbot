import pytest
from database.database import Base
from database.repositories import (
    conversation_repository,
    memory_repository,
    message_repository,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create an isolated file-backed SQLite database for each test."""

    database_path = tmp_path / "test.db"

    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
    )

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=engine)

    # Redirect repository operations to the isolated test database.
    monkeypatch.setattr(
        conversation_repository,
        "SessionLocal",
        TestingSessionLocal,
    )

    monkeypatch.setattr(
        message_repository,
        "SessionLocal",
        TestingSessionLocal,
    )

    monkeypatch.setattr(
        memory_repository,
        "SessionLocal",
        TestingSessionLocal,
    )

    yield TestingSessionLocal

    Base.metadata.drop_all(bind=engine)
    engine.dispose()