from database.models import Conversation
from database.repositories.conversation_repository import (
    create_or_update_conversation,
    list_conversations,
)


def test_create_conversation(test_db):
    create_or_update_conversation(
        "thread-1",
        "Hello, how are you?",
    )

    db = test_db()

    try:
        conversation = (
            db.query(Conversation)
            .filter_by(thread_id="thread-1")
            .first()
        )

        assert conversation is not None
        assert conversation.thread_id == "thread-1"
        assert conversation.title == "Hello, how are you?"
    finally:
        db.close()


def test_create_conversation_without_message(test_db):
    create_or_update_conversation("thread-2")

    db = test_db()

    try:
        conversation = (
            db.query(Conversation)
            .filter_by(thread_id="thread-2")
            .first()
        )

        assert conversation.title == "New Chat"
    finally:
        db.close()


def test_conversation_title_is_limited_to_40_characters(test_db):
    message = "A" * 60

    create_or_update_conversation("thread-3", message)

    db = test_db()

    try:
        conversation = (
            db.query(Conversation)
            .filter_by(thread_id="thread-3")
            .first()
        )

        assert conversation.title == ("A" * 40) + "..."
    finally:
        db.close()


def test_existing_conversation_is_updated_not_duplicated(test_db):
    create_or_update_conversation("thread-4", "First message")
    create_or_update_conversation("thread-4", "Second message")

    db = test_db()

    try:
        conversations = (
            db.query(Conversation)
            .filter_by(thread_id="thread-4")
            .all()
        )

        assert len(conversations) == 1
        assert conversations[0].title == "First message"
    finally:
        db.close()


def test_list_conversations(test_db):
    create_or_update_conversation("thread-5", "First chat")
    create_or_update_conversation("thread-6", "Second chat")

    conversations = list_conversations()

    assert len(conversations) == 2
    assert {item.thread_id for item in conversations} == {
        "thread-5",
        "thread-6",
    }
