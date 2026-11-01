
from database.repositories.message_repository import (
    get_chat_history,
    save_chat_message,
)


def test_save_chat_message(test_db):
    save_chat_message(
        "thread-1",
        "user",
        "Hello Chatty GPT",
    )

    messages = get_chat_history("thread-1")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello Chatty GPT"


def test_save_multiple_messages(test_db):
    save_chat_message("thread-1", "user", "Hello")
    save_chat_message("thread-1", "assistant", "Hi there!")

    messages = get_chat_history("thread-1")

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_messages_are_separated_by_thread_id(test_db):
    save_chat_message("thread-1", "user", "Message A")
    save_chat_message("thread-2", "user", "Message B")

    messages_a = get_chat_history("thread-1")
    messages_b = get_chat_history("thread-2")

    assert len(messages_a) == 1
    assert len(messages_b) == 1

    assert messages_a[0].content == "Message A"
    assert messages_b[0].content == "Message B"


def test_empty_history_returns_empty_list(test_db):
    messages = get_chat_history("nonexistent-thread")

    assert messages == []
