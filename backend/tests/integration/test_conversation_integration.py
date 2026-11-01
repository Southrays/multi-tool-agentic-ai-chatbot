from database.repositories.conversation_repository import (
    create_or_update_conversation,
    list_conversations,
)
from database.repositories.message_repository import (
    get_chat_history,
    save_chat_message,
)


def test_conversation_and_message_persistence(test_db):
    thread_id = "repository-integration-thread"

    create_or_update_conversation(
        thread_id,
        "Testing persistent conversation storage",
    )

    save_chat_message(
        thread_id,
        "user",
        "First message",
    )

    save_chat_message(
        thread_id,
        "assistant",
        "First response",
    )

    save_chat_message(
        thread_id,
        "user",
        "Second message",
    )

    history = get_chat_history(thread_id)

    assert len(history) == 3

    assert history[0].role == "user"
    assert history[0].content == "First message"

    assert history[1].role == "assistant"
    assert history[1].content == "First response"

    assert history[2].role == "user"
    assert history[2].content == "Second message"

    conversations = list_conversations()

    assert len(conversations) == 1

    conversation = conversations[0]

    assert conversation.thread_id == thread_id
    assert conversation.title == (
        "Testing persistent conversation storage"
    )


def test_updating_conversation_does_not_create_duplicate(test_db):
    thread_id = "duplicate-check-thread"

    create_or_update_conversation(
        thread_id,
        "Original title",
    )

    create_or_update_conversation(
        thread_id,
        "A different title",
    )

    conversations = list_conversations()

    matching = [
        item
        for item in conversations
        if item.thread_id == thread_id
    ]

    assert len(matching) == 1

    # Your repository preserves the original title
    # when updating an existing conversation.
    assert matching[0].title == "Original title"


def test_conversation_title_is_limited_to_40_characters(test_db):
    thread_id = "long-title-thread"

    message = "A" * 50

    create_or_update_conversation(
        thread_id,
        message,
    )

    conversations = list_conversations()

    conversation = next(
        item
        for item in conversations
        if item.thread_id == thread_id
    )

    assert conversation.title == ("A" * 40) + "..."
