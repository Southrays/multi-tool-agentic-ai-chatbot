from database.repositories.memory_repository import (
    save_memory,
    search_memory,
)


def test_save_memory(test_db):
    result = save_memory(
        "thread-1",
        "The user prefers concise explanations.",
    )

    assert result == "Memory saved successfully."


def test_search_saved_memory(test_db):
    save_memory(
        "thread-1",
        "The user prefers concise explanations.",
    )

    result = search_memory(
        "thread-1",
        "preferences",
    )

    assert "The user prefers concise explanations." in result


def test_memory_is_separated_by_thread_id(test_db):
    save_memory("thread-1", "Memory belonging to A")
    save_memory("thread-2", "Memory belonging to B")

    result = search_memory("thread-1", "memory")

    assert "Memory belonging to A" in result
    assert "Memory belonging to B" not in result


def test_search_without_memory(test_db):
    result = search_memory("thread-1", "anything")

    assert result == "No saved memory found."
