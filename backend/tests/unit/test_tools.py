from ai_layer.tools import tools as tools_module


def test_calculator_addition():
    result = tools_module.calculator.invoke({"expression": "2 + 3"})
    assert result == "5"


def test_calculator_multiplication():
    result = tools_module.calculator.invoke({"expression": "10 * 5"})
    assert result == "50"


def test_calculator_square_root():
    result = tools_module.calculator.invoke(
        {"expression": "math.sqrt(25)"}
    )
    assert result == "5.0"


def test_calculator_division_by_zero():
    result = tools_module.calculator.invoke(
        {"expression": "10 / 0"}
    )
    assert "Calculation error" in result


def test_remember_this_saves_memory(monkeypatch):
    captured = {}

    def fake_save_memory(thread_id, memory):
        captured["thread_id"] = thread_id
        captured["memory"] = memory
        return "Memory saved successfully."

    monkeypatch.setattr(
        tools_module,
        "save_memory",
        fake_save_memory
    )

    result = tools_module.remember_this_impl(
        memory="User prefers concise answers.",
        thread_id="thread-123",
    )

    assert result == "Memory saved successfully."
    assert captured["thread_id"] == "thread-123"
    assert captured["memory"] == "User prefers concise answers."


def test_recall_memory_retrieves_memory(monkeypatch):
    captured = {}

    def fake_search_memory(thread_id, query):
        captured["thread_id"] = thread_id
        captured["query"] = query
        return "User prefers concise answers."

    monkeypatch.setattr(
        tools_module,
        "search_memory",
        fake_search_memory
    )

    result = tools_module.recall_memory_impl(
        query="What are the user's preferences?",
        thread_id="thread-456",
    )

    assert result == "User prefers concise answers."
    assert captured["thread_id"] == "thread-456"
    assert captured["query"] == "What are the user's preferences?"


def test_search_uploaded_documents(monkeypatch):
    captured = {}

    def fake_retrieve_from_rag(query, thread_id):
        captured["query"] = query
        captured["thread_id"] = thread_id
        return "Relevant document content."

    import ai_layer.rag.rag as rag_module

    monkeypatch.setattr(
        rag_module,
        "retrieve_from_rag",
        fake_retrieve_from_rag
    )

    result = tools_module.search_uploaded_documents_impl(
        query="What does the document say?",
        thread_id="thread-789",
    )

    assert result == "Relevant document content."
    assert captured["query"] == "What does the document say?"
    assert captured["thread_id"] == "thread-789"


def test_tools_list_contains_expected_tools():
    tool_names = [tool.name for tool in tools_module.tools]

    assert "calculator" in tool_names
    assert "remember_this" in tool_names
    assert "recall_memory" in tool_names
    assert "search_uploaded_documents" in tool_names
