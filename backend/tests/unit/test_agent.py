from ai_layer.agent import agent as agent_module


def test_normalize_model_name_returns_default_for_none():
    assert (
        agent_module.normalize_model_name(None)
        == agent_module.DEFAULT_MODEL
    )


def test_normalize_model_name_returns_default_for_invalid_model():
    assert (
        agent_module.normalize_model_name("invalid-model")
        == agent_module.DEFAULT_MODEL
    )


def test_normalize_model_name_accepts_valid_model():
    model = agent_module.ALLOWED_MODELS[0]

    assert agent_module.normalize_model_name(model) == model


def test_normalize_model_name_strips_whitespace():
    model = agent_module.ALLOWED_MODELS[0]

    assert agent_module.normalize_model_name(f" {model} ") == model


def test_get_agent_creates_agent_once(monkeypatch):
    fake_agent = object()
    calls = []

    def fake_build_agent(model_name):
        calls.append(model_name)
        return fake_agent

    monkeypatch.setattr(agent_module, "_AGENT_CACHE", {})
    monkeypatch.setattr(agent_module, "build_agent", fake_build_agent)

    first_agent = agent_module.get_agent("gemini-3.8-flash")
    second_agent = agent_module.get_agent("gemini-3.8-flash")

    assert first_agent is fake_agent
    assert second_agent is fake_agent
    assert len(calls) == 1


def test_get_agent_creates_separate_agents_for_different_models(monkeypatch):
    calls = []

    def fake_build_agent(model_name):
        calls.append(model_name)
        return object()

    monkeypatch.setattr(agent_module, "_AGENT_CACHE", {})
    monkeypatch.setattr(agent_module, "build_agent", fake_build_agent)

    agent_module.get_agent("gemini-3.8-flash")
    agent_module.get_agent("gemini-3.8-pro")

    assert len(calls) == 2
    assert calls == ["gemini-3.8-flash", "gemini-3.8-pro"]
