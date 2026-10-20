import pytest

from agent.loop import Agent
from fakes import FakeClient, response, text, tool_use


def test_tool_call_then_answer():
    client = FakeClient([
        response([text("Looking that up."), tool_use("tu_1", "search_facts", {"query": "Cybage sprints"})], "tool_use"),
        response([text("He delivered 26 consecutive on-time sprints at Cybage [F41].")], "end_turn"),
    ])
    agent = Agent(client)
    reply = agent.ask("How reliable was his delivery at Cybage?")

    assert reply.stop_reason == "end_turn"
    assert reply.steps == 2
    assert reply.report.ok, reply.report.summary()
    assert reply.input_tokens == 200 and reply.output_tokens == 40

    # Second request: history is user -> assistant(tool_use) -> user(tool_result)
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    result = second[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1"
    assert "F41" in result["content"] and "is_error" not in result
    # Tools and system prompt are sent on every call
    assert {t["name"] for t in client.requests[1]["tools"]} == {"search_facts", "get_topic", "list_topics"}
    assert "Ask Harsha" in client.requests[0]["system"]


def test_parallel_tool_calls_answered_in_one_message():
    client = FakeClient([
        response([tool_use("a", "search_facts", {"query": "MBA"}), tool_use("b", "get_topic", {"topic": "certifications"})], "tool_use"),
        response([text("He has an MBA [F30] and is CSPO certified [F50].")], "end_turn"),
    ])
    reply = Agent(client).ask("Education and certs?")
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert reply.report.ok


def test_tool_error_is_sent_back_with_is_error():
    client = FakeClient([
        response([tool_use("x", "get_topic", {"topic": "astronomy"})], "tool_use"),
        response([text("That isn't in Harsha's record.")], "end_turn"),
    ])
    Agent(client).ask("Astronomy?")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "Unknown topic" in result["content"]


def test_step_limit_stops_runaway_loops_and_keeps_history_valid():
    loop_forever = [response([tool_use(f"t{i}", "list_topics", {})], "tool_use") for i in range(3)]
    agent = Agent(FakeClient(loop_forever), max_steps=3)
    reply = agent.ask("?")
    assert reply.stop_reason == "max_steps"
    roles = [m["role"] for m in agent.messages]
    assert roles[-1] == "assistant" and all(a != b for a, b in zip(roles, roles[1:]))


def test_history_carries_across_turns():
    client = FakeClient([
        response([text("Hi! Ask me about Harsha.")], "end_turn"),
        response([text("Sure.")], "end_turn"),
    ])
    agent = Agent(client)
    agent.ask("hello")
    agent.ask("and again")
    assert len(client.requests[1]["messages"]) == 3


def test_api_failure_rolls_back_history():
    client = FakeClient([
        response([tool_use("t1", "list_topics", {})], "tool_use"),
        RuntimeError("network down"),
    ])
    agent = Agent(client)
    with pytest.raises(RuntimeError):
        agent.ask("topics?")
    assert agent.messages == []


def test_invented_number_is_caught():
    client = FakeClient([
        response([tool_use("t", "search_facts", {"query": "AI agent adoption"})], "tool_use"),
        response([text("He drove 60% adoption [F14].")], "end_turn"),
    ])
    reply = Agent(client).ask("AI adoption?")
    assert not reply.report.ok and "60%" in reply.report.unsupported_numbers


def test_chat_calls_skip_up_front_thinking_and_report_timing():
    from agent import speed
    from fakes import FakeClient, response, text
    from agent.loop import Agent
    assert speed.params("claude-sonnet-5-5", "chat") == {"output_config": {"effort": "medium"}, "thinking": {"type": "between_tools"}}
    assert speed.params("claude-haiku-5-5", "helper") == {"output_config": {"effort": "low"}, "thinking": {"type": "disabled"}}
    assert speed.params("claude-sonnet-5-5", "deep") == {"output_config": {"effort": "medium"}}
    assert speed.params("some-other-model", "chat") == {}          # unknown models get nothing extra
    client = FakeClient([response([text("Hi.")], "end_turn")])
    reply = Agent(client, model="claude-sonnet-5-5").ask("Hello")
    req = client.requests[0]
    assert req["thinking"] == {"type": "between_tools"} and req["output_config"] == {"effort": "medium"}
    assert set(reply.timing) >= {"total_s", "model_s", "tools_s", "calls_s", "first_text_s"} and len(reply.timing["calls_s"]) == 1


def test_chat_effort_can_be_changed_by_env(monkeypatch):
    from agent import speed
    monkeypatch.setenv("CHAT_EFFORT", "low")
    assert speed.params("claude-sonnet-5-5", "chat")["output_config"] == {"effort": "low"}
    monkeypatch.setenv("CHAT_EFFORT", "max")
    assert speed.params("claude-sonnet-5-5", "chat")["thinking"] == {"type": "adaptive"}   # between_tools isn't allowed at max
