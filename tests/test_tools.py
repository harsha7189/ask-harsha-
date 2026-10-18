import json

from agent.kb import KnowledgeBase
from agent.tools import ToolRegistry


def reg():
    return ToolRegistry(KnowledgeBase.load())


def test_specs_are_valid_tool_definitions():
    for spec in reg().specs():
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object"


def test_search_records_retrieved_ids():
    r = reg()
    out, err = r.call("search_facts", {"query": "Cybage sprints"})
    assert not err
    ids = [x["id"] for x in json.loads(out)["results"]]
    assert "F41" in ids and "F41" in r.retrieved


def test_get_topic_returns_all_facts_in_topic():
    r = reg()
    out, err = r.call("get_topic", {"topic": "certifications"})
    assert not err and len(json.loads(out)["results"]) == 5


def test_errors_are_returned_not_raised():
    r = reg()
    assert r.call("search_facts", {})[1] is True
    assert r.call("search_facts", {"query": "x", "k": 99})[1] is True
    assert r.call("get_topic", {"topic": "astronomy"})[1] is True
    msg, err = r.call("delete_everything", {})
    assert err and "Unknown tool" in msg


def test_no_results_explains_next_step():
    out, err = reg().call("search_facts", {"query": "quantum chromodynamics"})
    assert not err and json.loads(out)["results"] == [] and "note" in json.loads(out)
