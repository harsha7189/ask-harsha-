import pytest

from agent.kb import Fact, KnowledgeBase, tokenize


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


def test_loads_all_facts_with_unique_ids(kb):
    assert len(kb.facts) >= 30
    assert len(kb.by_id) == len(kb.facts)


def test_tokenize_keeps_money_and_drops_stopwords():
    assert tokenize("What did Harsha do for $8M+ retention?") == ["8m", "retention"]


@pytest.mark.parametrize("query,expected", [
    ("Cybage sprints", "F41"),
    ("AI agent adoption", "F14"),
    ("DRM Widevine FairPlay", "F21"),
    ("MBA degree", "F30"),
    ("how do I contact him", "F70"),
    ("redis cache latency", "F25"),
    ("ROAS marketing", "F43"),
    ("where did he study", "F30"),
])
def test_search_finds_the_right_fact(kb, query, expected):
    ids = [f.id for f, _ in kb.search(query, k=3)]
    assert expected in ids, f"{query!r} returned {ids}"


def test_search_returns_nothing_for_unrelated_query(kb):
    assert kb.search("quantum chromodynamics lattice", k=5) == []


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([Fact("F01", "a", "x"), Fact("F01", "b", "y")])
