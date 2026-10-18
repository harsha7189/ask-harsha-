from agent import grounding
from agent.kb import KnowledgeBase

KB = KnowledgeBase.load()


def test_clean_answer_passes():
    r = grounding.check("He delivered 26 consecutive on-time sprints [F41].", {"F41"}, KB)
    assert r.ok and r.cited == ["F41"]


def test_unknown_citation_flagged():
    r = grounding.check("He won an award [F99].", {"F41"}, KB)
    assert r.unknown_ids == ["F99"] and not r.ok


def test_citation_without_lookup_flagged():
    r = grounding.check("He is CSPO certified [F50].", {"F41"}, KB)
    assert r.unretrieved_ids == ["F50"]


def test_invented_number_flagged():
    r = grounding.check("He drove 48% adoption [F14].", {"F14"}, KB)
    assert "48%" in r.unsupported_numbers


def test_percent_cannot_borrow_support_from_plain_number():
    r = grounding.check("Adoption reached 60% [F14].", {"F14"}, KB)  # F14 mentions "60 days", not 60%
    assert "60%" in r.unsupported_numbers


def test_abbreviated_amounts_match_full_numbers():
    r = grounding.check("He governs 12K+ video hours a month [F23].", {"F23"}, KB)
    assert r.ok, r.summary()


def test_supported_numbers_pass_in_other_formats():
    r = grounding.check("Adoption hit 35% across 200+ users, protecting $8M+ [F14][F12].", {"F14", "F12"}, KB)
    assert r.ok, r.summary()
