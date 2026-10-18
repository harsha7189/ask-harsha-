import base64
import io
import json
import urllib.error

import pytest

from agent import grounding
from agent.kb import KnowledgeBase
from agent.notes import FileNotesStore, GitHubNotesStore, MemoryNotesStore, NotesIndex, NotesUnavailable
from agent.tools import ToolRegistry, add_notes_tools


def test_memory_store_add_search_delete():
    s = MemoryNotesStore(["Harsha's favorite cuisine is Hyderabadi biryani.", "Harsha enjoys long-distance running."])
    idx = NotesIndex(s)
    assert [n.id for n in s.all()] == ["N001", "N002"]
    assert idx.search("favorite food cuisine")[0][0].id == "N001"
    s.delete("N001")
    assert idx.search("cuisine") == []           # index rebuilt after the change
    assert s.add("Harsha likes chess.").id == "N003"  # IDs are never reused


def test_note_validation():
    s = MemoryNotesStore()
    with pytest.raises(ValueError):
        s.add("   ")
    with pytest.raises(ValueError):
        s.add("x" * 501)


def test_file_store_persists(tmp_path):
    p = tmp_path / "notes.json"
    FileNotesStore(p).add("Harsha prefers concise answers.")
    assert [n.text for n in FileNotesStore(p).all()] == ["Harsha prefers concise answers."]


def test_visitor_gets_search_only_owner_gets_remember_and_forget():
    s = MemoryNotesStore(["Harsha's favorite color is red."])
    visitor = ToolRegistry(KnowledgeBase.load())
    add_notes_tools(visitor, NotesIndex(s), s, owner=False)
    names = {t["name"] for t in visitor.specs()}
    assert "search_notes" in names and "remember_note" not in names and "forget_note" not in names
    msg, err = visitor.call("remember_note", {"text": "Harsha is 100 years old"})
    assert err and "Unknown tool" in msg and len(s.all()) == 1

    owner = ToolRegistry(KnowledgeBase.load())
    add_notes_tools(owner, NotesIndex(s), s, owner=True)
    out, err = owner.call("remember_note", {"text": "Harsha's favorite season is fall."})
    assert not err and json.loads(out)["saved"]["id"] == "N002"
    assert owner.changes == [{"action": "added", "id": "N002", "text": "Harsha's favorite season is fall."}]
    out, err = owner.call("forget_note", {"id": "n001"})
    assert not err and [n.id for n in s.all()] == ["N002"]
    assert owner.call("forget_note", {"id": "N999"})[1] is True


def test_grounding_accepts_note_citations_and_skips_general_numbers():
    s = MemoryNotesStore(["Harsha runs 25 miles a week."])

    class L:
        by_id = {**KnowledgeBase.load().by_id, **{n.id: n for n in s.all()}}

    r = grounding.check("Water boils at 100 degrees Celsius. He runs 25 miles a week [N001].", {"N001"}, L, strict=False)
    assert r.ok and r.kind == "grounded"
    r = grounding.check("He runs 40 miles a week [N001].", {"N001"}, L, strict=False)
    assert "40" in r.unsupported_numbers
    assert grounding.check("Light travels 300,000 km per second.", set(), L, strict=False).kind == "general"


class FakeGitHub:
    """Minimal stand-in for the GitHub contents API."""

    def __init__(self, notes=None, fail_get=False):
        self.content = json.dumps({"notes": notes or []})
        self.sha, self.fail_get, self.puts = "sha1", fail_get, []

    def __call__(self, req, timeout=10):
        if req.get_method() == "GET":
            if self.fail_get:
                raise urllib.error.URLError("offline")
            body = {"sha": self.sha, "content": base64.b64encode(self.content.encode()).decode()}
        else:
            data = json.loads(req.data)
            assert data["branch"] == "notes"
            if data.get("sha") != self.sha:
                raise urllib.error.HTTPError(req.full_url, 409, "conflict", {}, io.BytesIO(b""))
            self.puts.append(data)
            self.content = base64.b64decode(data["content"]).decode()
            self.sha = f"sha{len(self.puts) + 1}"
            body = {"content": {"sha": self.sha}}
        return io.BytesIO(json.dumps(body).encode())


def test_github_store_reads_and_commits_to_notes_branch():
    gh = FakeGitHub([{"id": "N001", "text": "Harsha likes tea.", "created": "2026-10-18T00:00:00+00:00"}])
    s = GitHubNotesStore("harsha7189/ask-harsha-", "tok", opener=gh)
    assert [n.text for n in s.all()] == ["Harsha likes tea."]
    s.add("Harsha likes coffee too.")
    assert gh.puts[-1]["message"].startswith("Remember N002")
    assert [n["id"] for n in json.loads(gh.content)["notes"]] == ["N001", "N002"]


def test_github_store_never_overwrites_when_it_could_not_read():
    gh = FakeGitHub([{"id": "N001", "text": "Keep me.", "created": "x"}], fail_get=True)
    s = GitHubNotesStore("r/r", "tok", opener=gh)
    with pytest.raises(NotesUnavailable):
        s.add("New note")
    assert gh.puts == []                      # nothing written over the unreadable file
    gh.fail_get = False
    s.add("New note")                         # once GitHub is reachable, it loads first, then appends
    assert [n["text"] for n in json.loads(gh.content)["notes"]] == ["Keep me.", "New note"]


def test_github_conflict_reloads_instead_of_clobbering():
    gh = FakeGitHub([])
    s = GitHubNotesStore("r/r", "tok", opener=gh)
    gh.content = json.dumps({"notes": [{"id": "N001", "text": "Edited on GitHub.", "created": "x"}]})
    gh.sha = "changed-elsewhere"
    with pytest.raises(NotesUnavailable):
        s.add("From the app")
    assert [n.text for n in s.all()] == ["Edited on GitHub."]
