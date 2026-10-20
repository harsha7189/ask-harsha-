"""A scripted stand-in for the Anthropic client, so the loop can be tested offline."""
from __future__ import annotations

import copy
from types import SimpleNamespace


def text(t):
    return {"type": "text", "text": t}


def tool_use(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def flatten(kwargs):
    """Tests read the system prompt as one string; keep the original blocks too."""
    sysp = kwargs.get("system")
    if isinstance(sysp, list):
        kwargs["system_blocks"] = sysp
        kwargs["system"] = "".join(b.get("text", "") for b in sysp)
    return kwargs


def response(content, stop_reason, inp=100, out=20):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=inp, output_tokens=out))


class FakeClient:
    """Returns the scripted responses in order and records every request it receives."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self  # so code can call client.messages.create(...)

    def create(self, **kwargs):
        self.requests.append(flatten(copy.deepcopy(kwargs)))
        if not self.script:
            raise AssertionError("FakeClient ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeStream:
    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.resp.content:
            t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
            if t == "server_tool_use":
                yield SimpleNamespace(type="content_block_start", content_block=b)
            elif t == "text":
                words = b["text"].split(" ")
                for i, w in enumerate(words):
                    yield SimpleNamespace(type="text", text=w + (" " if i < len(words) - 1 else ""))

    def get_final_message(self):
        return self.resp


def _stream(self, **kwargs):
    return _FakeStream(self.create(**kwargs))


FakeClient.stream = _stream


class RoutingClient(FakeClient):
    """Like FakeClient, but Proof Mode's checklist call (which runs in parallel) gets its own answer."""

    def __init__(self, script, plan=None):
        super().__init__(script)
        self.plan = plan or {"expects_file": False, "file_types": [], "checks": [], "cannot_check": []}
        self.plan_requests = []

    def create(self, **kwargs):
        if (kwargs.get("tool_choice") or {}).get("name") == "set_checklist":
            self.plan_requests.append(kwargs)
            return response([tool_use("plan", "set_checklist", self.plan)], "tool_use")
        return super().create(**kwargs)
