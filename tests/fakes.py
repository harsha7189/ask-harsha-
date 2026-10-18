"""A scripted stand-in for the Anthropic client, so the loop can be tested offline."""
from __future__ import annotations

import copy
from types import SimpleNamespace


def text(t):
    return {"type": "text", "text": t}


def tool_use(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def response(content, stop_reason, inp=100, out=20):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=inp, output_tokens=out))


class FakeClient:
    """Returns the scripted responses in order and records every request it receives."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self  # so code can call client.messages.create(...)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
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
