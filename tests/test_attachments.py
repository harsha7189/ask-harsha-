import base64
import io
import zipfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import attachments as files  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text  # noqa: E402
from server.app import FILES_ACK, Settings, create_app  # noqa: E402

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()
JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 64).decode()


def pdf(pages=2):
    body = b"%PDF-1.4\n" + b"".join(b"%d 0 obj<</Type /Page /Parent 1 0 R>>endobj\n" % (i + 2) for i in range(pages))
    body += b"1 0 obj<</Type /Pages /Count %d>>endobj\n%%%%EOF" % pages
    return base64.b64encode(body).decode()


def docx(paragraphs):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = f'<w:document {ns}><w:body>' + "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs) + "</w:body></w:document>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return base64.b64encode(buf.getvalue()).decode()


def test_image_pdf_docx_and_text_become_blocks():
    out = files.prepare([
        {"name": "sign.jpg", "kind": "image", "media_type": "image/jpeg", "data": JPEG},
        {"name": "lease.pdf", "kind": "pdf", "media_type": "application/pdf", "data": pdf(3)},
        {"name": "notes.docx", "kind": "docx", "data": docx(["Meeting notes", "Ship on Friday"])},
        {"name": "list.csv", "kind": "text", "data": "item,qty\nmilk,2"},
    ])
    kinds = [b["type"] for b in out.blocks]
    assert kinds == ["image", "document", "text", "text"]
    assert out.blocks[1]["title"] == "lease.pdf" and out.summary[1] == {"kind": "pdf", "pages": 3}
    assert "Meeting notes\nShip on Friday" in out.blocks[2]["text"] and '<file name="notes.docx">' in out.blocks[2]["text"]
    assert out.blocks[-1]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in out.blocks[0]


@pytest.mark.parametrize("item, message", [
    ({"name": "a.png", "kind": "image", "media_type": "image/png", "data": JPEG}, "isn't a readable image"),
    ({"name": "a.bmp", "kind": "image", "media_type": "image/bmp", "data": JPEG}, "can't read"),
    ({"name": "a.pdf", "kind": "pdf", "data": PNG}, "isn't a readable PDF"),
    ({"name": "a.pdf", "kind": "pdf", "data": "not base64!!"}, "didn't upload correctly"),
    ({"name": "a.docx", "kind": "docx", "data": PNG}, "doesn't look like a Word document"),
    ({"name": "a.txt", "kind": "text", "data": "   "}, "is empty"),
])
def test_bad_files_get_a_clear_message(item, message):
    with pytest.raises(files.AttachmentError, match=message):
        files.prepare([item])


def test_limits_on_count_and_pages():
    with pytest.raises(files.AttachmentError, match="up to 4"):
        files.prepare([{"name": "t", "kind": "text", "data": "x"}] * 5)
    with pytest.raises(files.AttachmentError, match="about 40 pages"):
        files.prepare([{"name": "big.pdf", "kind": "pdf", "data": pdf(40)}], max_pdf_pages=30)
    assert files.prepare([{"name": "big.pdf", "kind": "pdf", "data": pdf(40)}], max_pdf_pages=100).summary[0]["pages"] == 40


def make(script):
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    fake = FakeClient(script)
    return TestClient(create_app(client=fake, settings=s, notes=MemoryNotesStore())), fake


def test_chat_sends_files_first_and_labels_the_answer():
    c, fake = make([response([text("The sign says the store opens at 9.")], "end_turn")])
    hist = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    r = c.post("/api/chat", json={"question": "What time does it open?", "history": hist,
                                  "attachments": [{"name": "sign.jpg", "kind": "image", "media_type": "image/jpeg", "data": JPEG}]})
    assert r.status_code == 200 and r.json()["label"] == "file"
    msgs = fake.requests[0]["messages"]
    assert msgs[0]["role"] == "user" and msgs[0]["content"][1]["type"] == "image"
    assert msgs[1] == {"role": "assistant", "content": FILES_ACK}
    assert [m["content"] for m in msgs[2:]] == ["Hi", "Hello!", "What time does it open?"]


def test_bad_attachment_is_a_400_before_any_model_call():
    c, fake = make([])
    r = c.post("/api/chat", json={"question": "x", "attachments": [{"name": "a.pdf", "kind": "pdf", "data": PNG}]})
    assert r.status_code == 400 and "readable PDF" in r.json()["message"] and fake.requests == []
    assert c.post("/api/chat", json={"question": "x", "attachments": [{"name": "a", "kind": "zip", "data": "x"}]}).status_code == 422


def test_visitors_get_a_lower_page_limit_than_the_owner():
    c, _ = make([response([text("ok")], "end_turn")])
    body = {"question": "Summarize", "attachments": [{"name": "r.pdf", "kind": "pdf", "data": pdf(40)}]}
    assert c.post("/api/chat", json=body).status_code == 400
    assert c.post("/api/chat", json=body, headers={"x-owner-passcode": "open-sesame"}).status_code == 200


def test_stream_says_it_is_reading_the_file_and_cache_tokens_are_charged():
    from types import SimpleNamespace
    resp = SimpleNamespace(content=[text("It's a lease.")], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_creation_input_tokens=4000, cache_read_input_tokens=0))
    c, _ = make([resp])
    r = c.post("/api/chat/stream", json={"question": "What is it?", "attachments": [{"name": "l.pdf", "kind": "pdf", "data": pdf(1)}]})
    assert "Reading your file" in r.text
    final = [line for line in r.text.split("\n\n") if line.startswith("event: final")][0]
    import json
    data = json.loads(final.split("data: ", 1)[1])
    # 100 + 4000 * 1.25 input tokens at $2/MTok plus 20 output at $10/MTok
    assert abs(data["cost_usd"] - ((100 + 5000) * 2 + 20 * 10) / 1e6) < 1e-6
