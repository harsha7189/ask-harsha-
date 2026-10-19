"""Turn files a visitor attaches (photos, PDFs, Word documents, text files) into content blocks the model can read.

The page does the heavy lifting on the phone: photos are resized to at most 1568 px on the long edge and sent as
JPEG, PDFs and Word files are sent as base64, and text files as text. This module checks everything again, because
the server can't trust the page:
  image  jpeg/png/gif/webp, real image bytes, at most 5 MB (the API's per-image limit)
  pdf    real PDF bytes, at most 10 MB and a page limit (stricter for visitors)
  docx   real Word file; the text is pulled out here (paragraphs and tables), since the model can't open .docx
  text   plain text, code, CSV, JSON, Markdown and so on, at most 200,000 characters

Nothing is stored or logged except counts. The last block gets a cache marker, so follow-up questions about the same
file (and the loop's own tool steps) reuse it at a fraction of the cost instead of paying for it again.
"""
from __future__ import annotations

import base64
import binascii
import io
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

MAX_FILES = 4
MAX_TOTAL_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAGIC = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG",), "image/gif": (b"GIF8",), "image/webp": (b"RIFF",)}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class AttachmentError(ValueError):
    """A file that can't be used; the message is safe to show the visitor."""


@dataclass
class Prepared:
    blocks: list[dict]
    summary: list[dict]   # [{"kind", "pages"?}], for logging and the page; never the content


def _b64(data: str, name: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise AttachmentError(f"“{name}” didn't upload correctly. Please try attaching it again.")


def count_pdf_pages(raw: bytes) -> int:
    """Estimate pages from page objects. Good enough for a limit; the API enforces its own too."""
    return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", raw))


def docx_text(raw: bytes, name: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            xml = z.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError):
        raise AttachmentError(f"“{name}” doesn't look like a Word document (.docx).")
    root = ElementTree.fromstring(xml)
    lines = []
    for el in root.iter():
        if el.tag == f"{W}p":
            text = "".join(t.text or "" for t in el.iter(f"{W}t"))
            if el.find(f".//{W}tab") is not None and not text:
                text = "\t"
            lines.append(text)
        elif el.tag == f"{W}tr":
            lines.append("")  # a little separation between table rows
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def prepare(items: list, max_pdf_pages: int = 30) -> Prepared:
    """items: objects or dicts with name, kind (image|pdf|docx|text), media_type, data."""
    def g(item, key, default=""):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    if len(items) > MAX_FILES:
        raise AttachmentError(f"You can attach up to {MAX_FILES} files at a time.")
    blocks, summary, total = [], [], 0
    for item in items:
        name = (g(item, "name") or "file").strip()[:120] or "file"
        kind, media, data = g(item, "kind"), g(item, "media_type"), g(item, "data") or ""
        if kind == "image":
            if media not in IMAGE_TYPES:
                raise AttachmentError(f"“{name}” is a kind of image I can't read. JPEG, PNG, GIF and WebP work.")
            raw = _b64(data, name)
            if not raw.startswith(MAGIC[media]):
                raise AttachmentError(f"“{name}” isn't a readable image.")
            if len(raw) > MAX_IMAGE_BYTES:
                raise AttachmentError(f"“{name}” is too large. Photos need to be under 5 MB.")
            total += len(raw)
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
            summary.append({"kind": "image"})
        elif kind == "pdf":
            raw = _b64(data, name)
            if not raw.startswith(b"%PDF"):
                raise AttachmentError(f"“{name}” isn't a readable PDF.")
            if len(raw) > MAX_PDF_BYTES:
                raise AttachmentError(f"“{name}” is too large. PDFs need to be under 10 MB.")
            pages = count_pdf_pages(raw)
            if pages > max_pdf_pages:
                raise AttachmentError(f"“{name}” has about {pages} pages. I can read up to {max_pdf_pages} pages at a time.")
            total += len(raw)
            blocks.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                           "title": name})
            summary.append({"kind": "pdf", "pages": pages})
        elif kind == "docx":
            raw = _b64(data, name)
            if len(raw) > MAX_PDF_BYTES:
                raise AttachmentError(f"“{name}” is too large. Documents need to be under 10 MB.")
            text = docx_text(raw, name)
            if not text:
                raise AttachmentError(f"“{name}” has no text I can read.")
            total += len(raw)
            blocks.append({"type": "text", "text": f'<file name="{name}">\n{text[:MAX_TEXT_CHARS]}\n</file>'})
            summary.append({"kind": "docx"})
        elif kind == "text":
            if len(data) > MAX_TEXT_CHARS:
                raise AttachmentError(f"“{name}” is too long. Text files can be up to {MAX_TEXT_CHARS:,} characters.")
            if not data.strip():
                raise AttachmentError(f"“{name}” is empty.")
            total += len(data.encode())
            blocks.append({"type": "text", "text": f'<file name="{name}">\n{data}\n</file>'})
            summary.append({"kind": "text"})
        else:
            raise AttachmentError(f"I can't read “{name}” yet. Photos, PDFs, Word documents and text files work.")
        if total > MAX_TOTAL_BYTES:
            raise AttachmentError("Those files are too large together. Try fewer or smaller files.")
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return Prepared(blocks, summary)
