"""Three chunking strategies.

Why chunk at all: an embedding represents one piece of text as one vector.
A whole document squeezed into one vector blurs every topic together, so search
can't point at the relevant passage. Chunks are the units that get embedded,
stored and returned by search.

* ``fixed``     - sliding window of ~``chunk_size`` characters with ``overlap``.
                  Predictable sizes; cut points move to the nearest whitespace so
                  words are never split.
* ``sentence``  - whole sentences packed up to ~``chunk_size`` characters.
                  Each chunk is a clean unit of meaning; good for precise questions.
* ``paragraph`` - one chunk per paragraph; short paragraphs (e.g. headings) are
                  merged into the following one, long ones are split by sentence.
                  Keeps a topic together; good for "how do I..." questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidArgumentError

STRATEGIES = ("fixed", "sentence", "paragraph")
DEFAULTS = {
    "fixed": {"chunk_size": 1000, "overlap": 200},
    "sentence": {"chunk_size": 800},
    "paragraph": {"chunk_size": 1500, "min_chars": 200},
}

# A sentence ends at . ! ? … (optionally followed by closing quotes/brackets) and whitespace.
# Works for English and Hebrew, which share Latin punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "no.", "approx.", "incl."}


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, strategy: str, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Split ``text`` with the given strategy and return numbered, non-empty chunks."""
    if strategy not in STRATEGIES:
        raise InvalidArgumentError(f"Unknown strategy '{strategy}'. Choose one of: {', '.join(STRATEGIES)}")
    size = chunk_size or DEFAULTS[strategy]["chunk_size"]
    if size < 100:
        raise InvalidArgumentError("--chunk-size must be at least 100 characters")

    if strategy == "fixed":
        ov = DEFAULTS["fixed"]["overlap"] if overlap is None else overlap
        if ov < 0 or ov >= size:
            raise InvalidArgumentError("--overlap must be >= 0 and smaller than --chunk-size")
        pieces = fixed_overlap(text, size, ov)
    elif strategy == "sentence":
        pieces = by_sentence(text, size)
    else:
        pieces = by_paragraph(text, size, DEFAULTS["paragraph"]["min_chars"])

    pieces = [p.strip() for p in pieces if p and p.strip()]
    return [Chunk(index=i, text=p) for i, p in enumerate(pieces)]


# ─────────────────────────── strategies ───────────────────────────

def fixed_overlap(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Windows of at most ``size`` characters; each starts ``overlap`` characters before the previous end."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start + size // 2, end + 1)  # back off to a word boundary
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        if nxt > 0 and text[nxt - 1] != " ":  # don't start in the middle of a word
            space = text.find(" ", nxt, end)
            nxt = space + 1 if space != -1 else nxt
        start = nxt
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common abbreviations ("e.g.") attached."""
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):  # a sentence never spans a paragraph break
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        parts = _SENTENCE_END.split(block)
        buf = ""
        for part in parts:
            if not part:
                continue
            buf = f"{buf} {part}".strip() if buf else part
            last_word = buf.rsplit(" ", 1)[-1].lower()
            leading_number = buf == last_word and re.fullmatch(r"\d{1,2}\.", buf)
            if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word) or leading_number:
                continue  # "e.g.", an initial, or a list/section number ("2. Billing")
            sentences.append(buf)
            buf = ""
        if buf:
            sentences.append(buf)
    return sentences


def by_sentence(text: str, max_chars: int = 800) -> list[str]:
    """Pack whole sentences into chunks of up to ``max_chars`` characters."""
    chunks: list[str] = []
    buf = ""
    for sent in split_sentences(text):
        if len(sent) > max_chars:  # a single very long "sentence" (e.g. a table row dump)
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(fixed_overlap(sent, max_chars, 0))
            continue
        is_heading = not re.search(r"[.!?…:;\"'”)\]]$", sent) and len(sent) < 120
        if is_heading and buf:  # a heading opens the next chunk instead of closing this one
            chunks.append(buf)
            buf = sent
            continue
        candidate = f"{buf} {sent}" if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)
    return chunks


def by_paragraph(text: str, max_chars: int = 1500, min_chars: int = 200) -> list[str]:
    """One chunk per paragraph; merge short ones forward, split long ones by sentence."""
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]

    merged: list[str] = []
    carry = ""
    for para in paragraphs:
        para = f"{carry}\n{para}" if carry else para
        if len(para) < min_chars:
            carry = para  # e.g. a heading: attach it to the paragraph that follows
            continue
        merged.append(para)
        carry = ""
    if carry:  # trailing short paragraph: attach to the previous chunk if it fits
        if merged and len(merged[-1]) + len(carry) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]}\n{carry}"
        else:
            merged.append(carry)

    chunks: list[str] = []
    for para in merged:
        chunks.extend([para] if len(para) <= max_chars else by_sentence(para, max_chars))
    return chunks
