
from __future__ import annotations

import ast
import json
import re
from typing import Iterable, Iterator


_FENCE_RE = re.compile(r"```(?:json|javascript|js|python)?\s*(.*?)```", flags=re.I | re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def parse_jsonish_payload(raw_text):
    """Return the first parseable JSON/Python-literal object or array.

    Models often wrap JSON in prose, return Python-style single-quoted dicts,
    or leave trailing commas. This function searches fenced blocks and balanced
    object/array spans before trying a small set of parse repairs.
    """

    for candidate in iter_jsonish_candidates(raw_text):
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def iter_jsonish_candidates(raw_text) -> Iterator[str]:
    text = str(raw_text or "").strip()
    if not text:
        return

    seen: set[str] = set()

    def add(candidate: str) -> Iterator[str]:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        yield candidate

    for match in _FENCE_RE.finditer(text):
        yield from add(match.group(1))

    yield from add(text)

    spans = list(_balanced_spans(text))
    spans.sort(key=lambda span: (span[1] - span[0]), reverse=True)
    for start, end in spans:
        yield from add(text[start:end])


def compact_prose_paragraphs(raw_text, *, max_paragraphs: int = 6) -> list[str]:
    """Extract plausible story/prose paragraphs from a non-JSON response."""

    text = _strip_fences(str(raw_text or "")).strip()
    if not text:
        return []
    text = re.sub(r"^\s*(?:Here(?:'s| is)|Sure[:,]?|Answer[:,]?)\s*", "", text, flags=re.I)
    chunks = [chunk.strip(" \t\r\n-*0123456789.）)") for chunk in re.split(r"\n\s*\n|\n(?=\s*\d+[\).])", text)]
    paragraphs = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if len(chunk.split()) >= 8]
    if paragraphs:
        return paragraphs[:max(1, max_paragraphs)]

    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    grouped: list[str] = []
    buffer: list[str] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        buffer.append(sentence.strip())
        if len(" ".join(buffer).split()) >= 35:
            grouped.append(" ".join(buffer))
            buffer = []
    if buffer:
        grouped.append(" ".join(buffer))
    return [chunk for chunk in grouped if len(chunk.split()) >= 8][:max(1, max_paragraphs)]


def _parse_candidate(candidate: str):
    text = _normalize_candidate(candidate)
    if not text:
        return None
    variants = [
        text,
        _TRAILING_COMMA_RE.sub(r"\1", text),
    ]
    for variant in variants:
        try:
            return json.loads(variant)
        except Exception:
            pass
    if text[:1] in "{[":
        for variant in variants:
            try:
                return ast.literal_eval(variant)
            except Exception:
                pass
    return None


def _normalize_candidate(candidate: str) -> str:
    text = str(candidate or "").strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    return text


def _strip_fences(text: str) -> str:
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text
    return "\n\n".join(match.group(1).strip() for match in matches if match.group(1).strip())


def _balanced_spans(text: str) -> Iterable[tuple[int, int]]:
    pairs = {"{": "}", "[": "]"}
    openers = set(pairs)
    closers = set(pairs.values())
    for start, char in enumerate(text):
        if char not in openers:
            continue
        stack: list[str] = []
        in_string = False
        quote_char = ""
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                    continue
                if current == "\\":
                    escape = True
                    continue
                if current == quote_char:
                    in_string = False
                    quote_char = ""
                continue
            if current in {'"', "'"}:
                in_string = True
                quote_char = current
                continue
            if current in openers:
                stack.append(pairs[current])
                continue
            if current in closers:
                if not stack or current != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    yield start, index + 1
                    break
