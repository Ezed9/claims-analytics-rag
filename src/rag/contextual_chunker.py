import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    ANTHROPIC_MODEL_NAME,
    CHUNK_CONTEXT_CACHE_PATH,
    DOCS_DIR,
    GEMINI_MODEL_NAME,
    has_active_llm,
    resolve_provider,
)

MAX_CHUNK_WORDS = 350
FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_NUMBER_RE = re.compile(r"^([\d.]+)\.?\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    scope: str
    heading_path: str
    raw_text: str
    contextualized_text: str


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw_fm, body = match.groups()
    front_matter: dict[str, str] = {}
    for line in raw_fm.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            front_matter[key.strip()] = value.strip()
    return front_matter, body


def _parse_heading(text: str) -> tuple[str | None, str]:
    match = HEADING_NUMBER_RE.match(text)
    if match:
        return match.group(1).rstrip("."), match.group(2).strip()
    return None, text


def _extract_title(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled Document"


def _build_sections(body: str) -> list[dict]:
    current_h2: dict | None = None
    current_h3: dict | None = None
    sections: list[dict] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if current_h3 is not None and current_h2 is not None:
                current_h2["h3_list"].append(current_h3)
                current_h3 = None
            if current_h2 is not None:
                sections.append(current_h2)
            number, title = _parse_heading(line[3:].strip())
            current_h2 = {"number": number, "title": title, "preamble": [], "h3_list": []}
        elif line.startswith("### "):
            if current_h3 is not None and current_h2 is not None:
                current_h2["h3_list"].append(current_h3)
            number, title = _parse_heading(line[4:].strip())
            current_h3 = {"number": number, "title": title, "lines": []}
        elif line.startswith("# "):
            continue
        else:
            if current_h3 is not None:
                current_h3["lines"].append(line)
            elif current_h2 is not None:
                current_h2["preamble"].append(line)

    if current_h3 is not None and current_h2 is not None:
        current_h2["h3_list"].append(current_h3)
    if current_h2 is not None:
        sections.append(current_h2)
    return sections


def _split_long_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    groups: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        word_count = len(paragraph.split())
        if current and current_words + word_count > MAX_CHUNK_WORDS:
            groups.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += word_count
    if current:
        groups.append("\n\n".join(current))
    return groups or [text]


def _subblocks_for_section(section: dict) -> list[tuple[str, str, str]]:
    subblocks: list[tuple[str, str, str]] = []
    preamble_text = "\n".join(section["preamble"]).strip()
    if preamble_text:
        subblocks.append((section["number"] or "0", section["title"], preamble_text))
    for h3 in section["h3_list"]:
        text = "\n".join(h3["lines"]).strip()
        if text:
            heading_path = f"{section['title']} > {h3['title']}"
            subblocks.append((h3["number"] or section["number"] or "0", heading_path, text))
    return subblocks


def _scope_string(front_matter: dict[str, str]) -> str:
    if "carrier" in front_matter:
        plan = front_matter.get("plan_name", "")
        return f"{front_matter['carrier']} / {plan}" if plan else front_matter["carrier"]
    if "device_family" in front_matter:
        return front_matter["device_family"]
    if "device" in front_matter:
        return front_matter["device"]
    return "General"


def _load_cache() -> dict[str, str]:
    if CHUNK_CONTEXT_CACHE_PATH.exists():
        return json.loads(CHUNK_CONTEXT_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CHUNK_CONTEXT_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _generate_situating_sentence(doc_title: str, chunk_text: str) -> str | None:
    prompt = (
        f"Document: {doc_title}\n\nChunk:\n{chunk_text}\n\n"
        "Write one concise sentence (at most 25 words) situating this chunk's content "
        "within the document, for search retrieval purposes. Return only the sentence."
    )
    provider = resolve_provider()
    try:
        if provider == "gemini":
            from google import genai

            client = genai.Client()
            response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
            return response.text.strip()
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=ANTHROPIC_MODEL_NAME,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
    except Exception:
        return None
    return None


def _chunk_for_subblock(
    doc_id: str, doc_title: str, scope: str, number: str, heading_path: str, text: str
) -> list[Chunk]:
    pieces = _split_long_text(text) if len(text.split()) > MAX_CHUNK_WORDS else [text]
    chunks: list[Chunk] = []
    cache = _load_cache() if has_active_llm() else {}
    cache_dirty = False

    for index, piece in enumerate(pieces):
        suffix = "" if len(pieces) == 1 else f"-{chr(ord('a') + index)}"
        chunk_id = f"{doc_id}#{number}{suffix}"
        header = f"[Document: {doc_title} | Scope: {scope} | Section: {heading_path}]"
        contextualized_text = f"{header}\n{piece}"

        if has_active_llm():
            situating = cache.get(chunk_id)
            if situating is None:
                situating = _generate_situating_sentence(doc_title, piece)
                if situating:
                    cache[chunk_id] = situating
                    cache_dirty = True
            if situating:
                contextualized_text = f"{header}\n{situating}\n{piece}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                scope=scope,
                heading_path=heading_path,
                raw_text=piece,
                contextualized_text=contextualized_text,
            )
        )

    if cache_dirty:
        _save_cache(cache)
    return chunks


def parse_document(path: Path) -> list[Chunk]:
    doc_id = path.stem
    raw_text = path.read_text()
    front_matter, body = _parse_front_matter(raw_text)
    doc_title = _extract_title(body)
    scope = _scope_string(front_matter)

    chunks: list[Chunk] = []
    for section in _build_sections(body):
        for number, heading_path, text in _subblocks_for_section(section):
            chunks.extend(_chunk_for_subblock(doc_id, doc_title, scope, number, heading_path, text))
    return chunks


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(parse_document(path))
    return chunks


if __name__ == "__main__":
    for chunk in build_chunks():
        print(chunk.chunk_id, "|", chunk.heading_path, "|", len(chunk.raw_text.split()), "words")
