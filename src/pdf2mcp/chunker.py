"""Markdown-aware text chunking with metadata."""

from __future__ import annotations

import bisect
import logging
import re

from pdf2mcp.models import ChunkMetadata, DocumentChunk

__all__ = ["chunk_markdown", "estimate_tokens"]

logger = logging.getLogger(__name__)

# Page break pattern emitted by pymupdf4llm
_PAGE_BREAK_RE = re.compile(r"^-{3,}\s*$", re.MULTILINE)

# Header pattern — split on ## and ### only (H1 is document title, kept as preamble)
_HEADER_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# Code block pattern (fenced)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

# Table row pattern
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")

# Separators tried in order for recursive splitting
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    return len(text) // 4


def chunk_markdown(
    markdown: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split markdown into chunks respecting document structure.

    Strategy:
    1. Extract page break positions for page number tracking.
    2. Split into sections by ## and ### headers, tracking offsets.
    3. Within each section, split into sized chunks while keeping
       code blocks and tables intact.
    4. Attach metadata (source file, section title, page numbers) to each chunk.
    """
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        raise ValueError(msg)

    page_breaks = _find_page_breaks(markdown)
    sections = _split_by_headers(markdown)

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section_title, section_text, section_offset in sections:
        section_chunks = _split_section(section_text, chunk_size, chunk_overlap)

        for text in section_chunks:
            if not text.strip():
                continue
            page_numbers = _resolve_page_numbers(section_offset, len(text), page_breaks)
            chunk = DocumentChunk(
                text=text,
                metadata=ChunkMetadata(
                    source_file=filename,
                    page_numbers=page_numbers,
                    section_title=section_title,
                    chunk_index=chunk_index,
                ),
            )
            chunks.append(chunk)
            chunk_index += 1
            section_offset += len(text)

    logger.info("%s: %d chunks created", filename, len(chunks))
    return chunks


def _find_page_breaks(markdown: str) -> list[int]:
    """Return character positions of page breaks in the markdown."""
    return [m.start() for m in _PAGE_BREAK_RE.finditer(markdown)]


def _resolve_page_numbers(start: int, length: int, page_breaks: list[int]) -> list[int]:
    """Determine which pages a chunk spans based on page break positions.

    Uses pre-computed start offset instead of searching for text.
    Page numbering starts at 1. A chunk before any page break is on page 1.
    """
    if not page_breaks:
        return []

    end = start + length

    # bisect_right gives the number of page breaks <= position,
    # so the page number is that count + 1.
    start_page = bisect.bisect_right(page_breaks, start) + 1
    end_page = bisect.bisect_right(page_breaks, end) + 1

    return list(range(start_page, end_page + 1))


def _split_by_headers(markdown: str) -> list[tuple[str, str, int]]:
    """Split markdown into (section_title, section_text, start_offset) tuples.

    Splits on ## and ### headers. Text before the first header gets
    section_title = "" (untitled). The start_offset is the character position
    in the original markdown where the section text begins.
    """
    sections: list[tuple[str, str, int]] = []
    matches = list(_HEADER_RE.finditer(markdown))

    if not matches:
        return [("", markdown, 0)]

    # Text before the first header
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()]
        if preamble.strip():
            sections.append(("", preamble, 0))

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        text = markdown[start:end]
        if text.strip():
            sections.append((title, text, start))

    return sections


def _split_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a section into chunks, keeping atomic blocks intact.

    Atomic blocks (code blocks, tables) are never split. If an atomic block
    exceeds chunk_size, it becomes its own chunk.
    """
    blocks = _extract_blocks(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_chars = chunk_overlap * 4  # Convert token estimate to chars

    for block in blocks:
        block_tokens = estimate_tokens(block)

        if _is_atomic_block(block):
            # Flush current buffer before atomic block
            if current:
                chunks.append("".join(current))
                current = []
                current_tokens = 0
            chunks.append(block)
            continue

        # If a single prose block is too large, recursively split it
        if block_tokens > chunk_size:
            if current:
                chunks.append("".join(current))
                current = []
                current_tokens = 0
            sub_chunks = _recursive_split(block, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue

        if current_tokens + block_tokens > chunk_size and current:
            chunk_text = "".join(current)
            chunks.append(chunk_text)
            # Overlap: carry tail of previous chunk into next
            overlap_text = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
            current = [overlap_text] if overlap_text else []
            current_tokens = estimate_tokens(overlap_text) if overlap_text else 0

        current.append(block)
        current_tokens += block_tokens

    if current:
        joined = "".join(current)
        if joined.strip():
            chunks.append(joined)

    return chunks


def _extract_blocks(text: str) -> list[str]:
    """Split text into a sequence of prose blocks and atomic blocks.

    Atomic blocks are fenced code blocks and tables. Everything else
    is prose (split by double newlines).
    """
    # Find all code block spans
    code_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _CODE_BLOCK_RE.finditer(text)
    ]

    # Find all table block spans by tracking line positions
    table_spans: list[tuple[int, int]] = []
    lines = text.split("\n")
    char_offsets: list[int] = []
    offset = 0
    for line in lines:
        char_offsets.append(offset)
        offset += len(line) + 1  # +1 for the newline

    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i].strip()):
            table_start_line = i
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                i += 1
            table_start = char_offsets[table_start_line]
            table_end = char_offsets[i - 1] + len(lines[i - 1])
            # Include trailing newline if present
            if table_end < len(text) and text[table_end] == "\n":
                table_end += 1
            table_spans.append((table_start, table_end))
        else:
            i += 1

    # Merge overlapping spans (code blocks containing pipe chars)
    all_spans = sorted(code_spans + table_spans, key=lambda s: s[0])
    atomic_spans: list[tuple[int, int]] = []
    for span in all_spans:
        if atomic_spans and span[0] < atomic_spans[-1][1]:
            atomic_spans[-1] = (
                atomic_spans[-1][0],
                max(atomic_spans[-1][1], span[1]),
            )
        else:
            atomic_spans.append(span)

    if not atomic_spans:
        paragraphs = text.split("\n\n")
        return [p + "\n\n" for p in paragraphs if p.strip()]

    blocks: list[str] = []
    pos = 0

    for start, end in atomic_spans:
        if start > pos:
            prose = text[pos:start]
            if prose.strip():
                paragraphs = prose.split("\n\n")
                blocks.extend(p + "\n\n" for p in paragraphs if p.strip())
        blocks.append(text[start:end])
        pos = end

    if pos < len(text):
        prose = text[pos:]
        if prose.strip():
            paragraphs = prose.split("\n\n")
            blocks.extend(p + "\n\n" for p in paragraphs if p.strip())

    return blocks


def _is_atomic_block(text: str) -> bool:
    """Return True if text is a code block or table that should not be split."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return True
    # All non-empty lines are table rows
    lines = [line for line in stripped.split("\n") if line.strip()]
    return bool(lines) and all(_TABLE_ROW_RE.match(line.strip()) for line in lines)


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split text using progressively finer separators."""
    overlap_chars = chunk_overlap * 4

    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if estimate_tokens(candidate) > chunk_size and current:
                chunks.append(current)
                # Add overlap from end of previous chunk
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = tail + sep + part if tail else part
            else:
                current = candidate

        if current.strip():
            chunks.append(current)

        if all(estimate_tokens(c) <= chunk_size * 1.5 for c in chunks):
            return chunks

    # Last resort: character-level splitting
    chars_per_chunk = chunk_size * 4
    step = max(1, chars_per_chunk - overlap_chars)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chars_per_chunk]
        if chunk.strip():
            chunks.append(chunk)
    return chunks
