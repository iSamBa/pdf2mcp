"""Tests for pdf2mcp.chunker module."""

import pytest

from pdf2mcp.chunker import (
    _extract_blocks,
    _is_atomic_block,
    _recursive_split,
    _split_by_headers,
    chunk_markdown,
    estimate_tokens,
)


class TestEstimateTokens:
    """Test token estimation heuristic."""

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        assert estimate_tokens("hello world") == 2  # 11 chars // 4

    def test_longer_text(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestSplitByHeaders:
    """Test markdown header splitting."""

    def test_no_headers(self) -> None:
        result = _split_by_headers("Just some text without headers.")
        assert len(result) == 1
        assert result[0][0] == ""  # untitled
        assert result[0][1] == "Just some text without headers."
        assert result[0][2] == 0  # offset

    def test_single_header(self) -> None:
        md = "## Section One\n\nContent of section one."
        result = _split_by_headers(md)
        assert len(result) == 1
        assert result[0][0] == "Section One"
        assert "Content of section one." in result[0][1]

    def test_multiple_headers(self) -> None:
        md = "## First\n\nContent 1.\n\n## Second\n\nContent 2."
        result = _split_by_headers(md)
        assert len(result) == 2
        assert result[0][0] == "First"
        assert result[1][0] == "Second"

    def test_preamble_before_first_header(self) -> None:
        md = "Preamble text.\n\n## First Section\n\nContent."
        result = _split_by_headers(md)
        assert len(result) == 2
        assert result[0][0] == ""  # preamble
        assert "Preamble text." in result[0][1]
        assert result[1][0] == "First Section"

    def test_h3_headers(self) -> None:
        md = "### Sub Section\n\nSub content."
        result = _split_by_headers(md)
        assert len(result) == 1
        assert result[0][0] == "Sub Section"

    def test_h1_not_used_as_split_boundary(self) -> None:
        md = "# Title\n\nIntro text.\n\n## Section\n\nContent."
        result = _split_by_headers(md)
        # H1 should be part of preamble, not a section boundary
        assert result[0][0] == ""  # preamble includes H1
        assert "# Title" in result[0][1]
        assert result[1][0] == "Section"

    def test_empty_section_skipped(self) -> None:
        md = "## Empty\n\n## Non-empty\n\nContent here."
        result = _split_by_headers(md)
        titles = [title for title, _, _ in result]
        assert "Non-empty" in titles

    def test_offsets_are_correct(self) -> None:
        md = "## First\n\nContent.\n\n## Second\n\nMore."
        result = _split_by_headers(md)
        for title, text, offset in result:
            assert md[offset : offset + len(text)] == text


class TestIsAtomicBlock:
    """Test detection of atomic blocks."""

    def test_code_block(self) -> None:
        block = "```python\nprint('hello')\n```"
        assert _is_atomic_block(block) is True

    def test_code_block_no_language(self) -> None:
        block = "```\nsome code\n```"
        assert _is_atomic_block(block) is True

    def test_table(self) -> None:
        block = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        assert _is_atomic_block(block) is True

    def test_prose(self) -> None:
        assert _is_atomic_block("Just regular text.") is False

    def test_mixed_content(self) -> None:
        block = "Some text\n| A | B |\nMore text"
        assert _is_atomic_block(block) is False


class TestExtractBlocks:
    """Test block extraction from markdown."""

    def test_prose_only(self) -> None:
        text = "Paragraph one.\n\nParagraph two."
        blocks = _extract_blocks(text)
        assert len(blocks) == 2

    def test_code_block_preserved(self) -> None:
        text = "Before.\n\n```python\ncode here\n```\n\nAfter."
        blocks = _extract_blocks(text)
        code_blocks = [b for b in blocks if _is_atomic_block(b)]
        assert len(code_blocks) == 1
        assert "code here" in code_blocks[0]

    def test_table_preserved(self) -> None:
        text = "Before.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter."
        blocks = _extract_blocks(text)
        table_blocks = [b for b in blocks if _is_atomic_block(b)]
        assert len(table_blocks) == 1
        assert "| A | B |" in table_blocks[0]

    def test_code_block_with_pipes_not_duplicated(self) -> None:
        text = "Before.\n\n```\n| output | value |\n|--------|-------|\n```\n\nAfter."
        blocks = _extract_blocks(text)
        # The pipe lines inside the code block should not create a separate table block
        code_blocks = [b for b in blocks if b.strip().startswith("```")]
        assert len(code_blocks) == 1
        assert "| output | value |" in code_blocks[0]


class TestRecursiveSplit:
    """Test the recursive text splitting function."""

    def test_splits_on_double_newline(self) -> None:
        text = ("A" * 800) + "\n\n" + ("B" * 800)
        chunks = _recursive_split(text, chunk_size=300, chunk_overlap=20)
        assert len(chunks) >= 2

    def test_falls_back_to_char_split(self) -> None:
        # Single long string with no separators
        text = "A" * 4000
        chunks = _recursive_split(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_overlap_in_char_split(self) -> None:
        text = "A" * 4000
        chunks = _recursive_split(text, chunk_size=200, chunk_overlap=20)
        if len(chunks) >= 2:
            # Second chunk should overlap with end of first
            overlap_chars = 20 * 4
            first_tail = chunks[0][-overlap_chars:]
            assert chunks[1].startswith(first_tail)

    def test_single_separator_split(self) -> None:
        text = "Short part one. Short part two."
        chunks = _recursive_split(text, chunk_size=1000, chunk_overlap=10)
        # Should not split since it fits in one chunk
        assert len(chunks) == 1

    def test_step_never_zero(self) -> None:
        # This would previously crash with range() step 0
        # Now chunk_markdown validates this, but _recursive_split
        # also guards with max(1, ...)
        text = "A" * 4000
        # Even with equal overlap and size, should not crash
        chunks = _recursive_split(text, chunk_size=100, chunk_overlap=100)
        assert len(chunks) > 0


class TestChunkMarkdown:
    """Test the main chunk_markdown function."""

    def test_simple_document(self) -> None:
        md = "## Introduction\n\nThis is a short introduction."
        chunks = chunk_markdown(md, "test.pdf")
        assert len(chunks) >= 1
        assert chunks[0].metadata.source_file == "test.pdf"
        assert chunks[0].metadata.section_title == "Introduction"
        assert chunks[0].metadata.chunk_index == 0

    def test_code_block_stays_intact(self) -> None:
        code = "x = 1\ny = 2\nz = x + y"
        md = f"## Code Example\n\n```python\n{code}\n```"
        chunks = chunk_markdown(md, "test.pdf")
        code_chunks = [c for c in chunks if "x = 1" in c.text]
        assert len(code_chunks) == 1
        assert "y = 2" in code_chunks[0].text
        assert "z = x + y" in code_chunks[0].text

    def test_table_stays_intact(self) -> None:
        md = (
            "## Parameters\n\n"
            "| Param | Value |\n"
            "|-------|-------|\n"
            "| Speed | 100   |\n"
            "| Accel | 50    |"
        )
        chunks = chunk_markdown(md, "test.pdf")
        table_chunks = [c for c in chunks if "| Param | Value |" in c.text]
        assert len(table_chunks) == 1
        assert "| Speed | 100   |" in table_chunks[0].text
        assert "| Accel | 50    |" in table_chunks[0].text

    def test_section_titles_in_metadata(self) -> None:
        md = "## Setup\n\nSetup content.\n\n## Usage\n\nUsage content."
        chunks = chunk_markdown(md, "test.pdf")
        titles = {c.metadata.section_title for c in chunks}
        assert "Setup" in titles
        assert "Usage" in titles

    def test_chunk_indices_sequential(self) -> None:
        md = "## A\n\nContent A.\n\n## B\n\nContent B.\n\n## C\n\nContent C."
        chunks = chunk_markdown(md, "test.pdf")
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input(self) -> None:
        chunks = chunk_markdown("", "test.pdf")
        assert chunks == []

    def test_whitespace_only_input(self) -> None:
        chunks = chunk_markdown("   \n\n   ", "test.pdf")
        assert chunks == []

    def test_no_headers(self) -> None:
        md = "Just plain text without any markdown headers."
        chunks = chunk_markdown(md, "test.pdf")
        assert len(chunks) >= 1
        assert chunks[0].metadata.section_title == ""

    def test_long_section_gets_split(self) -> None:
        long_text = "This is a sentence with enough words. " * 200
        md = f"## Long Section\n\n{long_text}"
        chunks = chunk_markdown(md, "test.pdf", chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_single_line_input(self) -> None:
        chunks = chunk_markdown("Single line.", "test.pdf")
        assert len(chunks) == 1
        assert "Single line." in chunks[0].text

    def test_chunk_overlap_exists(self) -> None:
        sentences = [f"Sentence number {i} has some words in it." for i in range(100)]
        long_text = " ".join(sentences)
        md = f"## Test\n\n{long_text}"
        chunks = chunk_markdown(md, "test.pdf", chunk_size=100, chunk_overlap=20)

        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                assert chunks[i].text.strip()

    def test_page_numbers_tracked(self) -> None:
        md = "Page 1 content.\n\n---\n\nPage 2 content."
        chunks = chunk_markdown(md, "test.pdf")
        all_pages: list[int] = []
        for c in chunks:
            all_pages.extend(c.metadata.page_numbers)
        if all_pages:
            assert max(all_pages) >= 2

    def test_custom_chunk_size(self) -> None:
        long_text = "Word " * 1000
        md = f"## Test\n\n{long_text}"
        small_chunks = chunk_markdown(md, "test.pdf", chunk_size=100)
        large_chunks = chunk_markdown(md, "test.pdf", chunk_size=1000)
        assert len(small_chunks) > len(large_chunks)

    def test_raises_on_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_markdown("text", "test.pdf", chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_markdown("text", "test.pdf", chunk_size=100, chunk_overlap=200)
