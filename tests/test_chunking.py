"""Chunking tests - pure functions, no API key or database needed."""

import pytest

from doc_indexer.chunking import by_paragraph, by_sentence, chunk_text, fixed_overlap, split_sentences
from doc_indexer.errors import InvalidArgumentError

WORDS = " ".join(f"word{i}" for i in range(600))  # ~4,500 characters of distinct words


class TestFixedOverlap:
    def test_respects_max_size(self):
        assert all(len(c) <= 500 for c in fixed_overlap(WORDS, 500, 100))

    def test_never_splits_a_word(self):
        words = set(WORDS.split())
        for chunk in fixed_overlap(WORDS, 500, 100):
            assert set(chunk.split()) <= words

    def test_consecutive_chunks_overlap(self):
        chunks = fixed_overlap(WORDS, 500, 100)
        for a, b in zip(chunks, chunks[1:]):
            assert b.split()[0] in a.split()

    def test_covers_all_text(self):
        chunks = fixed_overlap(WORDS, 500, 100)
        assert set(" ".join(chunks).split()) == set(WORDS.split())

    def test_zero_overlap_has_no_repeats(self):
        chunks = fixed_overlap(WORDS, 500, 0)
        assert sum(len(c.split()) for c in chunks) == len(WORDS.split())

    def test_short_text_is_one_chunk(self):
        assert fixed_overlap("short text", 500, 100) == ["short text"]


class TestSentences:
    def test_english_and_hebrew(self):
        text = "First sentence. Second one! Is this third? יש תקלה באזור. הצפי הוא 16:00."
        assert split_sentences(text) == [
            "First sentence.", "Second one!", "Is this third?", "יש תקלה באזור.", "הצפי הוא 16:00."]

    def test_keeps_abbreviations(self):
        assert split_sentences("Use a cable, e.g. for a TV. Then restart.") == [
            "Use a cable, e.g. for a TV.", "Then restart."]

    def test_keeps_section_numbers_with_their_heading(self):
        assert split_sentences("2. Billing refunds take two cycles.") == ["2. Billing refunds take two cycles."]

    def test_packs_whole_sentences(self):
        text = " ".join(f"This is sentence number {i}." for i in range(100))
        chunks = by_sentence(text, 300)
        assert all(len(c) <= 300 for c in chunks)
        assert all(c.endswith(".") for c in chunks)

    def test_oversized_sentence_is_split(self):
        assert all(len(c) <= 200 for c in by_sentence("x " * 400, 200))


class TestParagraphs:
    def test_heading_merges_with_next_paragraph(self):
        text = "1. Heading\n\n" + "Body text. " * 30
        chunks = by_paragraph(text, 1500, 200)
        assert len(chunks) == 1 and chunks[0].startswith("1. Heading")

    def test_long_paragraph_is_split(self):
        text = "A sentence that repeats. " * 200
        assert all(len(c) <= 1000 for c in by_paragraph(text, 1000, 200))

    def test_separate_paragraphs_stay_separate(self):
        a, b = "Alpha " * 60, "Beta " * 60
        assert len(by_paragraph(f"{a}\n\n{b}", 1500, 200)) == 2


class TestChunkText:
    @pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph"])
    def test_indexes_are_sequential_and_chunks_non_empty(self, strategy):
        chunks = chunk_text("Hello world. " * 300, strategy)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert all(c.text.strip() for c in chunks)

    def test_unknown_strategy(self):
        with pytest.raises(InvalidArgumentError):
            chunk_text("text", "words")

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(InvalidArgumentError):
            chunk_text("text", "fixed", chunk_size=500, overlap=500)

    def test_minimum_chunk_size(self):
        with pytest.raises(InvalidArgumentError):
            chunk_text("text", "sentence", chunk_size=50)
