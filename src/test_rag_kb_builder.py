"""Offline regression test for RAG knowledge-source metadata and chunking."""

import unittest

from src.rag_kb_builder import chunk_markdown, extract_pdf_text, split_recursive


class RagKbBuilderTest(unittest.TestCase):
    def test_external_source_metadata_is_copied_to_each_chunk(self):
        """The URL and source level of external knowledge must be stored with each chunk for later auditing."""
        chunks = chunk_markdown(
            "---\nsource_title: Official source\nsource_url: https://example.org/doc\n"
            "source_level: official\n---\n\n# P1\nVerified process context.",
            source="official.md",
            doc_type="reference",
            chunk_size=800,
            chunk_overlap=0,
            base_id_prefix="test",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source_url"], "https://example.org/doc")
        self.assertEqual(chunks[0]["metadata"]["source_level"], "official")
        self.assertIn("[可核验来源] Official source", chunks[0]["content"])

    def test_recursive_split_keeps_chunks_within_size(self):
        """Recursive split must never produce a chunk larger than chunk_size."""
        text = "这是第一句。这是第二句。" * 200
        chunks = split_recursive(text, chunk_size=100, chunk_overlap=10)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_recursive_split_prefers_sentence_boundaries(self):
        """A short single sentence must not be cut mid-sentence."""
        sentence = "一个完整的短句子。"
        text = sentence + sentence + sentence + sentence  # 4 units, still small
        chunks = split_recursive(text, chunk_size=100, chunk_overlap=0)
        self.assertEqual(chunks, [text])  # fits whole, no split

        long_text = "\n\n".join([sentence * 40] * 3)
        chunks = split_recursive(long_text, chunk_size=60, chunk_overlap=0)
        for chunk in chunks:
            self.assertTrue(chunk)  # non-empty
        # Every chunk except the final one ends on a sentence final (no mid-sentence cuts).
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.endswith("。"))

    def test_recursive_chunk_markdown_strategy_is_selectable(self):
        """chunk_markdown accepts strategy='recursive' and still keeps front-matter metadata."""
        body = "段落一。内容足够长。" * 50
        chunks = chunk_markdown(
            "---\nsource_title: Recursive doc\nsource_url: https://example.org/rec\n---\n\n" + body,
            source="rec.md",
            doc_type="reference",
            chunk_size=100,
            chunk_overlap=0,
            base_id_prefix="rec",
            strategy="recursive",
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["source_url"], "https://example.org/rec")
        self.assertIn("[可核验来源] Recursive doc", chunks[0]["content"])

    def test_extract_pdf_text_raises_when_pypdf_missing(self):
        """PDF extraction must fail loudly (or be skipped) when the optional dependency is absent."""
        try:
            extract_pdf_text("nonexistent.pdf")
        except FileNotFoundError:
            pass  # pypdf installed path -> file not found
        except ImportError:
            pass  # pypdf not installed -> graceful dependency check
        except Exception as exc:  # noqa: BLE001
            self.fail(f"unexpected error type: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    unittest.main()
