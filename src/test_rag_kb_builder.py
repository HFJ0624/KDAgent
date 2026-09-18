"""Offline regression test for RAG knowledge-source metadata."""

import unittest

from src.rag_kb_builder import chunk_markdown


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


if __name__ == "__main__":
    unittest.main()
