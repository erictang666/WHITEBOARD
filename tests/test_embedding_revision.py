from __future__ import annotations

import unittest
from unittest.mock import patch

import semantic_scorer


class EmbeddingRevisionTests(unittest.TestCase):
    def test_default_model_uses_immutable_revision(self):
        with patch.object(semantic_scorer, "get_hf_cache_hint", return_value="/not/cached"):
            with patch.object(semantic_scorer, "SentenceTransformer") as constructor:
                semantic_scorer.SemanticScorer()
        constructor.assert_called_once()
        args, kwargs = constructor.call_args
        self.assertEqual(args[0], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(
            kwargs["revision"],
            "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        )
        self.assertEqual(kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
