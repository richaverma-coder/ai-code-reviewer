"""Tests for the local Ollama reviewer."""

import unittest
from unittest.mock import patch

from app.ollama_reviewer import OllamaError, ensure_ollama_ready, review_with_ollama


class OllamaReviewerTests(unittest.TestCase):
    def test_connection_error_is_clear(self):
        with patch(
            "app.ollama_reviewer._request_json",
            side_effect=OllamaError(
                "Could not connect to Ollama at http://127.0.0.1:11434. "
                "Install Ollama from https://ollama.com"
            ),
        ):
            with self.assertRaises(OllamaError) as ctx:
                ensure_ollama_ready("http://127.0.0.1:11434", "llama3.2")
        self.assertIn("Ollama", str(ctx.exception))
        self.assertIn("https://ollama.com", str(ctx.exception))

    def test_missing_model_is_clear(self):
        with patch(
            "app.ollama_reviewer._request_json",
            return_value={"models": [{"name": "other:latest"}]},
        ):
            with self.assertRaises(OllamaError) as ctx:
                ensure_ollama_ready("http://127.0.0.1:11434", "llama3.2")
        message = str(ctx.exception)
        self.assertIn("llama3.2", message)
        self.assertIn("ollama pull llama3.2", message)

    def test_review_requires_source(self):
        with self.assertRaises(OllamaError):
            review_with_ollama("")


if __name__ == "__main__":
    unittest.main()
