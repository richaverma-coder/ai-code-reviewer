"""Tests for review comment generation."""

import unittest

from app.ai_reviewer import review_issues
from app.analyzer import Issue


class ReviewerTests(unittest.TestCase):
    def test_builds_comment_from_issue(self):
        issue = Issue(
            file_path="demo.py",
            line=3,
            column=0,
            severity="warning",
            rule="bare-except",
            message="Bare except.",
            suggestion="Catch a specific exception.",
        )
        comments = review_issues([issue])
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].title, "Bare except clause")
        self.assertIn("Catch a specific exception.", comments[0].body)


if __name__ == "__main__":
    unittest.main()
