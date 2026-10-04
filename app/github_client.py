"""GitHub client placeholder.

GitHub integration is intentionally not implemented in this version.
Local Python analysis is the current focus.
"""

from __future__ import annotations


class GitHubClient:
    """Reserved for a later version that can comment on pull requests."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def fetch_pull_request(self, repo: str, number: int) -> None:
        self._not_implemented("fetch_pull_request")

    def post_review_comment(self, repo: str, number: int, body: str) -> None:
        self._not_implemented("post_review_comment")

    def _not_implemented(self, action: str) -> None:
        raise NotImplementedError(
            f"GitHub action `{action}` is not available yet. "
            "Run local analysis with: python main.py sample_code"
        )
