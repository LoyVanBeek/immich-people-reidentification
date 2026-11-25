"""Tests for the runtime git-commit indicator (version.py)."""

from __future__ import annotations

import subprocess

from reid.version import git_commit


def test_git_commit_matches_repo_head() -> None:
    # Tests run from a git checkout, so the dev fallback must resolve HEAD.
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert git_commit() == proc.stdout.strip()


def test_git_commit_is_cached_and_nonempty() -> None:
    assert git_commit() == git_commit()
    assert git_commit()
