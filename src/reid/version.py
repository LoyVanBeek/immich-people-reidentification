"""The git commit of the running code, for the UI's version indicator.

In container images the commit is baked at build time (Dockerfile build-arg
GIT_COMMIT -> src/reid/GIT_COMMIT.txt, packaged into the installed wheel).
In a dev checkout it falls back to asking git; anywhere else it is "unknown".
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Short commit hash of the running code, or "unknown"."""
    baked = Path(__file__).parent / "GIT_COMMIT.txt"
    try:
        value = baked.read_text(encoding="utf-8").strip()
        if value and value != "unknown":
            return value
    except OSError:
        pass
    # Dev checkout: repo root is two levels up from src/reid/version.py.
    repo = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "unknown"
