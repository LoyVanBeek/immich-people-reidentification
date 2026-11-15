"""The local store: an SQLite database persisting everything the pipeline needs.

This slice persists only the new-asset feed checkpoint; later tickets add
embeddings, proposals, dry-run batches, the write ledger, and tombstones.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# The checkpoint for the checkpointed new-asset feed.
NEW_ASSET_FEED = "new-assets"


class StoreError(Exception):
    """The store cannot be opened or written."""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(path)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open the local store at {path}: {exc}") from exc
        try:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint (feed TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"cannot initialize the local store at {path}: {exc}") from exc

    def get_checkpoint(self, feed: str) -> str | None:
        row = self._conn.execute("SELECT value FROM checkpoint WHERE feed = ?", (feed,)).fetchone()
        return None if row is None else str(row[0])

    def set_checkpoint(self, feed: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO checkpoint (feed, value) VALUES (?, ?)
                ON CONFLICT(feed) DO UPDATE SET value = excluded.value
                """,
                (feed, value),
            )
