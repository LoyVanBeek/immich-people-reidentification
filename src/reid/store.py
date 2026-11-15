"""The local store: an SQLite database persisting everything the pipeline needs.

This slice persists only the new-asset feed checkpoint; later tickets add
embeddings, proposals, dry-run batches, the write ledger, and tombstones.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

# The checkpoint for the checkpointed new-asset feed.
NEW_ASSET_FEED = "new-assets"


class StoreError(Exception):
    """The store cannot be opened or written."""


@dataclass(frozen=True)
class Appearance:
    """One stored person appearance: its embedding and where it sits."""

    asset_id: str
    appearance: int  # zero-based index of the appearance within the asset
    embedding: bytes  # 512 float32 values, L2-normalized
    confidence: float  # detector confidence
    box_left: float
    box_top: float
    box_right: float
    box_bottom: float
    pixel_source: str  # "preview" or "original" — where the crop came from


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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appearance (
                    asset_id TEXT NOT NULL,
                    appearance INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    confidence REAL NOT NULL,
                    box_left REAL NOT NULL,
                    box_top REAL NOT NULL,
                    box_right REAL NOT NULL,
                    box_bottom REAL NOT NULL,
                    pixel_source TEXT NOT NULL,
                    PRIMARY KEY (asset_id, appearance)
                )
                """,
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

    # -- appearances ---------------------------------------------------------

    def add_appearances(self, asset_id: str, rows: list[Appearance]) -> None:
        """Persist the person appearances of one asset, replacing any old ones.

        A crashed run that reprocesses the asset replaces its rows wholesale,
        so the store never keeps stale or duplicate appearances.
        """
        with self._conn:
            self._conn.execute("DELETE FROM appearance WHERE asset_id = ?", (asset_id,))
            self._conn.executemany(
                """
                INSERT INTO appearance (
                    asset_id, appearance, embedding, confidence,
                    box_left, box_top, box_right, box_bottom, pixel_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        asset_id,
                        row.appearance,
                        row.embedding,
                        row.confidence,
                        row.box_left,
                        row.box_top,
                        row.box_right,
                        row.box_bottom,
                        row.pixel_source,
                    )
                    for row in rows
                ],
            )

    def appearances_by_asset(self) -> dict[str, list[Appearance]]:
        """All stored appearances, grouped by asset in appearance order."""
        rows = self._conn.execute(
            "SELECT * FROM appearance ORDER BY asset_id, appearance"
        ).fetchall()
        by_asset: dict[str, list[Appearance]] = {}
        for row in rows:
            appearance = Appearance(
                asset_id=str(row[0]),
                appearance=int(row[1]),
                embedding=bytes(row[2]),
                confidence=float(row[3]),
                box_left=float(row[4]),
                box_top=float(row[5]),
                box_right=float(row[6]),
                box_bottom=float(row[7]),
                pixel_source=str(row[8]),
            )
            by_asset.setdefault(appearance.asset_id, []).append(appearance)
        return by_asset

    # -- wipe ----------------------------------------------------------------

    def wipe(self) -> int:
        """Erase everything the pipeline stored; returns the rows removed.

        A batch-run/config-level operation: the confirmation UI never
        triggers a wipe. Clearing the checkpoint restarts the feed, so the
        next run re-fetches and re-embeds every photo from scratch.
        """
        with self._conn:
            removed = self._conn.execute("SELECT count(*) FROM appearance").fetchone()[0]
            self._conn.execute("DELETE FROM appearance")
            self._conn.execute("DELETE FROM checkpoint")
        return int(removed)
