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
    captured_at: str = ""  # asset capture time; upload fallback resolved by the client

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.box_left, self.box_top, self.box_right, self.box_bottom)


# Proposal statuses: what a run concluded about one appearance. Singletons
# get no proposal row at all; anchor-less clusters below the minimum do not
# either.
PROPOSED = "proposed"
ASK_NAME = "ask_name"
CONFLICTED = "conflicted"
LABELED = "labeled"


@dataclass(frozen=True)
class Proposal:
    """One run's conclusion for one appearance: a proposal or its blocking."""

    asset_id: str
    appearance: int
    window: str  # timespan window key the cluster was built in
    cluster: int  # zero-based cluster index within the window
    status: str  # one of PROPOSAL_STATUSES
    proposed_name: str | None  # the unanimous anchor name, when proposed


@dataclass(frozen=True)
class WriteLedgerEntry:
    """One confirmed write: (asset_id, person_id) pair that was written to Immich."""

    asset_id: str
    person_id: str
    cluster: int
    window: str
    timestamp: str  # ISO timestamp of the write


@dataclass(frozen=True)
class Tombstone:
    """A label the user removed in Immich: never re-propose this asset-person pair."""

    asset_id: str
    person_id: str


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
                    captured_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (asset_id, appearance)
                )
                """
            )
            self._migrate()
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposal (
                    asset_id TEXT NOT NULL,
                    appearance INTEGER NOT NULL,
                    window TEXT NOT NULL,
                    cluster INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    proposed_name TEXT,
                    PRIMARY KEY (asset_id, appearance)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS write_ledger (
                    asset_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    cluster INTEGER NOT NULL,
                    window TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (asset_id, person_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tombstone (
                    asset_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    PRIMARY KEY (asset_id, person_id)
                )
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"cannot initialize the local store at {path}: {exc}") from exc

    def _migrate(self) -> None:
        """Add columns introduced after a store was first created.

        Stores written before capture times were persisted get an empty
        captured_at; those appearances are excluded from clustering until
        the store is wiped and rebuilt.
        """
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(appearance)")}
        if "captured_at" not in columns:
            self._conn.execute("ALTER TABLE appearance ADD COLUMN captured_at TEXT NOT NULL DEFAULT ''")

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
                    box_left, box_top, box_right, box_bottom, pixel_source, captured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        row.captured_at,
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
                captured_at=str(row[9]),
            )
            by_asset.setdefault(appearance.asset_id, []).append(appearance)
        return by_asset

    # -- proposals ------------------------------------------------------------

    def replace_proposals(self, rows: list[Proposal]) -> None:
        """Persist one run's proposals, replacing any previous run's.

        Clusters are rebuilt from scratch every run, so the proposal table
        always mirrors the most recent run's conclusions.
        """
        with self._conn:
            self._conn.execute("DELETE FROM proposal")
            self._conn.executemany(
                """
                INSERT INTO proposal (asset_id, appearance, window, cluster, status, proposed_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.asset_id,
                        row.appearance,
                        row.window,
                        row.cluster,
                        row.status,
                        row.proposed_name,
                    )
                    for row in rows
                ],
            )

    def proposals(self) -> list[Proposal]:
        """All proposal rows of the most recent run."""
        rows = self._conn.execute(
            "SELECT asset_id, appearance, window, cluster, status, proposed_name FROM proposal"
        ).fetchall()
        return [
            Proposal(
                asset_id=str(row[0]),
                appearance=int(row[1]),
                window=str(row[2]),
                cluster=int(row[3]),
                status=str(row[4]),
                proposed_name=None if row[5] is None else str(row[5]),
            )
            for row in rows
        ]

    # -- write ledger ---------------------------------------------------------

    def add_write_ledger_entry(self, entry: WriteLedgerEntry) -> None:
        """Record a confirmed write to Immich."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO write_ledger (asset_id, person_id, cluster, window, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, person_id) DO NOTHING
                """,
                (entry.asset_id, entry.person_id, entry.cluster, entry.window, entry.timestamp),
            )

    def has_write_ledger_entry(self, asset_id: str, person_id: str) -> bool:
        """Check if a write was already confirmed for this asset-person pair."""
        row = self._conn.execute(
            "SELECT 1 FROM write_ledger WHERE asset_id = ? AND person_id = ?",
            (asset_id, person_id),
        ).fetchone()
        return row is not None

    def get_write_ledger_entries(self) -> list[WriteLedgerEntry]:
        """Get all write ledger entries."""
        rows = self._conn.execute(
            "SELECT asset_id, person_id, cluster, window, timestamp FROM write_ledger"
        ).fetchall()
        return [
            WriteLedgerEntry(
                asset_id=str(row[0]),
                person_id=str(row[1]),
                cluster=int(row[2]),
                window=str(row[3]),
                timestamp=str(row[4]),
            )
            for row in rows
        ]

    # -- tombstones -----------------------------------------------------------

    def add_tombstone(self, tombstone: Tombstone) -> None:
        """Record a label the user removed in Immich."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tombstone (asset_id, person_id)
                VALUES (?, ?)
                ON CONFLICT(asset_id, person_id) DO NOTHING
                """,
                (tombstone.asset_id, tombstone.person_id),
            )

    def has_tombstone(self, asset_id: str, person_id: str) -> bool:
        """Check if a tombstone exists for this asset-person pair."""
        row = self._conn.execute(
            "SELECT 1 FROM tombstone WHERE asset_id = ? AND person_id = ?",
            (asset_id, person_id),
        ).fetchone()
        return row is not None

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
            self._conn.execute("DELETE FROM proposal")
            self._conn.execute("DELETE FROM write_ledger")
            self._conn.execute("DELETE FROM tombstone")
            self._conn.execute("DELETE FROM checkpoint")
        return int(removed)
