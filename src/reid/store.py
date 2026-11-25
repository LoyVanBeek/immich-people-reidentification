"""The local store: an SQLite database persisting everything the pipeline needs.

This slice persists only the new-asset feed checkpoint; later tickets add
embeddings, proposals, dry-run batches, the write ledger, and tombstones.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .enums import ProposalStatus

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
    captured_at: datetime | None  # asset capture time; None for rows without one

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.box_left, self.box_top, self.box_right, self.box_bottom)


@dataclass(frozen=True)
class Proposal:
    """One run's conclusion for one appearance: a proposal or its blocking."""

    asset_id: str
    appearance: int
    window: str  # timespan window key the cluster was built in
    cluster: int  # zero-based cluster index within the window
    status: ProposalStatus
    proposed_name: str | None  # the unanimous anchor name, when proposed


@dataclass(frozen=True)
class WriteLedgerEntry:
    """One confirmed write: (asset_id, person_id) pair that was written to Immich."""

    asset_id: str
    person_id: str
    cluster: int
    window: str
    timestamp: datetime  # wall-clock UTC instant of the write


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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appearance_rejection (
                    asset_id TEXT NOT NULL,
                    appearance INTEGER NOT NULL,
                    proposed_name TEXT NOT NULL,
                    PRIMARY KEY (asset_id, appearance, proposed_name)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_dismissal (
                    window TEXT NOT NULL,
                    cluster INTEGER NOT NULL,
                    PRIMARY KEY (window, cluster)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_confirmation (
                    window TEXT NOT NULL,
                    cluster INTEGER NOT NULL,
                    PRIMARY KEY (window, cluster)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appearance_pin (
                    asset_id TEXT NOT NULL,
                    appearance INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    PRIMARY KEY (asset_id, appearance)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appearance_ignore (
                    asset_id TEXT NOT NULL,
                    appearance INTEGER NOT NULL,
                    PRIMARY KEY (asset_id, appearance)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_status (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    pid INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    windows_done INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tuning (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.commit()
            self._migrate()
        except sqlite3.Error as exc:
            raise StoreError(
                f"cannot initialize the local store at {path}: {exc}"
            ) from exc

    def _migrate(self) -> None:
        """Add columns introduced after a store was first created.

        Stores written before capture times were persisted get an empty
        captured_at; those appearances are excluded from clustering until
        the store is wiped and rebuilt.
        """
        columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(appearance)")
        }
        if "captured_at" not in columns:
            self._conn.execute(
                "ALTER TABLE appearance ADD COLUMN captured_at TEXT NOT NULL DEFAULT ''"
            )
        # Retired tables (dry-run batches, write decisions) are dropped, not
        # migrated: their contents were ephemeral approvals, and the proposals
        # they were built from survive in the proposal table.
        for table in ("dry_run_batch", "dry_run_action", "write_decision"):
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")

    def get_checkpoint(self, feed: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM checkpoint WHERE feed = ?", (feed,)
        ).fetchone()
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
                        row.captured_at.isoformat() if row.captured_at else "",
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
            captured_at_str = str(row[9])
            captured_at = (
                datetime.fromisoformat(captured_at_str) if captured_at_str else None
            )
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
                captured_at=captured_at,
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
                        row.status.value,
                        row.proposed_name,
                    )
                    for row in rows
                ],
            )

    def update_proposal_status(
        self, asset_id: str, appearance: int, status: ProposalStatus
    ) -> None:
        """Change one proposal's status (e.g. operator rejected it)."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE proposal SET status = ? WHERE asset_id = ? AND appearance = ?",
                (status.value, asset_id, appearance),
            )
            if cursor.rowcount == 0:
                raise StoreError(f"no proposal for {asset_id}#{appearance}")

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
                status=ProposalStatus(str(row[4])),
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
                (
                    entry.asset_id,
                    entry.person_id,
                    entry.cluster,
                    entry.window,
                    entry.timestamp.isoformat(),
                ),
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
                timestamp=datetime.fromisoformat(str(row[4])),
            )
            for row in rows
        ]

    def write_ledger(self) -> list[WriteLedgerEntry]:
        """All write ledger entries (alias for get_write_ledger_entries)."""
        return self.get_write_ledger_entries()

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

    def tombstones(self) -> list[Tombstone]:
        """All tombstoned asset-person pairs."""
        rows = self._conn.execute(
            "SELECT asset_id, person_id FROM tombstone"
        ).fetchall()
        return [Tombstone(asset_id=str(row[0]), person_id=str(row[1])) for row in rows]

    # -- wipe ----------------------------------------------------------------

    def wipe(self) -> int:
        """Erase everything the pipeline stored; returns the rows removed.

        A batch-run/config-level operation: the confirmation UI never
        triggers a wipe. Clearing the checkpoint restarts the feed, so the
        next run re-fetches and re-embeds every photo from scratch.

        Review decisions survive a wipe: rejections, cluster dismissals and
        tuning knobs are the operator's voice, not pipeline state.
        """
        with self._conn:
            removed = self._conn.execute("SELECT count(*) FROM appearance").fetchone()[
                0
            ]
            self._conn.execute("DELETE FROM appearance")
            self._conn.execute("DELETE FROM proposal")
            self._conn.execute("DELETE FROM write_ledger")
            self._conn.execute("DELETE FROM tombstone")
            self._conn.execute("DELETE FROM checkpoint")
        return int(removed)

    # -- review decisions -----------------------------------------------------

    def add_appearance_rejection(
        self, asset_id: str, appearance: int, proposed_name: str
    ) -> None:
        """Record 'this appearance is not this person' for a proposal name.

        Survives batch runs: matching proposals are never created again, and
        confirmations skip the pair.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO appearance_rejection (asset_id, appearance, proposed_name)
                VALUES (?, ?, ?)
                ON CONFLICT(asset_id, appearance, proposed_name) DO NOTHING
                """,
                (asset_id, appearance, proposed_name),
            )

    def appearance_rejections(self) -> list[tuple[str, int, str]]:
        """All (asset_id, appearance, proposed_name) rejections."""
        rows = self._conn.execute(
            "SELECT asset_id, appearance, proposed_name FROM appearance_rejection"
        ).fetchall()
        return [(str(r[0]), int(r[1]), str(r[2])) for r in rows]

    def remove_appearance_rejection(
        self, asset_id: str, appearance: int, proposed_name: str
    ) -> None:
        """Undo a rejection: the pair may be proposed again."""
        with self._conn:
            self._conn.execute(
                """
                DELETE FROM appearance_rejection
                WHERE asset_id = ? AND appearance = ? AND proposed_name = ?
                """,
                (asset_id, appearance, proposed_name),
            )

    def set_pin(self, asset_id: str, appearance: int, person_name: str) -> None:
        """Record 'this appearance is this person' as an operator decision.

        Survives batch runs: anchoring always groups the appearance with the
        pinned person (or surfaces it anchored to them), whatever the
        embeddings say. Assignments flow through the normal proposal path —
        nothing writes to Immich here.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO appearance_pin (asset_id, appearance, person_name)
                VALUES (?, ?, ?)
                ON CONFLICT(asset_id, appearance) DO UPDATE
                SET person_name = excluded.person_name
                """,
                (asset_id, appearance, person_name),
            )

    def delete_pin(self, asset_id: str, appearance: int) -> None:
        """Remove the operator's pin; normal anchoring applies again."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM appearance_pin WHERE asset_id = ? AND appearance = ?",
                (asset_id, appearance),
            )

    def appearance_pins(self) -> dict[tuple[str, int], str]:
        """All pins as (asset_id, appearance) -> person_name."""
        rows = self._conn.execute(
            "SELECT asset_id, appearance, person_name FROM appearance_pin"
        ).fetchall()
        return {(str(r[0]), int(r[1])): str(r[2]) for r in rows}

    def ignore_appearance(self, asset_id: str, appearance: int) -> None:
        """Record 'this detection is wrong — never label it'.

        Survives batch runs and wipes: the appearance is excluded from
        clustering entirely (never proposed, never ask-for-name) and
        confirmations skip it.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO appearance_ignore (asset_id, appearance)
                VALUES (?, ?)
                ON CONFLICT(asset_id, appearance) DO NOTHING
                """,
                (asset_id, appearance),
            )

    def unignore_appearance(self, asset_id: str, appearance: int) -> None:
        """Remove the ignore; normal clustering applies again."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM appearance_ignore WHERE asset_id = ? AND appearance = ?",
                (asset_id, appearance),
            )

    def appearance_ignores(self) -> set[tuple[str, int]]:
        """All ignored (asset_id, appearance) pairs."""
        rows = self._conn.execute(
            "SELECT asset_id, appearance FROM appearance_ignore"
        ).fetchall()
        return {(str(r[0]), int(r[1])) for r in rows}

    # -- pipeline runs -------------------------------------------------------

    def start_run(
        self, run_id: str, trigger: str, pid: int | None, started_at: datetime
    ) -> None:
        """Record a pipeline run starting (the job manager owns exactly one)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO run_status (id, state, trigger, pid, started_at)
                VALUES (?, 'running', ?, ?, ?)
                """,
                (run_id, trigger, pid, started_at.isoformat()),
            )

    def update_run_progress(self, run_id: str, windows_done: int) -> None:
        """Record per-window progress of a running pipeline job."""
        with self._conn:
            self._conn.execute(
                "UPDATE run_status SET windows_done = ? WHERE id = ?",
                (windows_done, run_id),
            )

    def finish_run(
        self, run_id: str, state: str, finished_at: datetime, error: str | None = None
    ) -> None:
        """Record a run reaching its end state (done/failed/cancelled)."""
        with self._conn:
            self._conn.execute(
                "UPDATE run_status SET state = ?, finished_at = ?, error = ? WHERE id = ?",
                (state, finished_at.isoformat(), error, run_id),
            )

    def mark_interrupted_runs(self) -> None:
        """Close runs a previous core process left open (crash/restart).

        A run row only means something while its subprocess is alive; at
        core startup none is, so a leftover 'running' row is stale.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE run_status SET state = 'interrupted', finished_at = ?
                WHERE state = 'running'
                """,
                (datetime.now(UTC).isoformat(),),
            )

    def latest_run(self) -> dict[str, Any] | None:
        """The most recent run, for status display."""
        row = self._conn.execute(
            """
            SELECT id, state, trigger, pid, started_at, finished_at,
                   windows_done, error
            FROM run_status ORDER BY started_at DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "state": str(row[1]),
            "trigger": str(row[2]),
            "pid": row[3],
            "started_at": str(row[4]),
            "finished_at": row[5],
            "windows_done": int(row[6]),
            "error": row[7],
        }

    def dismiss_cluster(self, window: str, cluster: int, dismissed: bool) -> None:
        """Persist (or clear) 'I don't care about this cluster'."""
        with self._conn:
            if dismissed:
                self._conn.execute(
                    """
                    INSERT INTO cluster_dismissal (window, cluster)
                    VALUES (?, ?)
                    ON CONFLICT(window, cluster) DO NOTHING
                    """,
                    (window, cluster),
                )
            else:
                self._conn.execute(
                    "DELETE FROM cluster_dismissal WHERE window = ? AND cluster = ?",
                    (window, cluster),
                )

    def dismissed_clusters(self) -> set[tuple[str, int]]:
        """All (window, cluster) pairs the operator dismissed."""
        rows = self._conn.execute(
            "SELECT window, cluster FROM cluster_dismissal"
        ).fetchall()
        return {(str(r[0]), int(r[1])) for r in rows}

    def confirm_cluster(self, window: str, cluster: int, confirmed: bool) -> None:
        """Persist (or clear) 'this proposal is confirmed' — a review decision.

        Survives batch runs and wipes, like dismissals: the cluster leaves the
        open queue, and its appearances surface in the Final check tab for the
        actual write.
        """
        with self._conn:
            if confirmed:
                self._conn.execute(
                    """
                    INSERT INTO cluster_confirmation (window, cluster)
                    VALUES (?, ?)
                    ON CONFLICT(window, cluster) DO NOTHING
                    """,
                    (window, cluster),
                )
            else:
                self._conn.execute(
                    "DELETE FROM cluster_confirmation WHERE window = ? AND cluster = ?",
                    (window, cluster),
                )

    def confirmed_clusters(self) -> set[tuple[str, int]]:
        """All (window, cluster) pairs the operator confirmed."""
        rows = self._conn.execute(
            "SELECT window, cluster FROM cluster_confirmation"
        ).fetchall()
        return {(str(r[0]), int(r[1])) for r in rows}

    def tuning_values(self) -> dict[str, str]:
        """Operator tuning overrides (key → raw value)."""
        rows = self._conn.execute("SELECT key, value FROM tuning").fetchall()
        return {str(r[0]): str(r[1]) for r in rows}

    def set_tuning_value(self, key: str, value: str) -> None:
        """Persist one tuning override."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tuning (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
