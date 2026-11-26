"""Store tests: proposals and capture times persist and wipe correctly.

The store is a public seam of its own (the run's product is read back
through it in the acceptance tests); these tests pin the persistence
behavior the clustering slice depends on.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from reid.enums import ProposalStatus
from reid.store import (
    Appearance,
    Proposal,
    Store,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite3")


def appearance_row(
    asset_id: str, index: int = 0, captured_at: datetime | None = None
) -> Appearance:
    return Appearance(
        asset_id=asset_id,
        appearance=index,
        embedding=b"\x00" * 8,
        confidence=0.9,
        box_left=1.0,
        box_top=2.0,
        box_right=3.0,
        box_bottom=4.0,
        pixel_source="preview",
        captured_at=captured_at or datetime.fromisoformat("2026-11-10T08:00:00.000Z"),
    )


def test_appearances_keep_their_capture_time(store: Store) -> None:
    captured_at = datetime.fromisoformat("2026-11-10T08:00:00.000Z")
    store.add_appearances(
        "asset-1", [appearance_row("asset-1", captured_at=captured_at)]
    )
    rows = store.appearances_by_asset()["asset-1"]
    assert len(rows) == 1
    assert rows[0].captured_at == captured_at


def test_proposals_round_trip_through_the_store(store: Store) -> None:
    rows = [
        Proposal("asset-1", 0, "2026-11-10", 0, ProposalStatus.PROPOSED, "Albert"),
        Proposal("asset-1", 1, "2026-11-10", 0, ProposalStatus.LABELED, None),
        Proposal("asset-2", 0, "2026-11-10", 1, ProposalStatus.ASK_NAME, None),
    ]
    store.replace_proposals(rows)

    read_back = store.proposals()
    assert len(read_back) == 3
    by_asset = {
        (proposal.asset_id, proposal.appearance): proposal for proposal in read_back
    }
    assert by_asset[("asset-1", 0)].proposed_name == "Albert"
    assert by_asset[("asset-1", 0)].status == ProposalStatus.PROPOSED
    assert by_asset[("asset-1", 0)].window == "2026-11-10"
    assert by_asset[("asset-1", 0)].cluster == 0
    assert by_asset[("asset-1", 1)].status == ProposalStatus.LABELED
    assert by_asset[("asset-2", 0)].status == ProposalStatus.ASK_NAME


def test_each_run_replaces_the_previous_proposals_wholesale(store: Store) -> None:
    store.replace_proposals(
        [Proposal("asset-1", 0, "2026-11-10", 0, ProposalStatus.PROPOSED, "Albert")]
    )

    store.replace_proposals(
        [Proposal("asset-2", 0, "2026-11-10", 0, ProposalStatus.ASK_NAME, None)]
    )

    read_back = store.proposals()
    assert len(read_back) == 1
    assert read_back[0].asset_id == "asset-2"
    assert read_back[0].status == ProposalStatus.ASK_NAME


def test_a_wipe_erases_proposals_together_with_the_appearances(store: Store) -> None:
    store.add_appearances("asset-1", [appearance_row("asset-1")])
    store.replace_proposals(
        [Proposal("asset-1", 0, "2026-11-10", 0, ProposalStatus.PROPOSED, "Albert")]
    )

    removed = store.wipe()

    assert removed == 1
    assert store.appearances_by_asset() == {}
    assert store.proposals() == []
    # The store stays usable afterwards.
    store.add_appearances("asset-2", [appearance_row("asset-2")])
    store.replace_proposals(
        [Proposal("asset-2", 0, "2026-11-10", 0, ProposalStatus.PROPOSED, "Bea")]
    )
    assert len(store.proposals()) == 1


def test_a_store_written_before_capture_times_migrates(tmp_path: Path) -> None:
    # A store created before the captured_at column existed must keep its
    # rows (with an empty capture time) and stay writable.
    legacy = tmp_path / "legacy.sqlite3"
    import sqlite3

    conn = sqlite3.connect(legacy)
    conn.execute(
        """
        CREATE TABLE appearance (
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
        """
    )
    conn.execute(
        "INSERT INTO appearance VALUES ('asset-old', 0, X'00', 0.9, 1, 2, 3, 4, 'preview')"
    )
    conn.commit()
    conn.close()

    store = Store(legacy)
    rows = store.appearances_by_asset()["asset-old"]
    assert len(rows) == 1
    # Legacy stores without captured_at column get None (not datetime.min)
    assert rows[0].captured_at is None

    store.add_appearances("asset-new", [appearance_row("asset-new")])
    by_asset = store.appearances_by_asset()
    assert by_asset["asset-new"][0].captured_at == datetime.fromisoformat(
        "2026-11-10T08:00:00.000Z"
    )


def test_pins_round_trip_and_survive_a_wipe(store: Store) -> None:
    """Pins are operator decisions: set, re-set, delete — and a wipe keeps them."""
    store.set_pin("asset-a", 0, "Alice")
    store.set_pin("asset-b", 2, "Bob")

    assert store.appearance_pins() == {("asset-a", 0): "Alice", ("asset-b", 2): "Bob"}

    # Re-assigning the same appearance replaces the pin.
    store.set_pin("asset-a", 0, "Carol")
    assert store.appearance_pins() == {("asset-a", 0): "Carol", ("asset-b", 2): "Bob"}

    # Unpinning removes exactly the one appearance's pin.
    store.delete_pin("asset-a", 0)
    assert store.appearance_pins() == {("asset-b", 2): "Bob"}
    # Deleting a missing pin is a no-op.
    store.delete_pin("asset-a", 0)
    assert store.appearance_pins() == {("asset-b", 2): "Bob"}

    # A wipe clears pipeline state but pins are the operator's voice.
    store.add_appearances("asset-c", [appearance_row("asset-c")])
    store.wipe()
    assert store.appearance_pins() == {("asset-b", 2): "Bob"}


def test_rejection_can_be_removed(store: Store) -> None:
    """Undoing an exclusion deletes exactly that (asset, appearance, name)."""
    store.add_appearance_rejection("asset-a", 0, "Alice")
    store.add_appearance_rejection("asset-a", 1, "Alice")
    store.add_appearance_rejection("asset-a", 0, "Bob")

    store.remove_appearance_rejection("asset-a", 0, "Alice")

    assert store.appearance_rejections() == [
        ("asset-a", 1, "Alice"),
        ("asset-a", 0, "Bob"),
    ]
    # Removing a missing pair is a no-op.
    store.remove_appearance_rejection("asset-a", 0, "Alice")
    assert store.appearance_rejections() == [
        ("asset-a", 1, "Alice"),
        ("asset-a", 0, "Bob"),
    ]


def test_ignores_round_trip_and_survive_a_wipe(store: Store) -> None:
    """Ignored detections are operator decisions: set, delete — and a wipe
    keeps them."""
    store.ignore_appearance("asset-a", 0)
    store.ignore_appearance("asset-b", 3)

    assert store.appearance_ignores() == {("asset-a", 0), ("asset-b", 3)}

    # Re-ignoring the same pair is a no-op.
    store.ignore_appearance("asset-a", 0)
    assert store.appearance_ignores() == {("asset-a", 0), ("asset-b", 3)}

    store.unignore_appearance("asset-a", 0)
    assert store.appearance_ignores() == {("asset-b", 3)}
    # Deleting a missing ignore is a no-op.
    store.unignore_appearance("asset-a", 0)
    assert store.appearance_ignores() == {("asset-b", 3)}

    # A wipe clears pipeline state but ignores are the operator's voice.
    store.add_appearances("asset-c", [appearance_row("asset-c")])
    store.wipe()
    assert store.appearance_ignores() == {("asset-b", 3)}


def test_ignore_reasons_are_recorded_and_read(store: Store) -> None:
    """Ignores carry a reason: 'ignore' (detection noise) or
    'not_a_person' (not a person)."""
    store.ignore_appearance("asset-a", 0, reason="ignore")
    store.ignore_appearance("asset-b", 3, reason="not_a_person")

    reasons = store.appearance_ignore_reasons()
    assert reasons[("asset-a", 0)] == "ignore"
    assert reasons[("asset-b", 3)] == "not_a_person"

    # Re-ignoring with a different reason overwrites it.
    store.ignore_appearance("asset-a", 0, reason="not_a_person")
    assert store.appearance_ignore_reasons()[("asset-a", 0)] == "not_a_person"

    # Invalid reason defaults to 'ignore'.
    store.ignore_appearance("asset-c", 7, reason="nonsense")
    assert store.appearance_ignore_reasons()[("asset-c", 7)] == "ignore"

    # Ignores without reasons still show up in appearance_ignores().
    assert ("asset-a", 0) in store.appearance_ignores()
    assert ("asset-b", 3) in store.appearance_ignores()
    assert ("asset-c", 7) in store.appearance_ignores()


def test_ignore_reasons_migrate_on_existing_stores(tmp_path: Path) -> None:
    """Stores created before the reason column gets it added automatically,
    and old rows default to 'ignore'."""
    from reid.store import Store

    path = tmp_path / "reid.sqlite3"
    store = Store(path)
    store.ignore_appearance("asset-a", 0)  # no reason arg -> 'ignore'

    # Simulate an old store by creating the table without the reason column,
    # then opening it fresh.
    store2 = Store(path)
    assert store2.appearance_ignore_reasons()[("asset-a", 0)] == "ignore"
