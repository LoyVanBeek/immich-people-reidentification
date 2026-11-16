"""Acceptance tests for ticket 10: detect, gate, and embed person appearances.

All tests drive the real batch entry point (`reid-batch --config ...`)
against the stub Immich HTTP server, with fixture photos going through the
real YOLO11n detector and the real OSNet-AIN x1.0 embedder. The run's
product at this stage is the persisted appearance store, read back here
through the store's public API.
"""

from __future__ import annotations

from click.testing import Result
from reid.store import Store

from . import library
from .conftest import run_batch
from .stub_immich import StubImmich

# Counts pinned against the vendored weights on the fixture photos: a
# deterministic model over a fixed image must reproduce them (see the
# detection probe in the ticket comments).
EXPECTED_APPEARANCES = {
    library.PHOTO_1_STANDING_ID: 1,  # back to the camera: still detected
    library.PHOTO_2_BLURRED_ID: 1,  # blurred face: still detected
    library.PHOTO_3_GROUP_ID: 2,  # two people plus a low-confidence noise box
    library.PHOTO_TRIO_ID: 3,  # three detected people -> three appearances
    library.PHOTO_EDGE_CLIPPED_ID: 1,  # clipped by the right image edge: kept
}


def appearances(config_path) -> dict:
    """The run's product: stored appearances grouped by asset."""
    store_path = config_path.parent / "reid.sqlite3"
    return Store(store_path).appearances_by_asset()


def test_every_new_photo_contributes_person_appearances(
    detection_stub: StubImmich, make_config
) -> None:
    config = make_config(detection_stub)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    by_asset = appearances(config)
    for asset_id, expected_count in EXPECTED_APPEARANCES.items():
        rows = by_asset.get(asset_id, [])
        assert len(rows) == expected_count, (
            f"{asset_id}: expected {expected_count} appearances, got {len(rows)}"
        )


def test_run_is_incremental_only_new_assets_are_processed(
    detection_stub: StubImmich, make_config
) -> None:
    config = make_config(detection_stub)

    first = run_batch(config)
    assert first.exit_code == 0, first.output
    before = appearances(config)

    # Three photos land after the first run checkpointed.
    detection_stub.assets.extend(library.detection_late_photos())

    second = run_batch(config)
    assert second.exit_code == 0, second.output

    after = appearances(config)
    new_ids = {photo["id"] for photo in library.detection_late_photos()}
    for asset_id in new_ids:
        assert after.get(asset_id), f"{asset_id} was not processed"
    for asset_id in EXPECTED_APPEARANCES:
        assert len(after.get(asset_id, [])) == len(before.get(asset_id, [])), (
            f"{asset_id} was reprocessed"
        )


def test_edge_clipped_person_box_touches_the_image_edge(
    detection_stub: StubImmich, make_config
) -> None:
    config = make_config(detection_stub)
    result = run_batch(config)
    assert result.exit_code == 0, result.output
    rows = appearances(config)[library.PHOTO_EDGE_CLIPPED_ID]
    assert len(rows) == 1
    assert rows[0].box_right >= 1440 - 2, "box not at the image edge"


def test_quality_gates_run_in_order_and_counts_are_logged(
    detection_stub: StubImmich, make_config
) -> None:
    # Floor at maximum: every detection drops at the FIRST gate, so the
    # second gate can never claim the same detection.
    confident = make_config(detection_stub, detection={"confidence_floor": 1.0})
    result = run_batch(confident)
    assert result.exit_code == 0, result.output
    assert appearances(confident) == {}
    assert "gate drops: low_confidence=" in result.output
    assert "below_min_height=0" in result.output, (
        "a detection dropped by the first gate was double-counted"
    )

    # Height minimum above every person box: survivors drop at the SECOND
    # gate with the first gate never involved. A named config gets a fresh
    # store, so the second run starts from an empty checkpoint.
    tall = make_config(
        detection_stub, detection={"min_box_height": 100000}, name="tall"
    )
    result = run_batch(tall)
    assert result.exit_code == 0, result.output
    assert appearances(tall) == {}
    assert "below_min_height=0" not in result.output, (
        "every survivor must reach the height gate"
    )
    dropped = int(result.output.split("below_min_height=")[1].split()[0])
    assert dropped > 0, "no drop count was logged for the height gate"


def test_store_wipe_erases_embeddings_and_resets_the_feed(
    detection_stub: StubImmich, make_config
) -> None:
    config = make_config(detection_stub)
    assert run_batch(config).exit_code == 0, "run must embed before the wipe"
    assert appearances(config), "the run must have stored appearances"

    wiped = wipe(config)
    assert wiped.exit_code == 0, wiped.output
    assert appearances(config) == {}

    # Starting clean means the feed starts over, too: no createdAfter filter.
    again = run_batch(config)
    assert again.exit_code == 0, again.output
    bodies = [r.body for r in detection_stub.search_requests()]
    assert "createdAfter" not in bodies[-1]
    for asset_id in library.photo_ids():
        assert asset_id in again.output


def wipe(config_path) -> Result:
    from click.testing import CliRunner
    from reid.wipe import main

    return CliRunner().invoke(
        main, ["--config", str(config_path)], catch_exceptions=False
    )
