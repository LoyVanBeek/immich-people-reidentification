"""Acceptance tests for ticket 11: cluster appearances and anchor them.

All tests drive the real batch entry point (`reid-batch --config ...`)
against the stub Immich HTTP server, with fixture photos going through the
real detector and embedder. With `distance_threshold = 0` only identical
embeddings cluster, so the shared indoor and park fixture photos give
deterministic multi-appearance clusters:
indoor photos (1 appearance each) form one 5-appearance cluster, park
photos another 5-appearance cluster; every distinct photo is a singleton
(8 in total across the detection registry).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

from reid.clustering import capture_day
from reid.enums import ProposalStatus
from reid.store import Store

from . import library
from .conftest import run_batch
from .stub_immich import StubImmich

# Face boxes fully inside the person boxes the vendored YOLO11n finds on
# the fixture photos (probed with the vendored weights).
INDOOR_FACE_BOX = (600.0, 200.0, 800.0, 400.0)
# Only identical embeddings cluster: the same fixture photo produces the
# same embedding, and identical pairs sit at float32-epsilon distance
# (~1e-7) while the nearest genuinely different pair sits at 0.295.
CLUSTERING = {"distance_threshold": 0.01}


def face_payload(
    asset_id: str,
    person: tuple[str, str] | None,
    box: tuple[float, float, float, float],
) -> dict[str, Any]:
    """One GET /faces detection: person (id, name) or None when unassigned."""
    return {
        "id": f"face-{asset_id}",
        "assetId": asset_id,
        "person": None if person is None else {"id": person[0], "name": person[1]},
        "boundingBoxX1": box[0],
        "boundingBoxY1": box[1],
        "boundingBoxX2": box[2],
        "boundingBoxY2": box[3],
        "imageWidth": 1440,
        "imageHeight": 1440,
        "sourceType": "machine-learning",
    }


@pytest.fixture
def reid_stub() -> Iterator[StubImmich]:
    """Every fixture photo in one registry, late photos included."""
    server = StubImmich(
        library.detection_registry()
        + library.late_photos()
        + library.detection_late_photos()
    )
    server.start()
    yield server
    server.stop()


def store_of(config_path: Path) -> Store:
    return Store(config_path.parent / f"{config_path.stem}.sqlite3")


def window_line(result: Result) -> str:
    lines = [line for line in result.output.splitlines() if line.startswith("window ")]
    assert len(lines) == 1, result.output
    return lines[0]


def proposals_by_asset(config_path: Path) -> dict[tuple[str, int], Any]:
    return {
        (proposal.asset_id, proposal.appearance): proposal
        for proposal in store_of(config_path).proposals()
    }


def test_unanimous_anchor_proposes_the_name_for_every_unlabeled_cluster_member(
    reid_stub: StubImmich, make_config: Any
) -> None:
    reid_stub.faces[library.PHOTO_4_INDOOR_ID] = [
        face_payload(
            library.PHOTO_4_INDOOR_ID, ("person-albert", "Albert"), INDOOR_FACE_BOX
        )
    ]
    config = make_config(reid_stub, clustering=CLUSTERING)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    # The anchor's own appearance is labeled: never proposed. The rest of
    # the indoor cluster proposes Albert; the anchor-less park cluster
    # (5 appearances >= minimum 2) enters the ask-for-name queue.
    rows = proposals_by_asset(config)
    assert rows[(library.PHOTO_4_INDOOR_ID, 0)].status == ProposalStatus.LABELED
    assert rows[(library.PHOTO_4_INDOOR_ID, 0)].proposed_name is None
    for asset_id in (
        library.PHOTO_5_INDOOR_ID,
        library.PHOTO_6_INDOOR_ID,
        library.PHOTO_7_INDOOR_ID,
        library.PHOTO_12_LATE_ID,
    ):
        assert rows[(asset_id, 0)].status == ProposalStatus.PROPOSED, asset_id
        assert rows[(asset_id, 0)].proposed_name == "Albert"
    for asset_id in (
        library.PHOTO_8_PARK_ID,
        library.PHOTO_9_PARK_ID,
        library.PHOTO_10_PARK_ID,
        library.PHOTO_11_LATE_ID,
        library.PHOTO_13_LATE_ID,
    ):
        assert rows[(asset_id, 0)].status == ProposalStatus.ASK_NAME, asset_id
        assert rows[(asset_id, 0)].proposed_name is None

    # 8 distinct photos stayed singletons: stored, never surfaced.
    surfaced_assets = {asset_id for asset_id, _ in rows}
    assert library.PHOTO_1_STANDING_ID not in surfaced_assets
    assert library.PHOTO_2_BLURRED_ID not in surfaced_assets
    assert library.PHOTO_3_GROUP_ID not in surfaced_assets
    assert library.PHOTO_TRIO_ID not in surfaced_assets
    assert library.PHOTO_EDGE_CLIPPED_ID not in surfaced_assets

    assert window_line(result) == (
        "window 2026-11-10: appearances=18 clusters=2 anchored=50.0% "
        "conflicted=0.0% anchor_less=50.0% singletons=8 proposals=4 ask_for_name=5"
    ), result.output


def test_conflicting_anchor_names_block_the_whole_cluster(
    reid_stub: StubImmich, make_config: Any
) -> None:
    reid_stub.faces[library.PHOTO_4_INDOOR_ID] = [
        face_payload(
            library.PHOTO_4_INDOOR_ID, ("person-albert", "Albert"), INDOOR_FACE_BOX
        )
    ]
    reid_stub.faces[library.PHOTO_5_INDOOR_ID] = [
        face_payload(library.PHOTO_5_INDOOR_ID, ("person-bea", "Bea"), INDOOR_FACE_BOX)
    ]
    config = make_config(reid_stub, clustering=CLUSTERING)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    rows = proposals_by_asset(config)
    for asset_id in (
        library.PHOTO_4_INDOOR_ID,
        library.PHOTO_5_INDOOR_ID,
        library.PHOTO_6_INDOOR_ID,
        library.PHOTO_7_INDOOR_ID,
        library.PHOTO_12_LATE_ID,
    ):
        assert rows[(asset_id, 0)].status in (
            ProposalStatus.LABELED,
            ProposalStatus.CONFLICTED,
        ), asset_id
        assert rows[(asset_id, 0)].proposed_name is None
    assert rows[(library.PHOTO_4_INDOOR_ID, 0)].status == ProposalStatus.LABELED
    assert rows[(library.PHOTO_6_INDOOR_ID, 0)].status == ProposalStatus.CONFLICTED
    # Nothing was proposed anywhere in this run.
    assert window_line(result) == (
        "window 2026-11-10: appearances=18 clusters=2 anchored=0.0% "
        "conflicted=50.0% anchor_less=50.0% singletons=8 proposals=0 ask_for_name=5"
    ), result.output


def test_appearances_immich_already_labeled_never_receive_proposals(
    reid_stub: StubImmich, make_config: Any
) -> None:
    reid_stub.faces[library.PHOTO_4_INDOOR_ID] = [
        face_payload(
            library.PHOTO_4_INDOOR_ID, ("person-albert", "Albert"), INDOOR_FACE_BOX
        )
    ]
    # Assigned but still unnamed: labels the appearance, anchors nothing.
    reid_stub.faces[library.PHOTO_5_INDOOR_ID] = [
        face_payload(library.PHOTO_5_INDOOR_ID, ("person-unnamed", ""), INDOOR_FACE_BOX)
    ]
    config = make_config(reid_stub, clustering=CLUSTERING)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    rows = proposals_by_asset(config)
    assert rows[(library.PHOTO_5_INDOOR_ID, 0)].status == ProposalStatus.LABELED
    for asset_id in (
        library.PHOTO_6_INDOOR_ID,
        library.PHOTO_7_INDOOR_ID,
        library.PHOTO_12_LATE_ID,
    ):
        assert rows[(asset_id, 0)].status == ProposalStatus.PROPOSED, asset_id
        assert rows[(asset_id, 0)].proposed_name == "Albert"
    # One distinct name among the anchors: unanimous, not conflicted.
    assert "conflicted=0.0%" in window_line(result)


def test_anchor_less_clusters_below_the_minimum_never_surface(
    reid_stub: StubImmich, make_config: Any
) -> None:
    config = make_config(
        reid_stub, clustering={"distance_threshold": 0.01, "min_cluster_size": 6}
    )

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    # Both 5-appearance clusters sit below the minimum of 6: no queue, no
    # rows anywhere, but the summary still counts them anchor-less.
    assert proposals_by_asset(config) == {}
    assert window_line(result) == (
        "window 2026-11-10: appearances=18 clusters=2 anchored=0.0% "
        "conflicted=0.0% anchor_less=100.0% singletons=8 proposals=0 ask_for_name=0"
    ), result.output


def test_clusters_are_rebuilt_from_scratch_every_run(
    reid_stub: StubImmich, make_config: Any
) -> None:
    reid_stub.faces[library.PHOTO_4_INDOOR_ID] = [
        face_payload(
            library.PHOTO_4_INDOOR_ID, ("person-albert", "Albert"), INDOOR_FACE_BOX
        )
    ]
    config = make_config(reid_stub, clustering=CLUSTERING)

    first = run_batch(config)
    assert first.exit_code == 0, first.output
    proposals_after_first = proposals_by_asset(config)

    # A second run fetches zero assets yet re-anchors everything: it
    # fetches faces again and reproduces the same proposals.
    second = run_batch(config)
    assert second.exit_code == 0, second.output
    assert "assets fetched=0" in second.output
    assert window_line(second) == window_line(first)
    assert proposals_by_asset(config) == proposals_after_first
    # 14 appearance-bearing assets, asked for faces on both runs.
    assert len(reid_stub.faces_requests()) == 30


def test_window_membership_follows_the_capture_time(
    reid_stub: StubImmich, make_config: Any
) -> None:
    # Two late park photos were captured the next day (EXIF); their upload
    # time stays on 2026-11-10, so only the capture time can window them.
    for asset in reid_stub.assets:
        if asset["id"] in (library.PHOTO_11_LATE_ID, library.PHOTO_13_LATE_ID):
            asset["exifInfo"] = {"dateTimeOriginal": "2026-11-11T12:00:00.000Z"}
    config = make_config(reid_stub, clustering=CLUSTERING)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    day_one = capture_day(datetime.fromisoformat("2026-11-10T08:00:00.000Z"))
    day_two = capture_day(datetime.fromisoformat("2026-11-11T12:00:00.000Z"))
    assert day_one is not None and day_two is not None and day_one != day_two
    lines = {
        line.split(":")[0]: line
        for line in result.output.splitlines()
        if line.startswith("window ")
    }
    assert len(lines) == 2, result.output
    # Window one: the 5-appearance indoor cluster and a 3-appearance park
    # cluster (both anchor-less -> queue), plus the 8 singletons.
    assert f"window {day_one.isoformat()}" in lines
    assert "clusters=2" in lines[f"window {day_one.isoformat()}"]
    assert "singletons=8" in lines[f"window {day_one.isoformat()}"]
    assert "ask_for_name=8" in lines[f"window {day_one.isoformat()}"]
    # Window two: only the two late park photos, clustered together.
    assert f"window {day_two.isoformat()}" in lines
    assert "clusters=1" in lines[f"window {day_two.isoformat()}"]
    assert "singletons=0" in lines[f"window {day_two.isoformat()}"]
    assert "ask_for_name=2" in lines[f"window {day_two.isoformat()}"]

    rows = proposals_by_asset(config)
    assert rows[(library.PHOTO_11_LATE_ID, 0)].window == day_two.isoformat()
    assert rows[(library.PHOTO_13_LATE_ID, 0)].window == day_two.isoformat()
    assert rows[(library.PHOTO_8_PARK_ID, 0)].window == day_one.isoformat()
    assert rows[(library.PHOTO_4_INDOOR_ID, 0)].window == day_one.isoformat()


def test_capture_time_prefers_exif_then_falls_back_to_upload(
    reid_stub: StubImmich, make_config: Any
) -> None:
    for asset in reid_stub.assets:
        if asset["id"] == library.PHOTO_11_LATE_ID:
            asset["exifInfo"] = {"dateTimeOriginal": "2026-11-09T10:00:00.000Z"}
        if asset["id"] == library.PHOTO_13_LATE_ID:
            del asset["fileCreatedAt"]  # upload-time fallback

    config = make_config(reid_stub, clustering=CLUSTERING)
    result = run_batch(config)
    assert result.exit_code == 0, result.output

    by_asset = store_of(config).appearances_by_asset()
    assert by_asset[library.PHOTO_11_LATE_ID][0].captured_at == datetime.fromisoformat(
        "2026-11-09T10:00:00.000Z"
    )
    assert by_asset[library.PHOTO_13_LATE_ID][0].captured_at == datetime.fromisoformat(
        library.PHOTO_13_CREATED_AT
    )
    assert by_asset[library.PHOTO_1_STANDING_ID][
        0
    ].captured_at == datetime.fromisoformat(library.PHOTO_1_CREATED_AT)


def test_window_days_merge_consecutive_capture_days_into_one_window(
    reid_stub: StubImmich, make_config: Any
) -> None:
    # The two late park photos were captured the next day; with
    # window_days = 2 both capture days form ONE window, so the park
    # cluster spans 2026-11-10 and 2026-11-11 again.
    for asset in reid_stub.assets:
        if asset["id"] in (library.PHOTO_11_LATE_ID, library.PHOTO_13_LATE_ID):
            asset["exifInfo"] = {"dateTimeOriginal": "2026-11-11T12:00:00.000Z"}
    config = make_config(
        reid_stub, clustering={"distance_threshold": 0.01, "window_days": 2}
    )

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    day_one = capture_day(datetime.fromisoformat("2026-11-10T08:00:00.000Z"))
    assert day_one is not None
    assert window_line(result) == (
        f"window {day_one.isoformat()}: appearances=18 clusters=2 anchored=0.0% "
        "conflicted=0.0% anchor_less=100.0% singletons=8 proposals=0 ask_for_name=10"
    ), result.output
    rows = proposals_by_asset(config)
    assert rows[(library.PHOTO_11_LATE_ID, 0)].window == day_one.isoformat()
    assert rows[(library.PHOTO_13_LATE_ID, 0)].window == day_one.isoformat()


def test_a_wipe_clears_proposals_with_everything_else(
    reid_stub: StubImmich, make_config: Any
) -> None:
    config = make_config(reid_stub, clustering=CLUSTERING)
    assert run_batch(config).exit_code == 0
    assert proposals_by_asset(config), "the run must have surfaced clusters"

    from .test_pipeline import wipe

    wiped = wipe(config)
    assert wiped.exit_code == 0, wiped.output
    assert store_of(config).proposals() == []


def test_a_run_without_any_appearance_fetches_no_faces(
    reid_stub: StubImmich, make_config: Any
) -> None:
    config = make_config(reid_stub, detection={"confidence_floor": 1.0})

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    assert "window " not in result.output
    assert reid_stub.faces_requests() == []
