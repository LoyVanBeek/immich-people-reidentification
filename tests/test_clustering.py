"""Unit tests for the pure clustering and anchoring rules (ticket 11).

The fixture photos cannot economically pin down threshold edge cases, so
these tests use synthetic embedding vectors against the pure functions
directly: unanimity, conflicts, singleton handling, ask-for-name queue
sizing, and threshold edges.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
from reid.anchoring import AnchorConfig, build_window, claim_face, face_person_overlap
from reid.clustering import (
    average_linkage_clusters,
    capture_day,
    cosine_distance_matrix,
    partition_windows,
)
from reid.enums import ProposalStatus
from reid.immich import Face
from reid.store import Appearance


def vectors_3d(gram: np.ndarray) -> np.ndarray:
    """Unit vectors whose pairwise dot products equal the given Gram matrix."""
    return np.linalg.cholesky(gram)


def test_identical_embeddings_have_zero_cosine_distance() -> None:
    matrix = cosine_distance_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert matrix[0, 1] == 0.0


def test_opposite_embeddings_have_cosine_distance_two() -> None:
    matrix = cosine_distance_matrix(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert matrix[0, 1] == 2.0


def test_embeddings_within_the_threshold_cluster_and_across_it_do_not() -> None:
    # Pairwise cosine distance 0.3: one cluster at threshold 0.5, two at 0.25.
    vectors = vectors_3d(np.array([[1.0, 0.7, 0.7], [0.7, 1.0, 0.7], [0.7, 0.7, 1.0]]))
    together = average_linkage_clusters(cosine_distance_matrix(vectors), 0.5)
    apart = average_linkage_clusters(cosine_distance_matrix(vectors), 0.25)
    assert together == [(0, 1, 2)]
    assert apart == [(0,), (1,), (2,)]


def test_average_linkage_never_chains_through_a_middle_point() -> None:
    # A-B and B-C sit at distance 0.4, but A-C sits at 0.8. Single linkage
    # would chain all three through B; average linkage must refuse the
    # {A,B}+C merge (mean 0.6) at threshold 0.5.
    vectors = vectors_3d(np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.6], [0.2, 0.6, 1.0]]))
    clusters = average_linkage_clusters(cosine_distance_matrix(vectors), 0.5)
    assert sorted(clusters) == [(0, 1), (2,)]

    # Raising the threshold past the mean 0.6 merges everything.
    everything = average_linkage_clusters(cosine_distance_matrix(vectors), 0.65)
    assert everything == [(0, 1, 2)]


def test_distance_threshold_zero_clusters_only_identical_embeddings() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    clusters = average_linkage_clusters(cosine_distance_matrix(vectors), 0.0)
    assert sorted(clusters) == [(0, 1), (2,)]


def test_a_lone_appearance_is_its_own_cluster() -> None:
    vectors = np.array([[1.0, 0.0]])
    clusters = average_linkage_clusters(cosine_distance_matrix(vectors), 0.5)
    assert clusters == [(0,)]


OFFSET_EAST = timezone(timedelta(hours=2))  # e.g. Europe/Berlin in summer


def test_capture_day_converts_the_instant_into_the_given_timezone() -> None:
    # 23:00 UTC is already the next day two timezones east.
    assert capture_day(
        datetime.fromisoformat("2026-11-10T23:00:00.000Z"), OFFSET_EAST
    ) == date(2026, 11, 11)
    assert capture_day(datetime.fromisoformat("2026-11-10T23:00:00.000Z"), UTC) == date(
        2026, 11, 10
    )


def test_capture_day_treats_a_naive_timestamp_as_local_wall_time() -> None:
    assert capture_day(
        datetime.fromisoformat("2026-11-10T23:00:00"), OFFSET_EAST
    ) == date(2026, 11, 10)


def test_capture_day_rejects_unparseable_timestamps() -> None:
    # Now we use datetime.min to indicate unusable (empty string from DB)
    assert capture_day(datetime.min) is None


def test_one_day_per_window_by_default() -> None:
    days = [date(2026, 11, 12), date(2026, 11, 10), date(2026, 11, 11)]
    windows = partition_windows(days, 1)
    assert windows == {
        date(2026, 11, 10): "2026-11-10",
        date(2026, 11, 11): "2026-11-11",
        date(2026, 11, 12): "2026-11-12",
    }


def test_multi_day_windows_fill_consecutive_calendar_days_greedily() -> None:
    days = [date(2026, 11, 10), date(2026, 11, 11), date(2026, 11, 12)]
    windows = partition_windows(days, 2)
    assert windows == {
        date(2026, 11, 10): "2026-11-10",
        date(2026, 11, 11): "2026-11-10",
        date(2026, 11, 12): "2026-11-12",
    }


def test_a_gap_in_the_days_starts_a_new_window() -> None:
    days = [date(2026, 11, 10), date(2026, 11, 13)]
    windows = partition_windows(days, 2)
    assert windows == {
        date(2026, 11, 10): "2026-11-10",
        date(2026, 11, 13): "2026-11-13",
    }


# -- anchoring ---------------------------------------------------------------


def appearance(left: float, top: float, right: float, bottom: float) -> Appearance:
    return Appearance(
        asset_id="asset-1",
        appearance=0,
        embedding=b"",
        confidence=0.9,
        box_left=left,
        box_top=top,
        box_right=right,
        box_bottom=bottom,
        pixel_source="preview",
        captured_at=datetime.min,
    )


def face(
    left: float,
    top: float,
    right: float,
    bottom: float,
    person: str | None = "p-1",
    name: str | None = "Albert",
) -> Face:
    return Face(
        asset_id="asset-1",
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        person_id=person,
        person_name=name,
    )


def test_a_face_fully_inside_the_person_box_claims_it_completely() -> None:
    ratio = face_person_overlap(
        appearance(100, 100, 900, 900), face(400, 200, 600, 400)
    )
    assert ratio == 1.0


def test_the_overlap_ratio_is_intersection_over_face_area() -> None:
    # Face area 40000, half of it outside the person box horizontally.
    ratio = face_person_overlap(
        appearance(100, 100, 500, 500), face(400, 200, 600, 400)
    )
    assert ratio == 0.5


def test_faces_below_the_containment_minimum_never_claim() -> None:
    appearance_row = appearance(100, 100, 500, 500)
    overlapping = face(400, 200, 600, 400)  # ratio 0.5
    barely_outside = face(450, 200, 650, 400)  # ratio 0.25
    assert claim_face(appearance_row, [barely_outside], 0.5) is None
    assert claim_face(appearance_row, [overlapping], 0.5) is overlapping


def test_non_overlapping_faces_are_ignored() -> None:
    appearance_row = appearance(100, 100, 500, 500)
    assert claim_face(appearance_row, [face(600, 200, 700, 300)], 0.5) is None


def test_the_best_qualifying_face_wins_the_claim() -> None:
    appearance_row = appearance(100, 100, 500, 500)
    weaker = face(400, 200, 600, 400)  # ratio 0.5
    stronger = face(200, 200, 400, 400)  # ratio 1.0
    claimed = claim_face(appearance_row, [weaker, stronger], 0.5)
    assert claimed is stronger


def test_a_zero_area_face_box_never_claims() -> None:
    appearance_row = appearance(100, 100, 500, 500)
    assert claim_face(appearance_row, [face(200, 200, 200, 200)], 0.5) is None


# -- cluster proposals --------------------------------------------------------

Cfg = AnchorConfig(face_overlap=0.5, min_cluster_size=2)


def asset_appearance(
    asset_id: str,
    index: int = 0,
    box: tuple[float, float, float, float] = (100.0, 100.0, 500.0, 500.0),
) -> Appearance:
    return Appearance(
        asset_id=asset_id,
        appearance=index,
        embedding=b"",
        confidence=0.9,
        box_left=box[0],
        box_top=box[1],
        box_right=box[2],
        box_bottom=box[3],
        pixel_source="preview",
        captured_at=datetime.fromisoformat("2026-11-10T08:00:00.000Z"),
    )


def anchored_faces(asset_id: str, name: str | None = "Albert") -> list[Face]:
    """One face at (200,200,300,300): fully inside the default person box.

    name=None is an unassigned detection (person: null); "" is an assigned
    but still unnamed person.
    """
    return [
        Face(
            asset_id=asset_id,
            left=200.0,
            top=200.0,
            right=300.0,
            bottom=300.0,
            person_id=None if name is None else f"person-{asset_id}",
            person_name=name or None,
        )
    ]


def window_of(
    members: list[Appearance],
    faces: Callable[[], dict[str, list[Face]]],
    clusters: list[tuple[int, ...]] | None = None,
    cfg: AnchorConfig = Cfg,
):
    if clusters is None:
        clusters = [tuple(range(len(members)))]
    return build_window("2026-11-10", members, faces(), clusters, cfg)


def status_of(outcome, asset_id: str, index: int = 0) -> ProposalStatus:
    for proposal in outcome.proposals:
        if proposal.asset_id == asset_id and proposal.appearance == index:
            return proposal.status
    return ProposalStatus.LABELED  # won't be used for absent


def test_unanimous_named_anchors_propose_for_every_unlabeled_member() -> None:
    members = [
        asset_appearance("asset-1"),  # carries the named anchor face
        asset_appearance("asset-2"),
        asset_appearance("asset-3"),
    ]
    outcome = window_of(
        members, lambda: {"asset-1": anchored_faces("asset-1", "Albert")}
    )

    assert status_of(outcome, "asset-1") == ProposalStatus.LABELED
    for asset_id in ("asset-2", "asset-3"):
        row = next(p for p in outcome.proposals if p.asset_id == asset_id)
        assert row.status == ProposalStatus.PROPOSED
        assert row.proposed_name == "Albert"
    assert outcome.summary.proposals == 2
    assert outcome.summary.anchored == 1


def test_an_anchor_carries_one_name_regardless_of_its_count() -> None:
    members = [asset_appearance(f"asset-{i}") for i in range(1, 4)]
    faces = {
        "asset-1": anchored_faces("asset-1", "Albert"),
        "asset-2": anchored_faces("asset-2", "Albert"),
    }
    outcome = window_of(members, lambda: faces)
    assert outcome.summary.anchored == 1
    assert outcome.summary.conflicted == 0
    for proposal in outcome.proposals:
        if proposal.status == ProposalStatus.PROPOSED:
            assert proposal.proposed_name == "Albert"


def test_conflicting_anchor_names_block_the_whole_cluster() -> None:
    members = [asset_appearance(f"asset-{i}") for i in range(1, 4)]
    faces = {
        "asset-1": anchored_faces("asset-1", "Albert"),
        "asset-2": anchored_faces("asset-2", "Bea"),
    }
    outcome = window_of(members, lambda: faces)

    assert outcome.summary.conflicted == 1
    assert outcome.summary.anchored == 0
    for proposal in outcome.proposals:
        assert proposal.status in (
            ProposalStatus.LABELED,
            ProposalStatus.CONFLICTED,
        ), proposal
        assert proposal.proposed_name is None
    assert status_of(outcome, "asset-3") == ProposalStatus.CONFLICTED


def test_an_unnamed_immich_face_labels_but_never_anchors() -> None:
    members = [asset_appearance(f"asset-{i}") for i in range(1, 4)]
    faces = {
        "asset-1": anchored_faces("asset-1", "Albert"),
        "asset-2": anchored_faces("asset-2", ""),  # assigned person, no name yet
    }
    outcome = window_of(members, lambda: faces)

    # One distinct name: unanimous, not conflicted.
    assert outcome.summary.anchored == 1
    assert outcome.summary.conflicted == 0
    # The unnamed member is already labeled: no proposal for it.
    assert status_of(outcome, "asset-2") == ProposalStatus.LABELED
    proposed = [
        p.asset_id for p in outcome.proposals if p.status == ProposalStatus.PROPOSED
    ]
    assert proposed == ["asset-3"]


def test_unassigned_faces_are_not_labels_and_not_anchors() -> None:
    members = [asset_appearance("asset-1"), asset_appearance("asset-2")]
    outcome = window_of(
        members,
        lambda: {"asset-1": anchored_faces("asset-1", None)},
    )
    # No named anchor, two appearances -> ask-for-name queue.
    assert status_of(outcome, "asset-1") == ProposalStatus.ASK_NAME
    assert status_of(outcome, "asset-2") == ProposalStatus.ASK_NAME


def test_anchor_less_clusters_of_minimum_size_enter_the_ask_for_name_queue() -> None:
    members = [asset_appearance("asset-1"), asset_appearance("asset-2")]
    outcome = window_of(members, dict)
    assert [p.status for p in outcome.proposals] == [
        ProposalStatus.ASK_NAME,
        ProposalStatus.ASK_NAME,
    ]
    assert outcome.summary.ask_for_name == 2
    assert outcome.summary.anchor_less == 1


def test_anchor_less_clusters_below_the_minimum_are_never_surfaced() -> None:
    members = [asset_appearance("asset-1")]
    outcome = window_of(members, dict)
    assert outcome.proposals == ()
    assert outcome.summary.singletons == 1
    assert outcome.summary.clusters == 0

    # A tunable minimum above the cluster size also stays unsurfaced,
    # while still counting as an anchor-less cluster in the summary.
    two_members = [asset_appearance("asset-1"), asset_appearance("asset-2")]
    strict = AnchorConfig(face_overlap=0.5, min_cluster_size=3)
    outcome = window_of(two_members, dict, cfg=strict)
    assert outcome.proposals == ()
    assert outcome.summary.clusters == 1
    assert outcome.summary.anchor_less == 1
    assert outcome.summary.ask_for_name == 0


def test_singletons_are_stored_but_never_surfaced() -> None:
    members = [asset_appearance(f"asset-{i}") for i in range(1, 3)]
    outcome = window_of(members, dict, clusters=[(0,), (1,)])
    assert outcome.proposals == ()
    assert outcome.summary.clusters == 0
    assert outcome.summary.singletons == 2
    assert outcome.summary.appearances == 2


def test_labeled_members_of_a_conflicted_cluster_stay_labeled() -> None:
    members = [asset_appearance(f"asset-{i}") for i in range(1, 3)]
    faces = {
        "asset-1": anchored_faces("asset-1", "Albert"),
        "asset-2": anchored_faces("asset-2", "Bea"),
    }
    outcome = window_of(members, lambda: faces)
    assert status_of(outcome, "asset-1") == "labeled"
    assert status_of(outcome, "asset-2") == "labeled"


def test_the_window_summary_counts_every_cluster_state() -> None:
    # Two multi-appearance clusters (one anchored, one anchor-less), one
    # conflicted cluster, and two singletons.
    members = [asset_appearance(f"asset-{i}") for i in range(1, 8)]
    clusters = [(0, 1), (2, 3), (4, 5), (6,)]
    faces = {
        "asset-1": anchored_faces("asset-1", "Albert"),
        "asset-5": anchored_faces("asset-5", "Albert"),
        "asset-6": anchored_faces("asset-6", "Bea"),
    }
    outcome = window_of(members, lambda: faces, clusters=clusters)

    summary = outcome.summary
    assert summary.window == "2026-11-10"
    assert summary.appearances == 7
    assert summary.clusters == 3
    assert summary.anchored == 1
    assert summary.anchor_less == 1
    assert summary.conflicted == 1
    assert summary.singletons == 1
    assert summary.proposals == 1  # asset-2 in the anchored cluster
    assert summary.ask_for_name == 2  # the anchor-less pair
