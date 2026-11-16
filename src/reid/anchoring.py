"""Pure anchoring rules: face claims turn clusters into proposals.

An Immich face detection claims an appearance when its face box overlaps
the appearance's person box with intersection / face-box-area at least the
containment minimum (best match wins if several faces qualify). Rules per
cluster:

- Named anchors unanimous (any count, one name) -> propose that name for
  every unlabeled appearance in the cluster.
- Two or more distinct anchor names -> the whole cluster is blocked and
  surfaced as conflicted; v1 never guesses between conflicting truth.
- No named anchor and at least min_cluster_size appearances -> ask-for-name
  queue. Lone appearances are singletons: stored, never surfaced.
- Appearances Immich has already labeled (any assigned person, named or
  not) never receive proposals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .enums import ClusterOutcomeStatus, ProposalStatus
from .immich import Face
from .store import Appearance, Proposal


@dataclass(frozen=True)
class AnchorConfig:
    """Anchoring thresholds: spec tuning knobs."""

    face_overlap: float = 0.5
    min_cluster_size: int = 2


@dataclass(frozen=True)
class ClusterOutcome:
    """What anchoring concluded about one cluster."""

    cluster_id: int
    members: tuple[int, ...]  # indices into the window's appearance list
    anchor_names: tuple[str, ...]  # distinct named anchors, sorted
    status: ClusterOutcomeStatus


@dataclass(frozen=True)
class WindowSummary:
    """The window summary every run emits for one timespan window."""

    window: str
    appearances: int
    clusters: int  # clusters of two or more appearances
    anchored: int
    conflicted: int
    anchor_less: int
    singletons: int  # appearances that joined no cluster
    proposals: int
    ask_for_name: int


@dataclass(frozen=True)
class WindowOutcome:
    """The proposals and summary a run produced for one window."""

    window: str
    clusters: tuple[ClusterOutcome, ...]
    proposals: tuple[Proposal, ...]
    summary: WindowSummary


def face_person_overlap(appearance: Appearance, face: Face) -> float:
    """How much of the face box sits inside the appearance's box: intersection / face area."""
    person_box = appearance.box
    left = max(person_box[0], face.left)
    top = max(person_box[1], face.top)
    right = min(person_box[2], face.right)
    bottom = min(person_box[3], face.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, face.right - face.left) * max(0.0, face.bottom - face.top)
    return intersection / area if area > 0.0 else 0.0


def claim_face(
    appearance: Appearance, faces: Sequence[Face], min_overlap: float
) -> Face | None:
    """The best face whose overlap with the appearance's box reaches the minimum.

    Ties keep the earlier face in the list, so the claim is deterministic.
    """
    best: Face | None = None
    best_ratio = -1.0
    for face in faces:
        ratio = face_person_overlap(appearance, face)
        if ratio >= min_overlap and ratio > best_ratio:
            best, best_ratio = face, ratio
    return best


def build_window(
    window: str,
    appearances: Sequence[Appearance],
    faces_by_asset: Mapping[str, Sequence[Face]],
    clusters: Sequence[Sequence[int]],
    cfg: AnchorConfig,
) -> WindowOutcome:
    """Anchor every cluster in one window and turn the results into proposals.

    The clusters partition the window's appearances (indices into
    `appearances`). Every appearance of a surfaced cluster gets a proposal
    row; singletons and anchor-less clusters below the minimum get none.
    """
    claims: list[Face | None] = [
        claim_face(
            appearance, faces_by_asset.get(appearance.asset_id, ()), cfg.face_overlap
        )
        for appearance in appearances
    ]

    outcomes: list[ClusterOutcome] = []
    proposals: list[Proposal] = []
    for cluster_id, members in enumerate(clusters):
        anchor_names = sorted(
            {
                face.person_name
                for index in members
                if (face := claims[index]) is not None and face.person_name is not None
            }
        )
        if len(members) == 1:
            status = ClusterOutcomeStatus.SINGLETON
        elif len(anchor_names) >= 2:
            status = ClusterOutcomeStatus.CONFLICTED
        elif len(anchor_names) == 1:
            status = ClusterOutcomeStatus.ANCHORED
        else:
            status = ClusterOutcomeStatus.ANCHOR_LESS
        outcomes.append(
            ClusterOutcome(cluster_id, tuple(members), tuple(anchor_names), status)
        )

        if status == ClusterOutcomeStatus.SINGLETON:
            continue
        for index in members:
            appearance = appearances[index]
            face = claims[index]
            if face is not None and face.person_id is not None:
                proposals.append(
                    _row(appearance, window, cluster_id, ProposalStatus.LABELED, None)
                )
            elif status == ClusterOutcomeStatus.ANCHORED:
                proposals.append(
                    _row(
                        appearance,
                        window,
                        cluster_id,
                        ProposalStatus.PROPOSED,
                        anchor_names[0],
                    )
                )
            elif status == ClusterOutcomeStatus.CONFLICTED:
                proposals.append(
                    _row(
                        appearance, window, cluster_id, ProposalStatus.CONFLICTED, None
                    )
                )
            elif len(members) >= cfg.min_cluster_size:
                proposals.append(
                    _row(appearance, window, cluster_id, ProposalStatus.ASK_NAME, None)
                )
            # Anchor-less below the minimum: stored, never surfaced.

    surfaced = [outcome for outcome in outcomes if len(outcome.members) >= 2]
    summary = WindowSummary(
        window=window,
        appearances=len(appearances),
        clusters=len(surfaced),
        anchored=sum(
            1 for outcome in surfaced if outcome.status == ClusterOutcomeStatus.ANCHORED
        ),
        conflicted=sum(
            1
            for outcome in surfaced
            if outcome.status == ClusterOutcomeStatus.CONFLICTED
        ),
        anchor_less=sum(
            1
            for outcome in surfaced
            if outcome.status == ClusterOutcomeStatus.ANCHOR_LESS
        ),
        singletons=sum(
            1
            for outcome in outcomes
            if outcome.status == ClusterOutcomeStatus.SINGLETON
        ),
        proposals=sum(
            1 for proposal in proposals if proposal.status == ProposalStatus.PROPOSED
        ),
        ask_for_name=sum(
            1 for proposal in proposals if proposal.status == ProposalStatus.ASK_NAME
        ),
    )
    return WindowOutcome(window, tuple(outcomes), tuple(proposals), summary)


def _row(
    appearance: Appearance,
    window: str,
    cluster_id: int,
    status: ProposalStatus,
    name: str | None,
) -> Proposal:
    return Proposal(
        asset_id=appearance.asset_id,
        appearance=appearance.appearance,
        window=window,
        cluster=cluster_id,
        status=status,
        proposed_name=name,
    )
