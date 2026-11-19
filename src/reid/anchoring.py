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

Operator pins override all of the above for their appearance: a pin moves
the appearance to its person's group (or surfaces it as its own group
anchored to them), the pin name counts as the anchor, and the proposal is
the pinned name — through the normal proposal path, never a direct write.
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


def _drop_rejected(
    faces: Sequence[Face],
    appearance: Appearance,
    rejected_faces: frozenset[tuple[str, int, str]],
) -> Sequence[Face]:
    """Faces the operator flagged as wrong labels for this appearance."""
    if not rejected_faces:
        return faces
    return [
        face
        for face in faces
        if (appearance.asset_id, appearance.appearance, face.person_name)
        not in rejected_faces
    ]


def _apply_pins(
    clusters: Sequence[Sequence[int]],
    appearances: Sequence[Appearance],
    claims: Sequence[Face | None],
    pins: Mapping[tuple[str, int], str],
) -> tuple[list[list[int]], dict[int, str]]:
    """Move pinned appearances to their person's group, or their own.

    Returns the modified partition and, per appearance index, the pinned
    name. A pin removes the appearance from its current group and joins the
    group where someone's face claim carries the pinned name; when no such
    group exists in this window, the appearance gets a standalone group.
    Pins for appearances outside this window are ignored (they persist for
    future runs).
    """
    groups = [list(c) for c in clusters]
    if not pins:
        return groups, {}

    index_by_key = {(a.asset_id, a.appearance): i for i, a in enumerate(appearances)}
    pinned_names: dict[int, str] = {}
    for key, name in pins.items():
        index = index_by_key.get(key)
        if index is None:
            continue
        pinned_names[index] = name
        for group in groups:
            if index in group:
                group.remove(index)
                break
        target = next(
            (
                g
                for g in groups
                if any(
                    (face := claims[i]) is not None and face.person_name == name
                    for i in g
                )
            ),
            None,
        )
        if target is None:
            target = []
            groups.append(target)
        target.append(index)
    return [g for g in groups if g], pinned_names


def build_window(
    window: str,
    appearances: Sequence[Appearance],
    faces_by_asset: Mapping[str, Sequence[Face]],
    clusters: Sequence[Sequence[int]],
    cfg: AnchorConfig,
    rejected_faces: frozenset[tuple[str, int, str]] = frozenset(),
    pins: Mapping[tuple[str, int], str] | None = None,
) -> WindowOutcome:
    """Anchor every cluster in one window and turn the results into proposals.

    The clusters partition the window's appearances (indices into
    `appearances`). Every appearance of a surfaced cluster gets a proposal
    row; singletons and anchor-less clusters below the minimum get none.

    ``rejected_faces`` carries (asset_id, appearance, person_name) triples the
    operator flagged as wrong labels: those faces never anchor or mark an
    appearance as already-labeled.

    ``pins`` carries (asset_id, appearance) -> person_name operator
    assignments: the appearance is moved to its person's group (or its own
    group), the pinned name anchors it, and the proposal is the pinned name.
    """
    claims: list[Face | None] = [
        claim_face(
            appearance,
            _drop_rejected(
                faces_by_asset.get(appearance.asset_id, ()), appearance, rejected_faces
            ),
            cfg.face_overlap,
        )
        for appearance in appearances
    ]

    groups, pinned_names = _apply_pins(clusters, appearances, claims, pins or {})

    outcomes: list[ClusterOutcome] = []
    proposals: list[Proposal] = []
    for cluster_id, members in enumerate(groups):
        # The operator's pin outranks any face claim: a pinned member
        # contributes only its pinned name to the anchors.
        anchor_names = sorted(
            {
                face.person_name
                for index in members
                if index not in pinned_names
                and (face := claims[index]) is not None
                and face.person_name is not None
            }
            | {pinned_names[index] for index in members if index in pinned_names}
        )
        if len(members) == 1 and members[0] not in pinned_names:
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
            pin_name = pinned_names.get(index)
            if pin_name is not None:
                if face is not None and face.person_name == pin_name:
                    # Immich already agrees with the operator: nothing to write.
                    proposals.append(
                        _row(
                            appearance, window, cluster_id, ProposalStatus.LABELED, None
                        )
                    )
                else:
                    # Operator decision wins: propose the pinned name through
                    # the normal review path.
                    proposals.append(
                        _row(
                            appearance,
                            window,
                            cluster_id,
                            ProposalStatus.PROPOSED,
                            pin_name,
                        )
                    )
            elif face is not None and face.person_id is not None:
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

    surfaced = [
        outcome
        for outcome in outcomes
        if len(outcome.members) >= 2
        or any(index in pinned_names for index in outcome.members)
    ]
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
