"""Enum types for the reid package.

Using str, Enum allows string comparison compatibility during transition.
"""

from enum import Enum


class ProposalStatus(str, Enum):
    """Status of a proposal from the anchoring phase."""

    PROPOSED = "proposed"
    ASK_NAME = "ask_name"
    CONFLICTED = "conflicted"
    LABELED = "labeled"
    REJECTED = "rejected"  # operator: this appearance is not this person


class ClusterOutcomeStatus(str, Enum):
    """Outcome status of a cluster after anchoring."""

    ANCHORED = "anchored"
    CONFLICTED = "conflicted"
    ANCHOR_LESS = "anchor_less"
    SINGLETON = "singleton"


class ClusterRowStatus(str, Enum):
    """UI-only status for a cluster row in the review queue."""

    OPEN = "open"
    CONFIRMED = "confirmed"
    SKIPPED = "skipped"


class EvidenceCardStatus(str, Enum):
    """UI-only status for an evidence card."""

    WILL_WRITE = "will_write"
    ALREADY_LABELED = "already_labeled"
    CONFLICT_SKIP = "conflict_skip"
    NO_HEAD_SKIP = "no_head_skip"
    ASK_NAME = "ask_for_name"
    REJECTED = "rejected"  # operator: not part of this cluster


class WriteResultStatus(str, Enum):
    """Result status from a write-faces operation."""

    SKIPPED = "skipped"
    WRITTEN = "written"
    ERROR = "error"
