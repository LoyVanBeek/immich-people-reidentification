"""The per-asset pipeline: detect persons, gate, crop, embed, store.

Quality gates run in order, before embedding: a detector confidence below
the floor drops first, then the person-box height minimum. Edge-clipped
persons are never dropped. Every drop is logged with its reason, and drop
counts are tallied per gate per run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection, PersonDetector, decode_pixels, is_edge_clipped
from .embedder import PersonEmbedder

_log = logging.getLogger("reid.pipeline")


@dataclass(frozen=True)
class GateConfig:
    confidence_floor: float = 0.35
    min_box_height: float = 100.0


@dataclass
class GateCounts:
    low_confidence: int = 0
    below_min_height: int = 0


@dataclass
class GateOutcome:
    kept: list[Detection] = field(default_factory=list)
    counts: GateCounts = field(default_factory=GateCounts)
    edge_clipped_kept: int = 0


def apply_gates(detections: list[Detection], cfg: GateConfig) -> GateOutcome:
    """Run the ordered quality gates over one asset's detections.

    A detection that fails both gates is claimed by the first one only, so
    per-gate counts always add up to the drop total.
    """
    outcome = GateOutcome()
    for detection in detections:
        if detection.confidence < cfg.confidence_floor:
            outcome.counts.low_confidence += 1
            continue
        if detection.height < cfg.min_box_height:
            outcome.counts.below_min_height += 1
            continue
        outcome.kept.append(detection)
    return outcome


@dataclass
class AssetAppearance:
    """One kept person appearance, ready to embed and store."""

    appearance: int
    detection: Detection
    embedding: bytes


def crop_person(image: np.ndarray, detection: Detection) -> np.ndarray:
    """The full-person crop: source pixels clamped to the image bounds."""
    height, width = image.shape[:2]
    left = max(0, int(detection.box[0]))
    top = max(0, int(detection.box[1]))
    right = min(width, int(detection.box[2]))
    bottom = min(height, int(detection.box[3]))
    return image[top:bottom, left:right]


def process_asset(
    asset_id: str,
    payload: bytes,
    detector: PersonDetector,
    embedder: PersonEmbedder,
    cfg: GateConfig,
) -> tuple[list[AssetAppearance], GateCounts, int]:
    """Detect, gate, and embed every person appearance in one asset's pixels.

    Returns the appearances ready to store, the per-gate drop counts, and
    the number of edge-clipped persons kept. Every drop is logged with its
    reason at this level, so the run log explains why each person did not
    make it.
    """
    image = decode_pixels(payload)
    height, width = image.shape[:2]
    outcome = apply_gates(detector.detect(image), cfg)
    counts = outcome.counts
    for detection in outcome.kept:
        if is_edge_clipped(detection, width, height):
            outcome.edge_clipped_kept += 1
    if counts.low_confidence:
        _log.info(
            "asset %s: dropped %d person detection(s): detector confidence below the floor %.2f",
            asset_id,
            counts.low_confidence,
            cfg.confidence_floor,
        )
    if counts.below_min_height:
        _log.info(
            "asset %s: dropped %d person detection(s): person-box height below the minimum %.0fpx",
            asset_id,
            counts.below_min_height,
            cfg.min_box_height,
        )
    if outcome.edge_clipped_kept:
        _log.info(
            "asset %s: kept %d edge-clipped person appearance(s)",
            asset_id,
            outcome.edge_clipped_kept,
        )
    appearances: list[AssetAppearance] = []
    for index, detection in enumerate(outcome.kept):
        embedding = embedder.embed(crop_person(image, detection))
        appearances.append(AssetAppearance(index, detection, embedding.tobytes()))
    return appearances, counts, outcome.edge_clipped_kept
