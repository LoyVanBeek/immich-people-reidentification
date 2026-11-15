"""The person detector: YOLO11n, COCO person class only (ADR 0002).

Runs on the pixels of every new asset and returns every person detection,
however small or uncertain: quality gating happens later, so drop reasons
can be counted per gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class 0 is "person".
PERSON_CLASS = 0

# Detections below ultralytics' own floor are discarded by the model call;
# anything above this counts as a detection and reaches the quality gates.
MODEL_CONF = 0.01

# Letterboxed input size, per spec.
IMGSZ = 640

# A detection counts as edge-clipped when its box comes within this many
# pixels of the image border.
EDGE_TOLERANCE_PX = 2


class DetectorError(Exception):
    """The detector could not run on an asset's pixels."""


@dataclass(frozen=True)
class Detection:
    """One person detection in source-image pixel coordinates."""

    confidence: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


class PersonDetector:
    def __init__(self, weights_path: str) -> None:
        self._model = YOLO(weights_path)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Detect people, sorted top-to-bottom then left-to-right.

        The stable reading-order sort gives person-appearances a
        deterministic index within an asset, independent of the detector's
        confidence ordering.
        """
        result = self._model.predict(
            bgr,
            imgsz=IMGSZ,
            classes=[PERSON_CLASS],
            conf=MODEL_CONF,
            device="cpu",
            verbose=False,
        )
        first = result[0]
        detections: list[Detection] = []
        boxes = first.boxes
        assert boxes is not None
        for box, confidence in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            left, top, right, bottom = (float(value) for value in box)
            detections.append(Detection(float(confidence), left, top, right, bottom))
        detections.sort(key=lambda d: (d.top, d.left))
        return detections


def decode_pixels(payload: bytes) -> np.ndarray:
    """Decode an asset's image bytes into the BGR array the models expect."""
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DetectorError("pixels are not a decodable image")
    return image


def is_edge_clipped(detection: Detection, width: int, height: int) -> bool:
    """True when the person box touches the image border.

    Edge-clipped persons are kept — the spec forbids dropping them — the
    predicate exists so runs can report how many appearances were kept
    despite clipping.
    """
    return (
        detection.left <= EDGE_TOLERANCE_PX
        or detection.top <= EDGE_TOLERANCE_PX
        or detection.right >= width - EDGE_TOLERANCE_PX
        or detection.bottom >= height - EDGE_TOLERANCE_PX
    )
