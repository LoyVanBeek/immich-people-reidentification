"""The head/face detector: SCRFD-500M via ONNX Runtime.

Runs on person crops (letterboxed to the model's expected input size) to
find a face box, which is then expanded geometrically to a head box.
This is the fallback when Immich's own unassigned face detections don't
cover the person box.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class HeadDetectorConfig:
    """Configuration for the SCRFD head detector."""

    weights_path: str
    input_size: int = 320  # fallback letterbox size; the model's shape wins
    confidence_threshold: float = 0.5  # minimum face score to accept
    # Face box -> head box expansion constants (tunable)
    top_expansion_ratio: float = 0.55  # extend top by this * face_height
    width_expansion_ratio: float = 1.8  # widen face box by this factor
    # NMS
    nms_threshold: float = 0.4


class HeadDetectorError(Exception):
    """The head detector could not run."""


@dataclass(frozen=True)
class FaceDetection:
    """One face detection in the crop's pixel coordinates."""

    left: float
    top: float
    right: float
    bottom: float
    score: float


class HeadDetector:
    """SCRFD-500M face detector running on person crops via ONNX Runtime."""

    def __init__(self, cfg: HeadDetectorConfig) -> None:
        self._cfg = cfg
        self._session = self._load_session(cfg.weights_path)
        # The packaged model fixes its input size (e.g. 640x640); letterbox
        # to whatever the model actually expects instead of the configured
        # default, or inference fails with a dimension mismatch.
        self._input_size = self._effective_input_size()

    def _effective_input_size(self) -> int:
        """The input size the loaded model expects (config value as fallback)."""
        shape = self._session.get_inputs()[0].shape
        dims = [d for d in shape[2:4] if isinstance(d, int) and d > 0]
        if len(dims) == 2 and dims[0] == dims[1]:
            return dims[0]
        return self._cfg.input_size

    def _load_session(self, weights_path: str) -> ort.InferenceSession:
        providers = ["CPUExecutionProvider"]
        try:
            return ort.InferenceSession(weights_path, providers=providers)
        except Exception as exc:
            raise HeadDetectorError(
                f"failed to load SCRFD weights from {weights_path}: {exc}"
            ) from exc

    def detect(self, crop_bgr: np.ndarray) -> FaceDetection | None:
        """Detect the best face in a person crop.

        Returns the highest-scoring face detection above the confidence
        threshold, or None if no face is found.
        """
        # Preprocess: letterbox to the model's expected input size
        input_tensor, scale, pad_left, pad_top = self._preprocess(crop_bgr)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})

        # Parse outputs (SCRFD multi-scale: cls, bbox, kps for each stride)
        # Output order typically: [cls_8, bbox_8, kps_8, cls_16, bbox_16, kps_16, cls_32, bbox_32, kps_32]
        # We only need cls and bbox; kps are ignored for head box expansion.
        detections = self._postprocess(
            outputs, scale, pad_left, pad_top, crop_bgr.shape[:2]
        )

        if not detections:
            return None

        # Return the highest-scoring detection
        best = max(detections, key=lambda d: d.score)
        if best.score < self._cfg.confidence_threshold:
            return None
        return best

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize to the model's input size, return tensor and transform params."""
        height, width = image.shape[:2]
        scale = self._input_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Resize
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Pad to square
        padded = np.full(
            (self._input_size, self._input_size, 3), 114, dtype=np.uint8
        )  # 114 = typical SCRFD pad value
        pad_top = (self._input_size - new_height) // 2
        pad_left = (self._input_size - new_width) // 2
        padded[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = (
            resized
        )

        # BGR to RGB, normalize to [0, 1], HWC -> CHW, add batch dim
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        tensor = chw[np.newaxis, ...]  # (1, 3, 320, 320)

        return tensor, scale, pad_left, pad_top

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: float,
        pad_left: int,
        pad_top: int,
        original_shape: tuple[int, int],
    ) -> list[FaceDetection]:
        """Parse SCRFD multi-scale outputs into face detections in original crop coordinates."""
        # The insightface scrfd2onnx export groups outputs by kind:
        # [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32].
        # Each score: (1, num_anchors*fh*fw, 1); each bbox: (..., 4).
        # Box values are per-coordinate distances to the anchor center,
        # scaled by the stride — no exp, no half-width form.
        strides = [8, 16, 32]
        num_anchors = 2
        detections: list[FaceDetection] = []

        for i, stride in enumerate(strides):
            cls_map = outputs[i][0]  # (N, 1)
            bbox_map = outputs[len(strides) + i][0]  # (N, 4)
            if cls_map.shape[0] != bbox_map.shape[0]:
                continue

            fm_h = self._input_size // stride
            fm_w = self._input_size // stride
            anchor_centers = self._generate_anchors(fm_h, fm_w, stride, num_anchors)
            if anchor_centers.shape[0] != cls_map.shape[0]:
                continue  # export layout mismatch: skip rather than misplace

            for j in range(cls_map.shape[0]):
                score = float(cls_map[j, 0])
                if score < self._cfg.confidence_threshold:
                    continue

                cx, cy = anchor_centers[j]
                d_left, d_top, d_right, d_bottom = (
                    float(v) * stride for v in bbox_map[j]
                )
                x1 = cx - d_left
                y1 = cy - d_top
                x2 = cx + d_right
                y2 = cy + d_bottom

                # Map back to original crop coordinates (remove padding, scale back)
                x1 = (x1 - pad_left) / scale
                y1 = (y1 - pad_top) / scale
                x2 = (x2 - pad_left) / scale
                y2 = (y2 - pad_top) / scale

                # Clamp to crop bounds
                crop_h, crop_w = original_shape
                x1 = max(0.0, min(float(crop_w), x1))
                y1 = max(0.0, min(float(crop_h), y1))
                x2 = max(0.0, min(float(crop_w), x2))
                y2 = max(0.0, min(float(crop_h), y2))

                if x2 > x1 and y2 > y1:
                    detections.append(FaceDetection(x1, y1, x2, y2, score))

        # NMS
        return self._nms(detections)

    def _generate_anchors(
        self, fm_h: int, fm_w: int, stride: int, num_anchors: int
    ) -> np.ndarray:
        """Generate anchor centers for a feature map.

        SCRFD places num_anchors per grid cell at ((i + 0.5) * stride,
        (j + 0.5) * stride), tiled so the flat order matches the ONNX
        outputs (row-major, each cell's anchors consecutive).
        """
        ys = (np.arange(fm_h) + 0.5) * stride
        xs = (np.arange(fm_w) + 0.5) * stride
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (fm_h*fm_w, 2)
        return np.repeat(centers, num_anchors, axis=0).astype(np.float32)

    def _nms(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        """Non-maximum suppression on face detections."""
        if not detections:
            return []

        boxes = np.array(
            [[d.left, d.top, d.right, d.bottom] for d in detections], dtype=np.float32
        )
        scores = np.array([d.score for d in detections], dtype=np.float32)

        # Standard NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self._cfg.nms_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def expand_face_to_head(
        self, face: FaceDetection, crop_shape: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        """Expand a face box to a head box using geometric expansion.

        Expansion strategy (calibrated against Immich's SCRFD boxes):
        - Extend top upward by ~0.55 * face_height (covers forehead/hair)
        - Widen horizontally by ~1.8x (covers ears/side of head)
        - Square-ify the result (head boxes are roughly square)
        - Clamp to crop bounds
        """
        crop_h, crop_w = crop_shape
        face_h = face.bottom - face.top

        # Expand
        new_top = face.top - self._cfg.top_expansion_ratio * face_h
        new_bottom = face.bottom  # Don't extend downward (chin is already included)
        new_height = new_bottom - new_top

        # Widen to match new height * width_expansion_ratio (or keep aspect)
        target_width = new_height * self._cfg.width_expansion_ratio
        center_x = (face.left + face.right) * 0.5
        new_left = center_x - target_width * 0.5
        new_right = center_x + target_width * 0.5

        # Square-ify: make it roughly square by taking max of width/height
        final_w = new_right - new_left
        final_h = new_bottom - new_top
        max_dim = max(final_w, final_h)

        # Center the square
        cx = (new_left + new_right) * 0.5
        cy = (new_top + new_bottom) * 0.5
        half = max_dim * 0.5

        final_left = cx - half
        final_top = cy - half
        final_right = cx + half
        final_bottom = cy + half

        # Clamp to crop bounds
        final_left = max(0.0, min(float(crop_w), final_left))
        final_top = max(0.0, min(float(crop_h), final_top))
        final_right = max(0.0, min(float(crop_w), final_right))
        final_bottom = max(0.0, min(float(crop_h), final_bottom))

        # Ensure valid box
        if final_right <= final_left or final_bottom <= final_top:
            # Fallback: return original face box clamped
            return (
                max(0.0, min(float(crop_w), face.left)),
                max(0.0, min(float(crop_h), face.top)),
                max(0.0, min(float(crop_w), face.right)),
                max(0.0, min(float(crop_h), face.bottom)),
            )

        return (final_left, final_top, final_right, final_bottom)


def map_crop_box_to_full_image(
    crop_box: tuple[float, float, float, float],
    person_box: tuple[float, float, float, float],
    crop_shape: tuple[int, int],
    full_image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a box from person-crop coordinates to full-image coordinates.

    The crop is the person box clamped to image bounds. The crop_box is
    in crop pixel coordinates; we need to translate it back to full image.
    """
    crop_left = max(0, int(person_box[0]))
    crop_top = max(0, int(person_box[1]))

    return (
        crop_box[0] + crop_left,
        crop_box[1] + crop_top,
        crop_box[2] + crop_left,
        crop_box[3] + crop_top,
    )
