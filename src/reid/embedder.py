"""The person embedder: OSNet-AIN x1.0 + combineall weights (ADR 0001).

One full-person crop in, one 512-dim L2-normalized embedding out. The
preprocessing mirrors torchreid's test transform: the crop is resized to
256x128 (height x width), converted to RGB, scaled to [0, 1], and normalized
with the ImageNet statistics the weights were trained with.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ._vendor.torchreid.osnet_ain import osnet_ain_x1_0

# The spec's embedding input: 256x128 input -> 512-dim, L2-normalized.
INPUT_HEIGHT = 256
INPUT_WIDTH = 128

# ImageNet statistics from torchreid's transformer.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class EmbedderError(Exception):
    """The embedder weights cannot be loaded."""


class PersonEmbedder:
    def __init__(self, weights_path: str) -> None:
        # Vendored torchreid factory is unannotated upstream code; the
        # return is treated as nn.Module via attribute use below.
        self._model: Any = osnet_ain_x1_0(num_classes=1, pretrained=False)  # type: ignore[no-untyped-call]
        self._load_weights(weights_path)
        self._model.eval()

    def _load_weights(self, weights_path: str) -> None:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise EmbedderError(f"unexpected checkpoint format in {weights_path}")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self._model.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop the DataParallel `module.` prefix.
            if key.startswith("module."):
                key = key[len("module.") :]
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
        if not compatible:
            raise EmbedderError(f"no weights matched the model in {weights_path}")
        self._model.load_state_dict(model_state | compatible, strict=True)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        tensor = self._prepare(crop_bgr)
        with torch.no_grad():
            features = self._model(tensor)
        embedding = features.squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            raise EmbedderError("the model produced a zero embedding")
        normalized = embedding / norm
        return np.asarray(normalized, dtype=np.float32)

    def _prepare(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop = cv2.resize(crop_bgr, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(MEAN, dtype=np.float32)
        std = np.array(STD, dtype=np.float32)
        normalized = (rgb - mean) / std
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)
