# Immich ML stack pinning

Part of: [Person Re-ID v1 — Map](../map.md)
Type: research
Status: resolved

## Question

`goal.md` demands our container use the same PyTorch version as Immich itself. Find:

1. The exact PyTorch (or ONNX runtime) version pinned by Immich's machine-learning service, and why (check `machine-learning/pyproject.toml` in immich-app/immich and release notes).
2. The Python version the ML service runs.
3. Whether Immich publishes working ARM64 machine-learning images, and on what runtime they actually execute inference on ARM (ONNX? PyTorch?).
4. Whether "same PyTorch version as Immich" is even the right constraint if Immich's ML service itself runs ONNX — surface the fact so the spec can decide.

## Answer

Immich's ML service contains **no PyTorch at all** — every model runs through ONNX Runtime (`onnxruntime` `>=1.23.2,<2`, locked to 1.31.0 at v3.4.1), with per-device extras (CUDA/OpenVINO/ArmNN/RKNN/ROCm) all ONNX-Runtime-based or ONNX-session-compatible. Shipped images run Python 3.13. The default `immich-machine-learning` image is natively built for linux/amd64 + linux/arm64; on ARM64 it runs the ONNX Runtime CPUExecutionProvider (ArmNN and RKNN are opt-in arm64-only variants). "Same PyTorch version as Immich" is therefore a vacuous constraint; the real compatibility constraints are the pinned `immich-model` git rev, the ONNX model artifacts, and preprocessing — a torchreid/OSNet container may pick any torch version it wants.

Findings: [research/02-immich-ml-stack-pinning.md](../research/02-immich-ml-stack-pinning.md)
