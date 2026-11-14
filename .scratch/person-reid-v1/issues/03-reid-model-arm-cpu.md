# Re-ID model on ARM CPU

Part of: [Person Re-ID v1 — Map](../map.md)
Type: research
Status: resolved

## Question

Can OSNet/torchreid run well on a Raspberry Pi 5 (ARM64, CPU-only)?

1. Installation path for torchreid + PyTorch on ARM64 Linux under `uv` — what actually works.
2. Expected throughput on a Pi 5 CPU: photos/second for embedding with the small OSNet variants (`osnet_x0.25` etc.), and separately for a nano-class person detector (e.g. YOLOv8n/YOLO11n class). The nightly batch is dozens of photos; the pipeline pass budget is minutes.
3. Credible alternative lightweight Re-ID models worth considering before committing.
4. A backfill-cost estimate: total time to embed a tens-of-thousands-photo library on the Pi 5, to inform the later manual-backfill decision.

## Answer

Short version: **yes, OSNet/torchreid runs fine on a Pi 5 CPU** — with caveats about *packaging*, not performance.

1. **Install:** `uv add torch torchvision` just works on aarch64 (official CPU `manylinux_2_28_aarch64` wheels exist for torch 2.12–2.15). But do **not** `pip install torchreid` from PyPI: the PyPI package is a stale third-party sdist (Oct 2022) that drags in `tb-nightly` (TensorFlow nightly) and dead pins. Upstream has no wheels/pyproject at all. Best path: vendor `torchreid/models/osnet.py` (+ `load_pretrained_weights`) and only depend on torch, or use BoxMOT's Re-ID module (AGPL-3.0 caveat). Bake weights into the image — torchreid downloads them at runtime via gdown/Google Drive, which is quota-fragile.
2. **Throughput:** measured Pi 5 anchors: nano YOLO (640px) ≈ 7 img/s ONNX, ~11–15 img/s NCNN, ~3.3 img/s PyTorch eager. FLOP-scaled estimates: osnet_x0_25 embedding ≈ 50–100+ img/s (PyTorch) and higher via ONNX — embeddings are nearly free; the detector dominates. Full pipeline ≈ 0.15–0.4 s/photo → a 50-photo nightly batch in ~10–30 s. Minutes budget: easily met.
3. **Alternatives:** OSNet-AIN x0.25/0.5 (better cross-domain generalization — our exact case), HACNN (0.5 GFLOPs, best accuracy/FLOP in the torchreid zoo), MobileNetV2-based Re-ID (fast but weaker). Avoid ViT/CLIP-ReID-class models (10–100× compute) and Ultralytics' cls-model pseudo-embeddings.
4. **Backfill:** ~30k photos ≈ **2–10 h** on Pi 5 depending on runtime/model; **~30–90 min** if we reuse Immich's existing detections and only embed crops. Overnight-scale, not a reason to change hardware.

Findings (full detail, sources, uncertainty marking): [research/03-reid-model-arm-cpu.md](../research/03-reid-model-arm-cpu.md)
