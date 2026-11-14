# Person Re-ID v1 — Map

Labels: wayfinder:map

## Destination

A buildable spec for Re-ID v1: a Python (uv-packaged) Docker container on the Pi 5 that, on nightly manual/batch runs, detects persons, stores embeddings incrementally (wipeable via UI), rebuilds clusters from scratch each run within a configurable timespan window (default: same calendar day), anchors clusters via Immich face labels, and proposes name assignments — confirmed through a small web UI with dry-run mode before anything is written back to Immich. Ready for `/to-spec`.

## Notes

- `goal.md` at the repo root is the source idea; read it first.
- Single user. Immich runs on a Raspberry Pi 5 (CPU-only, ARM64) on the local network; the Re-ID container targets the same Pi.
- Python throughout; Astral's `uv` as package manager.
- Nightly manual/batch trigger. Embeddings are stored incrementally (new photos only) with UI buttons to erase the store; cluster assignments are rebuilt from scratch every run.
- The timespan window is a configurable parameter, default same calendar day.
- Immich face labels are ground truth, taken as-is in v1. Human confirmation is required before any write-back; dry-run mode exists from day one.
- The assumption "every cluster contains at least one Immich-recognized face" is a prediction to validate with real data; anchor-less clusters get an ask-for-name fallback in the UI.
- Skills per ticket type: grilling tickets run `grilling` + `domain-modeling`; research tickets run `research`; the prototype ticket runs `prototype`.

## Decisions so far

<!-- the index: one line per closed ticket, enough to judge relevance, then zoom the link for the detail the ticket holds -->

- [Immich API surface](issues/01-immich-api-surface.md): All four pipeline needs are served: new assets via `POST /search/metadata` (`createdAfter`) or the checkpointed `POST /sync/stream` feed; per-asset face detections with pixel boxes via `GET /faces?id=<assetId>` (the param is the asset ID despite misleading spec docs); face crops are NOT served — the pipeline must crop locally from originals using the returned boxes (validates the own-detector design); write-back fully supported (`PUT /people/{id}`, `PUT /people/{id}/reassign`, `PUT /faces/{faceId}`, `POST /faces` for manual face creation); API keys via `x-api-key` with fine-grained permission scoping (least-privilege read vs write keys possible).
- [Immich ML stack pinning](issues/02-immich-ml-stack-pinning.md): Immich's ML service contains no PyTorch at all — every model runs through ONNX Runtime (`onnxruntime>=1.23.2,<2`, Python 3.13, ARM64 images native with CPUExecutionProvider default). The goal.md constraint "same PyTorch version as Immich" is vacuous; the real compatibility constraints are the pinned `immich-model` git rev, the ONNX artifacts, and preprocessing.

- [Re-ID model on ARM CPU](issues/03-reid-model-arm-cpu.md): OSNet/torchreid is viable on Pi 5 CPU — official aarch64 PyTorch wheels (2.12–2.15) work under uv out of the box, but the PyPI `torchreid` package is a stale third-party sdist dragging in TensorFlow-nightly: vendor `torchreid/models/osnet.py` (or BoxMOT's Re-ID module, AGPL-3.0) with weights baked into the image. Throughput: nano-YOLO detector ~7–15 img/s measured, osnet_x0.25 embedding ~50–100+ img/s estimated → a 50-photo nightly batch in ~10–30 s. Full ~30k backfill: 2–10 h (embedding-only via reused Immich detections: under ~2 h). Best alternatives: OSNet-AIN x0.25/0.5, HACNN. All estimates carry confidence levels in the findings file.

## Not yet specified

<!-- see "Fog of war": in-scope fog you can't ticket yet; graduates as the frontier advances -->

- Manual backfill of the historical library: out of scope for v1 (see Out of scope); feasibility is now known — a full ~30k pass estimates at 2–10 h on the Pi 5, under ~2 h embedding-only via reused Immich detections. Returns as a fresh effort after v1.
- Scheduler mechanics: nightly run via cron/systemd timer vs a loop inside the container.
- Tuning knobs as they surface during use.
- Embedding/model versioning: what happens to stored embeddings when the embedding model or detector changes.

## Out of scope

<!-- work ruled beyond the destination; closed, never graduates -->

- Direct Immich integration (server-side); revisit after v1 works, possibly when open-sourcing.
- Video assets: photos only in v1.
- Multi-user support: single user.
- Automated on-upload triggering: manual/nightly batch only.
- Historical backfill in v1: existing photos are hand-labeled already; a manually triggered backfill is a later extension — feasible at 2–10 h for ~30k photos on the Pi 5 per [Re-ID model on ARM CPU](issues/03-reid-model-arm-cpu.md).
