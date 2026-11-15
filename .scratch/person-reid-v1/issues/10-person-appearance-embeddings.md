# 10: Detect, gate, and embed person appearances into the incremental store

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** A batch run over new photos finds every person appearance — including people with no face, a blurred face, or turned away — and stores one embedding per appearance, keyed by asset and appearance. Drop reasons are visible in the run log. The operator can wipe the store and start clean.

**Blocked by:** [09: Batch container with checkpointed new-asset feed](09-batch-container-checkpointed-feed.md)

**Status:** resolved

- [x] Person detection runs on every new asset: YOLO11n person class only, on Immich preview thumbnails with originals as a fallback knob
- [x] A photo with three detected people contributes three person-appearances, each with its full-person crop
- [x] Quality gates run in order before embedding: detector confidence below the floor → dropped; person-box height below the minimum → dropped; edge-clipped persons kept
- [x] Drop counts are logged per gate per run
- [x] Each kept appearance is embedded with OSNet-AIN x1.0 (combineall weights, 256×128 input → 512-dim, L2-normalized) and persisted keyed by asset and appearance
- [x] A follow-up run processes only assets added since (incremental embedding store)
- [x] A store wipe is available at batch-run/config level and erases embeddings; it never appears in the confirmation UI
- [x] Model weights are vendored into the image; the container needs no network beyond the Immich API
- [x] A run over the fixture photo set completes well within the 30-minute budget (coarse smoke expectation)
- [x] Fixture photos go through the real models in tests — same weights over the same images is deterministic

## Comments

Implemented on `main` (2026-11-15). All ten acceptance criteria verified: 5
new pytest acceptance tests drive the real CLI entry point against the stub
Immich server plus the extended fixture library (13 tests total),
`mypy --strict` clean, and `scripts/container-smoke.sh` now proves
appearances + gate drops + wipe across three container runs.

**Detector runtime: the `ultralytics` package (AGPL-3.0), YOLO11n.**
The spec fixed the model (YOLO11n); the runtime choice is recorded in
[ADR 0002](../../../docs/adr/0002-ultralytics-runtime-for-yolo-detector.md).
`yolo11n.pt` is committed under `src/reid/weights/` (picked up by the wheel as
package data, so the Dockerfile needs no COPY) and runs with
`imgsz=640, classes=[0], conf=0.01` — our own ordered gates do the filtering,
the detector returns raw detections. If the project ever needs a permissive
distribution, the recorded migration path is exporting to ONNX +
`onnxruntime`.

**Embedder: vendored torchreid OSNet-AIN x1.0.** Only `osnet_ain.py` is
vendored (MIT; download machinery stripped) at
`src/reid/_vendor/torchreid/osnet_ain.py`. The combineall checkpoint
(`osnet_ain_x1_0_combineall.pth`, 17.3 MB) is a plain `module.`-prefixed
state_dict: the loader strips the prefix, filters shape-mismatched keys (the
unused classifier), and loads `strict=True`. Preprocessing is
resize 256×128 → BGR→RGB → /255 → ImageNet mean/std → CHW; output is
512-dim, L2-normalized bytes stored in the appearance row. The feature path
is deterministic (the classifier is random-init but unused); two runs over
the same pixels produce identical embedding bytes.

**Gates run in order** (`src/reid/pipeline.py`,
`GateConfig`/`apply_gates`): confidence floor 0.35 (default), then minimum
box height 100 px (default); edge-clipped boxes are kept and counted
(`edge_clipped_kept` in the summary). Drop counts surface in the CLI echo as
`gate drops: low_confidence=N below_min_height=M`. Knobs live in the new
`[detection]` config section (`pixel_source`, `confidence_floor`,
`min_box_height`); `pixel_source = "original"` switches pixel fetches to
`/api/assets/{id}/original` as the fallback knob.

**Wipe is batch-run level only:** the separate `reid-wipe` console script
erases all appearance embeddings and resets the checkpoint; it is not
reachable from the confirmation UI (which ticket 09 has not shipped yet
anyway). `scripts/container-smoke.sh` runs it mid-sequence and proves the
next run re-embeds the fixture set.

**Verified detection counts (probe + tests):** trio fixture → 3 appearances,
0 drops; standing fixture → 1 kept + 1 low-conf drop; the full detection
registry (9 photos) → 12 appearances, 18 low-confidence drops, 2
edge-clipped boxes kept; `min_box_height = 100000` flips all 12 to
below_min_height drops in the gate-ordering test.

**Known boundary.** The model itself runs with `conf=0.01`, so a raw
detection below 0.01 confidence never reaches the gates and is counted by
no gate; with a 0.35 floor these are detector noise, not lost appearances.
The gate tally therefore counts every drop above the model floor.
