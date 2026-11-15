# 10: Detect, gate, and embed person appearances into the incremental store

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** A batch run over new photos finds every person appearance — including people with no face, a blurred face, or turned away — and stores one embedding per appearance, keyed by asset and appearance. Drop reasons are visible in the run log. The operator can wipe the store and start clean.

**Blocked by:** [09: Batch container with checkpointed new-asset feed](09-batch-container-checkpointed-feed.md)

**Status:** ready-for-agent

- [ ] Person detection runs on every new asset: YOLO11n person class only, on Immich preview thumbnails with originals as a fallback knob
- [ ] A photo with three detected people contributes three person-appearances, each with its full-person crop
- [ ] Quality gates run in order before embedding: detector confidence below the floor → dropped; person-box height below the minimum → dropped; edge-clipped persons kept
- [ ] Drop counts are logged per gate per run
- [ ] Each kept appearance is embedded with OSNet-AIN x1.0 (combineall weights, 256×128 input → 512-dim, L2-normalized) and persisted keyed by asset and appearance
- [ ] A follow-up run processes only assets added since (incremental embedding store)
- [ ] A store wipe is available at batch-run/config level and erases embeddings; it never appears in the confirmation UI
- [ ] Model weights are vendored into the image; the container needs no network beyond the Immich API
- [ ] A run over the fixture photo set completes well within the 30-minute budget (coarse smoke expectation)
- [ ] Fixture photos go through the real models in tests — same weights over the same images is deterministic
