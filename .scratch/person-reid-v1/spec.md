# Person Re-ID v1 — Spec

Part of: [Person Re-ID v1 — Map](map.md)
Status: ready-for-agent

## Problem Statement

Immich's face recognition fails whenever a person doesn't face the camera or the face is blurry — yet the rest of the person (clothes, build, pose) is often clearly identifiable within a short timespan. The user's photo library therefore stays under-labeled: photos of known people sit untagged even though Immich already knows those people from other photos. The user wants these unlabeled appearances found, clustered, and proposed for naming — with a human confirming every write-back, because face recognition remains the ground truth and must never be silently overridden.

## Solution

A Python (uv-packaged) Docker container running on the same Pi 5 as Immich. On a nightly manual/batch run it:

1. Fetches new photos not yet processed (checkpointed feed).
2. Detects persons in each photo, applies quality gates, and embeds each person appearance with OSNet-AIN x1.0 — storing embeddings incrementally, so only new photos are processed.
3. Rebuilds clusters from scratch each run within a configurable timespan window (default: same calendar day).
4. Anchors clusters with Immich's face labels (unanimous named anchors propose their name; disagreement blocks; anchor-less clusters land in an ask-for-name queue).
5. Surfaces proposals in a small 3-pane confirmation web UI, with dry-run mode from day one: nothing is written to Immich until the user confirms — or promotes a dry-run batch.

Writes go through Immich's face API as head boxes bound to people. Idempotency, conflicts, and manual removals in Immich are all handled without ever overwriting Immich's own labels.

## User Stories

1. As an operator, I want a single Docker container I can run on the Pi 5 alongside Immich, so that Re-ID runs locally on existing hardware.
2. As an operator, I want to trigger the nightly batch manually or on a schedule, so that I control when compute is spent.
3. As an operator, I want only new photos processed each run (incremental embedding store), so that nightly runs stay fast no matter how large the library grows.
4. As an operator, I want to wipe the stored embeddings, so that I can start clean after changing detector or embedding settings.
5. As an operator, I want clusters rebuilt from scratch every run, so that threshold changes apply to the whole window without stale cluster state.
6. As an operator, I want the timespan window configurable (default: same calendar day), so that clustering respects the short-timespan reliability limit of Re-ID.
7. As an operator, I want a run over one day's new photos to finish well within a 30-minute budget, so that it fits the nightly window on CPU-only hardware.
8. As a photo curator, I want persons detected in every new photo even when faces are absent, blurred, or turned away, so that photos Immich's face recognition missed get found.
9. As a photo curator, I want one embedding per person appearance (a photo with three people contributes three appearances), so that clustering operates on the right unit.
10. As a photo curator, I want tiny or low-confidence crops dropped before embedding, so that interpolation artifacts don't pollute clusters — with per-gate drop counts logged so I can see why.
11. As a photo curator, I want edge-clipped persons kept, so that useful clothes/pose signal isn't thrown away.
12. As a photo curator, I want singletons (appearances that joined no cluster) stored but never surfaced, so that detector noise never wastes review time.
13. As a photo curator, I want clusters anchored by Immich's own named face labels, so that proposed names come from ground truth, not guesses.
14. As a photo curator, I want a cluster's proposals blocked and surfaced as conflicted when its anchors disagree on the name, so that I'm never asked to rubber-stamp a coin flip.
15. As a photo curator, I want proposals only for appearances Immich has not already labeled, so that the pipeline doesn't duplicate Immich's work.
16. As a photo curator, I want anchor-less clusters of at least two appearances offered in an ask-for-name queue, so that known people whose faces Immich never recognized still get catchable.
17. As a photo curator, I want name assignment with autocomplete over existing Immich people, so that a cluster of a known person attaches to the existing person rather than creating a duplicate.
18. As a photo curator, I want a typed new name to create the person in Immich on write, so that onboarding a new family member is a one-step affair.
19. As a photo curator, I want per-appearance status chips (`will write` / `already labeled` / `conflict — skip` / `no head — skip`) in the evidence grid, so that I can see a confirmation's coverage before clicking.
20. As a photo curator, I want include/exclude checkboxes per asset over writable appearances, so that I can opt marginal appearances out of a write.
21. As a photo curator, I want head-box overlays on evidence crops, so that I can see exactly where the face will be placed in Immich.
22. As a photo curator, I want dry-run mode as a top-bar toggle with an unmistakable mode banner, so that I can explore proposals with provably zero Immich side effects.
23. As a photo curator, I want dry-run confirmations collected into batches with a per-batch Promote action, so that accepted proposals become real writes deliberately.
24. As a photo curator, I want assets already carrying a different person's overlapping face label skipped and flagged as conflicts, so that Immich's labels are never overwritten.
25. As a photo curator, I want re-confirming an already-written cluster to be a no-op, so that repeated runs and double-clicks can't duplicate writes.
26. As a photo curator, I want labels I deleted manually in Immich to become tombstones that are never re-proposed, so that my direct corrections in Immich are respected as overrides.
27. As an operator, I want a per-run window summary (cluster count, anchored %, conflicted %, anchor-less %, singleton count), so that I can watch whether the "every cluster has an anchor" assumption holds on real data.
28. As an operator, I want the five tuning knobs (timespan window, cluster distance threshold, detector confidence floor, min box height, face→person containment) in the UI's ⚙ dropdown, so that I can tune behavior without editing config files.
29. As an operator, I want everything else in a config file, so that the UI stays a review workspace rather than an admin panel.
30. As an operator, I want store wipes kept out of the confirmation UI (available at batch-run/config level), so that destructive actions can't happen mid-review.
31. As an operator, I want the pipeline to use a read-scoped Immich API key and only obtain a write-scoped key where writes happen, so that the least-privilege boundary is real.
32. As an operator, I want head detection to happen lazily at confirm/promote time, so that nightly runs aren't taxed and confirmation stays responsive.
33. As a photo curator, I want keyboard navigation (`j`/`k` move, `c` confirm, `s` skip), so that reviewing a queue of proposals is fast.
34. As an operator, I want a checkpoint on the new-asset feed, so that a crashed or skipped run doesn't reprocess the whole library.
35. As a photo curator, I want assets already carrying the target person's face skipped as already-labeled evidence, so that proposals stay minimal and writes stay idempotent.
36. As an operator, I want the run to process photos via Immich's preview thumbnails with originals as a fallback knob, so that throughput stays high without a coordinate-mapping burden.

## Implementation Decisions

**Platform and packaging.** Python 3.13 packaged with uv; Docker image built for ARM64 (Pi 5). All model weights (YOLO11n, OSNet-AIN x1.0 combineall, SCRFD-500M) are vendored into the image at build time; the container needs no network beyond the Immich API. OSNet runs via official aarch64 PyTorch wheels (per ADR 0001); SCRFD runs through ONNX Runtime.

**Immich client.** One client module wrapping the Immich API, used by both the pipeline and write-back:
- New assets: checkpointed feed via `POST /search/metadata` (`createdAfter`) or `POST /sync/stream`; the checkpoint persists between runs.
- Face detections: `GET /faces?id=<assetId>` — returns **all** detections including unassigned (`person: null`).
- Pixels: preview thumbnails (~1440px) for detection and cropping; originals as a fallback knob. No thumbnail↔original coordinate mapping in v1.
- Auth: `x-api-key` headers; read-scoped key for the pipeline, write-scoped key used only on confirm/promote (least privilege).

**Pipeline (detect → gate → embed → store).**
- Detector: YOLO11n, COCO-pretrained, person class only, 640×640 letterboxed input, run on every asset. Immich face boxes are anchoring material, never detection input.
- Quality gates, in order, before embedding: detector confidence below ~0.35 → drop; person-box height below ~100px in the source image → drop; edge-clipped persons → keep. Rejected crops are never embedded; drop counts are logged per gate per run. Thresholds are tuning knobs.
- Embeddings: OSNet-AIN x1.0 + combineall weights (ADR 0001), 256×128 input → 512-dim, L2-normalized. One embedding per person appearance.
- Incremental store: embeddings persisted keyed by asset and appearance; each run processes only new assets. A store wipe (batch-run/config level) erases embeddings; clusters are always rebuilt from scratch, so only embeddings need persisting.

**Clustering and anchoring (pure logic).**
- Cosine distance over normalized embeddings; agglomerative hierarchical clustering, average linkage, distance threshold (initial default 0.5 — to calibrate). Unfitted appearances remain singletons.
- A cluster is one person's appearance-group within one window — not a persistent identity; never merged across windows; two same-name clusters in one window are legitimate and both may propose.
- Window membership by EXIF capture time in the Pi's local timezone, upload-time fallback when EXIF is missing.
- Anchor claim: an Immich face detection claims an appearance when its face box overlaps the appearance's person box (intersection / face-box-area ≥ 0.5, tunable; best match wins if several qualify).
- Unanimous named anchors (any count, one name) → propose that name for every unlabeled appearance in the cluster. Any disagreement (≥ 2 distinct names) → block the cluster's proposals entirely; surface as conflicted.
- Named anchors only in v1; matching on unnamed person-IDs is a deferred knob.
- Anchor-less clusters with ≥ 2 appearances (tunable minimum, default 2) enter the ask-for-name queue.
- Every run emits a window summary: cluster count, anchored %, conflicted %, anchor-less %, singleton count.

**Proposals and stores.** A local store persists: the checkpoint, embeddings, proposals (with per-appearance write statuses), dry-run batches, the write ledger, and tombstones. Dry-run proposals are persisted flagged `mode: dry_run`; real confirmations write through immediately.

**Write-back and promotion.**
- Mechanism: `POST /faces` — one face entry per cluster asset carrying a head box bound to the target person. Reassigning existing faces is not the mechanism. Person resolution: an existing Immich person if the chosen name matches one; otherwise `POST /people` at confirm/promote time.
- Head boxes, resolved lazily at confirm/promote time for the assets about to be written: Immich's own unassigned face detections preferred where contained in our person box (same ≥ 0.5 containment rule); else SCRFD-500M at 320×320 on the person crop with geometric face→head expansion (constants tunable); if neither yields a head, that asset's write is skipped and flagged "no head found".
- Conflicts: an Immich face assigned to a different person whose box overlaps our person box → skip the write, flag the asset, never overwrite. Non-overlapping different-person faces are just group photos. An asset already carrying the target person's face is skipped as already-labeled.
- Idempotency: a write ledger keyed by `(assetId, personId)` journals every confirmed write; already-written pairs are excluded from new proposals (still counting as labeled evidence); re-confirming is a no-op. Before each `POST /faces`, the asset's existing faces are fetched and checked, so a crash between write and journal entry cannot produce duplicates on retry.
- Tombstones: manual removal of a label in Immich is recorded as a tombstone for the pair; never re-proposed.
- Write scope: all cluster members — the clustering threshold is the only confidence gate.
- Dry-run: zero Immich writes (no person creation either). Promotion replays the recorded actions in order: create person if new, then `POST /faces` per asset, journaling into the write ledger.

**Confirmation UI.** The 3-pane workspace from the prototype (branch `prototype/confirmation-ui`, commit 4d529a2, is the primary source for its behavior and layout): left sidebar queue of cluster rows; center evidence grid — one card per appearance with crop, head-box overlay, asset, date, confidence, Immich face label, and per-appearance status chip, with include/exclude checkboxes over writable appearances; right assignment pane with the name prompt (autocomplete over existing Immich people; new names create on write), stats, confirm/skip. Dry-run is a top-bar toggle with a full-width mode banner and a relabeled confirm button; dry-run batches are listed with per-batch Promote. The five tuning knobs live in a ⚙ top-bar dropdown; everything else is config-file. Keyboard: `j`/`k`, `c`, `s`. The UI talks to the container over a small HTTP API backing it.

## Testing Decisions

- Good tests assert external behavior only: the proposals a run produces, the exact Immich API calls made on confirm/promote, and user-visible UI/API states. No assertions on internal module structure, intermediate data shapes, or function call graphs.
- **Primary seam — the container boundary.** Tests drive the two real entry points (the batch run, and the HTTP API backing the confirmation UI) against a stub Immich HTTP server plus a small fixture photo library of real images with known people. The stub records requests, so tests assert on proposal output and Immich call sequences (person creation, `POST /faces` payloads, absence of writes in dry-run, replay order on promotion). This one seam covers detection → embedding → clustering → anchoring → proposal → write-back, including idempotency, conflicts, tombstones, and dry-run/promotion semantics.
- **Model weights are real in the primary seam.** Fixture photos go through the actual YOLO11n and OSNet-AIN x1.0 models. Same weights over the same images is deterministic; a handful of fixtures keeps runtime acceptable while keeping the tests honest about the full pipeline.
- **Lower seam — clustering/anchoring logic.** The pure clustering/anchoring rules are tested directly with synthetic embedding vectors, because fixture photos can't economically pin down threshold edge cases (unanimity, conflicts, singleton handling, ask-for-name queue sizing).
- Prior art: none — the repo is greenfield. The prototype (branch `prototype/confirmation-ui`) serves as the behavioral reference for UI tests, and the resolved tickets' worked rules serve as the reference for clustering tests.
- The 30-minute budget is checked as a coarse smoke expectation over the fixture set, not a precise performance test.

## Out of Scope

- Direct Immich server-side integration; the container talks to the API only (revisit after v1).
- Video assets: photos only.
- Multi-user support: single user.
- Automated on-upload triggering: manual/nightly batch only.
- Historical backfill of the existing library in v1 (feasible later at ~2–10 h for ~30k photos; returns as a fresh effort after v1).
- Matching on unnamed Immich person-IDs (deferred tuning knob).
- Skipping human confirmation based on confidence thresholds (goal.md's "later" idea).
- Embedding/model versioning scheme: a model change simply invalidates the embedding store (wipe + re-embed, per ADR 0001); any smarter scheme is fog.

## Further Notes

- Fog items intentionally left open for implementation: scheduler mechanics (cron/systemd timer vs an in-container loop — the spec only requires a batch-run entry point schedulable either way), and tuning knobs that surface during use.
- The "every cluster contains at least one anchor" assumption is a prediction, not a guarantee; it is validated live via the per-run window summary, with anchor-less % as the trigger to revisit unnamed-person-ID matching.
- SCRFD-500M weights are non-commercial-research-only licensed — acceptable for this private v1; YuNet (MIT + Apache-2.0) is the drop-in clean-license fallback if the container is ever shipped.
- Immich face crops are not served by the API; all cropping happens locally from fetched pixels using returned boxes.
- The map's "Not yet specified" list (backfill, scheduler, model versioning) should be re-triaged after v1 ships.
