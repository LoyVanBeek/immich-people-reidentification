# Write-back semantics

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: [Immich API surface](01-immich-api-surface.md)

## Question

Exactly what happens on confirmation:

1. Which Immich API calls write the person label to which assets.
2. How dry-run records differ from real ones, and where dry-run results are kept for review before promoting to a real write.
3. Conflict handling: an asset already carries a different person label.
4. Idempotency: re-running nightly and re-confirming must not duplicate or corrupt state.

## Comments

- Grilled 2026-11-14, three rounds. All decisions accepted as recommended except Q2 (chose **(b)**: a real head detector, not a heuristic box) and Q11 (chose **(b)**: Immich's unassigned face detections are reused as a head-box source, not ignored). Detector facts in [findings/08-head-detector.md](../findings/08-head-detector.md).

## Answer

### 1. Write mechanism

Immich attaches names to assets through faces — there is no direct "label asset with person" call. On confirmation, the pipeline writes via **`POST /faces`**: one face entry per asset in the confirmed cluster, carrying a **head box** and bound to the target person. This reaches assets Immich never face-tagged, which is the pipeline's whole purpose; reassigning existing faces (`PUT /faces/{faceId}`) only re-treads what Immich already knows and is not the mechanism. If the target person doesn't exist yet, `POST /people` creates it first (see person resolution below).

### 2. Head boxes

- **Primary external source (preferred when available)**: `GET /faces?id=<assetId>` returns **all** face detections including unassigned ones (`person: null`) — this corrects an earlier research note that claimed it only returned recognized faces (verified against api.immich.app, the v3 migration blog, and `face.controller.ts`; the map's issue-01 line has been amended). Where an Immich face box is contained in our person box (intersection / face-box-area ≥ 0.5, tunable), its box is reused as the head box; best match wins if several qualify.
- **Detector for the rest**: **SCRFD-500M** (`det_500m.onnx`) at 320×320, run per person crop through the existing ONNX Runtime; weights vendored into the image (issue 03 pattern). Face box → head box by geometric expansion (extend top ~0.5–0.6× face height, widen ×1.6–2.0, square-ify — tunable constants, calibrated against Immich's own SCRFD-family boxes). Estimated ~50–120 crop/s; the head stage adds ~2–6 s to a 50-photo batch — negligible. License: insightface code MIT, weights non-commercial-research-only — acceptable for private v1; **YuNet** (MIT + Apache-2.0) is the drop-in clean-license fallback if ever shipped.
- **Neither available** (back-of-head, heavy occlusion, frame-edge crops): the write for that asset is **skipped** and the asset is listed in the confirmation UI as "unwritable — no head found", so a confirmation's coverage is visible before clicking.
- **Timing**: head detection is **lazy** — computed at confirm/promote time for the assets about to be written, not during the nightly pass. Most proposals never get confirmed; a per-asset head detection is well under a second.

### 3. Dry-run vs real

- **Dry-run performs zero Immich writes** (provably side-effect-free; no person creation either, so abandoned dry-runs can't pollute the People page). Proposals are persisted in the local store flagged `mode: dry_run`.
- **Promotion** replays the recorded actions in order: create the person if new, then `POST /faces` per asset. Real-run confirmations write through immediately. Both paths journal into the write ledger.

### 4. Conflicts

- **Different person, overlapping box**: an Immich face assigned to a different person whose box overlaps our person box (containment/overlap threshold tunable) → **skip the write and flag the asset in the UI**. Never overwrite Immich's existing labels in v1 — the ground-truth principle cuts both ways.
- Non-overlapping different-person faces are just group photos, not conflicts.
- Asset already carries the target person's face → skip as already-labeled (same check-before-write as idempotency).

### 5. Idempotency

- A local **write ledger** keyed by `(assetId, personId)` journals every confirmed write. The nightly rebuild excludes already-written pairs from new proposals (they still count as labeled evidence); re-confirming a cluster is a no-op against written pairs.
- Before each `POST /faces`, the asset's existing faces are fetched and checked, so a crash between the write and its journal entry can't produce duplicates on retry.
- **Manual removal in Immich** (user deletes the label there) is treated as a human override: a **tombstone** is recorded for the pair, and it is never re-proposed.
- **Write scope**: all cluster members; the clustering threshold is the only confidence gate (no second per-asset threshold).

### 6. Person resolution for anchor-less clusters

The name prompt offers autocomplete over **existing Immich people** plus free text: a typed name matching an existing person attaches the cluster to that person; anything else creates a new person (`POST /people`) at confirm/promote time. So a cluster of an already-known person whose face Immich never recognized attaches to the existing person. (The prompt's UI shape lives in ticket 06; this fixes the write-back contract.)
