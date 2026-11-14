# Immich API surface

Part of: [Person Re-ID v1 — Map](../map.md)
Type: research
Status: resolved

## Question

What does the Immich REST API expose that this pipeline needs:

1. Listing assets added since a given date (the nightly batch of new photos).
2. Per-asset person/face detections and their labels — the anchor ground truth. Can face crops or person detections be obtained via the API at all? The design assumes we run our own person detector, so pin down exactly what Immich does and doesn't provide.
3. Writing confirmed person labels back: naming a face / assigning a person to an asset.
4. Authentication: API key mechanics and scoping.

Primary sources: the official Immich API docs (api.immich.app / immich.app docs) and the immich-app/immich repository (controller definitions).

## Answer

All four needs are met by the current API (spec v3.4.0-rc.0, `immich-app/immich@main`), with caveats pinned in [research/01-immich-api-surface.md](../research/01-immich-api-surface.md):

1. **New-asset listing**: `POST /search/metadata` with `createdAfter` (= upload time), cursor-paginated, `size ≤ 1000`; the mobile-style `POST /sync/stream` (`AssetsV2`, checkpointed via `/sync/ack`) is the purpose-built incremental feed. Timeline endpoints exist but are spec-flagged `Internal`.
2. **Face ground truth**: `GET /faces?id=<assetId>` (the param is the *asset* id, despite the spec's "Face ID" description) returns per-asset faces with pixel bounding boxes + assigned person + source type; asset responses only carry a deduped person list, no boxes. **Face crops are not served over the API** (only each person's feature-face thumbnail) — crop locally from originals using the returned boxes. Running our own detector remains necessary.
3. **Write-back**: fully supported — rename via `PUT /people/{id}`, bulk face assignment via `PUT /people/{id}/reassign` or `PUT /faces/{faceId}`, and manual-face creation with a bounding box via `POST /faces`. No endpoint exists for writing unlabelled boxes or embeddings.
4. **Auth**: `x-api-key` header (or `?apiKey=` query); keys are scoped per-key with the fine-grained `Permission` enum (wildcard `all`), so a least-privilege read-only key and a separate write-back key are both possible.

