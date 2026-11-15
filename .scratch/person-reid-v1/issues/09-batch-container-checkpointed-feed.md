# 09: Batch container with checkpointed new-asset feed

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** An operator can build and run the Re-ID container on the Pi 5 alongside Immich, and trigger a batch run that fetches only photos not yet processed. A repeated or crashed run never reprocesses the whole library. No detection or embedding yet — this slice is the runnable container, the checkpointed asset feed, and the test seam everything else builds on.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Building produces an ARM64 Docker image (Python 3.13, uv-packaged) that runs on the Pi 5 and needs no network beyond the Immich API
- [x] A batch run connects to Immich with a read-scoped API key and fetches new assets via the checkpointed feed
- [x] The checkpoint persists between runs: a second consecutive run processes zero assets
- [x] A crashed or skipped run resumes from the checkpoint rather than reprocessing
- [x] Run settings live in a config file (UI tuning knobs arrive in later tickets)
- [x] Tests drive the batch entry point against a stub Immich HTTP server that records requests, plus a small fixture photo library of real images with known people

## Comments

Implemented on `main` (2026-11-15). All six acceptance criteria verified: 8
pytest acceptance tests drive the real CLI entry point (`reid-batch --config`)
against a stub Immich HTTP server that records every request
(`tests/stub_immich.py`); `mypy --strict` clean; `scripts/container-smoke.sh`
builds the ARM64 image and proves the checkpointed feed end-to-end across two
container runs.

**Feed choice: `POST /api/search/metadata` (`createdAfter`), not `POST /sync/stream`.**
Ticket 01's map note allowed either. The sync/stream option is ruled out by
its credential model: Immich sync endpoints reject API keys outright
(measured against v3.x: `403 {"message": "Sync endpoints cannot be used with
API keys"}` — sync is for logged-in session-token clients), while the
checkpointed feed must use a read-scoped API key. Metadata search works with
`asset.read` keys via flat `page`/`size`/`createdAfter` fields across
Immich v2–v3 (`nextPage` pagination; the nested `filter` + `nextCursor`
shape is v3.2+ only).

**Checkpoint scheme.** The store (SQLite, `src/reid/store.py`) persists the
newest asset `createdAt` seen in a *fully completed* run; the next run
queries `createdAfter` = that value. The checkpoint advances only after every
page has been fetched, so a crashed run resumes from the last checkpoint
(at-least-once semantics; never reprocesses what already checkpointed). The
checkpoint compares server-side timestamps to each other, so host clock skew
is irrelevant. Boundary safety rests on Immich's `createdAfter` being a
strictly-greater filter, so the max asset of a completed run is never
re-served.

**Fixture photos.** The five fixture photos are AI-generated home-style
images of known people (standing with back to camera, blurred face, group of
two, plus two scenes), 1440px JPEG — license-clean and YOLO-detectable;
ticket 10 may extend the library if detection needs stricter poses.

**Known boundary.** The checkpoint filters strictly newer assets, so an
asset whose createdAt equals the final checkpoint of a completed run but
that Immich ingested after the run would never be fetched. On Immich's
monotonic single-node server clock, ingested-after implies createdAt
strictly greater than the checkpoint, so this is a theoretical boundary;
noted for completeness.
