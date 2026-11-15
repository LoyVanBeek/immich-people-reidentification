# 09: Batch container with checkpointed new-asset feed

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** An operator can build and run the Re-ID container on the Pi 5 alongside Immich, and trigger a batch run that fetches only photos not yet processed. A repeated or crashed run never reprocesses the whole library. No detection or embedding yet — this slice is the runnable container, the checkpointed asset feed, and the test seam everything else builds on.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Building produces an ARM64 Docker image (Python 3.13, uv-packaged) that runs on the Pi 5 and needs no network beyond the Immich API
- [ ] A batch run connects to Immich with a read-scoped API key and fetches new assets via the checkpointed feed
- [ ] The checkpoint persists between runs: a second consecutive run processes zero assets
- [ ] A crashed or skipped run resumes from the checkpoint rather than reprocessing
- [ ] Run settings live in a config file (UI tuning knobs arrive in later tickets)
- [ ] Tests drive the batch entry point against a stub Immich HTTP server that records requests, plus a small fixture photo library of real images with known people
