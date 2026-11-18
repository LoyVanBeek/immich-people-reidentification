# Immich Person Re-identification

A Raspberry Pi 5 container that detects persons in new Immich photos, embeds
them, clusters them within a timespan window, anchors clusters with Immich's
face labels, and proposes name assignments through a confirmation UI.

Status: early development — the batch container and checkpointed new-asset
feed are the first slice. See `.scratch/person-reid-v1/` for the spec and
tickets.

## What works today

`reid-batch` runs one batch: it connects to Immich with a read-scoped API key,
fetches every photo created after the last checkpoint, detects persons in each
photo with a vendored YOLO11n model, filters detections through ordered
quality gates, embeds each kept person with OSNet-AIN (512-dim, L2-normalized),
and persists one appearance per detected person in the local store. After
embedding, the run rebuilds clusters from scratch within the timespan window,
anchors them with Immich's named face labels, and stores the resulting
proposals locally — nothing is written back to Immich yet. The run summary
reports how many appearances were embedded, how many detections each gate
dropped, and, per window: cluster count, anchored %, conflicted %,
anchor-less %, singleton count, proposals, and ask-for-name entries. A second
consecutive run processes zero assets but re-anchors everything from scratch;
a crashed run resumes from the checkpoint rather than reprocessing the
library.

Detection knobs (pixel source, confidence floor, minimum box height) live in
the `[detection]` config section; clustering and anchoring knobs (timespan
window, distance threshold, ask-for-name minimum, face containment) live in
`[clustering]`. After changing them — or whenever you want to rebuild the
store from scratch — run `reid-wipe` with the same config: it erases all
stored embeddings and proposals and resets the checkpoint so the next batch
reprocesses every asset.

## Operator setup

1. Create a **read-scoped** API key in Immich (Account Settings → API Keys):
   enable `asset.read` only. The pipeline reads asset metadata and pixels; it
   writes nothing to Immich in this slice.
2. Copy `config/reid.example.toml`, edit the Immich base URL and the
   `[detection]` settings, and either put the API key in the file or
   (recommended) export `IMMICH_API_KEY`.
3. Build and run (the image is ARM64; build on the Pi 5 itself or with
   `docker buildx`):

   ```bash
   docker buildx build --platform linux/arm64 -t reid-batch .
   docker run --rm --user "$(id -u):$(id -g)" \
     -v "$PWD/reid.toml:/etc/reid/reid.toml:ro" \
     -v /srv/reid-data:/data \
     reid-batch --config /etc/reid/reid.toml
   ```

   `--user` makes the container write the store in `/data` as the host user —
   needed for bind mounts. Schedule the same command nightly via cron or a
   systemd timer.
4. To reset the store, run the wipe entry point with the same config:

   ```bash
   docker run --rm --entrypoint reid-wipe --user "$(id -u):$(id -g)" \
     -v "$PWD/reid.toml:/etc/reid/reid.toml:ro" \
     -v /srv/reid-data:/data \
     reid-batch --config /etc/reid/reid.toml
   ```
5. Sanity-check the image end-to-end with the stub Immich server used by the
   test suite: `scripts/container-smoke.sh`.

## Layout

- `src/reid/` — the Python package (batch entry point, Immich client, local store)
- `tests/` — the container-boundary test seam: a stub Immich HTTP server that
  records requests, plus a small fixture photo library
- `config/reid.example.toml` — annotated config file; real settings live in a
  copy of this file
- `scripts/container-smoke.sh` — end-to-end smoke test of the built ARM64 image
- `Dockerfile` — multi-stage build; the image needs no network beyond the Immich API

## Development

```bash
uv sync            # Python 3.13 environment
uv run pytest      # container-boundary tests
uv run mypy src tests
```
# CI trigger
