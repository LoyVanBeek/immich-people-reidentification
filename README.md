# Immich Person Re-identification

A Raspberry Pi 5 container that detects persons in new Immich photos, embeds
them, clusters them within a timespan window, anchors clusters with Immich's
face labels, and proposes name assignments through a confirmation UI.

Status: early development — the batch container and checkpointed new-asset
feed are the first slice. See `.scratch/person-reid-v1/` for the spec and
tickets.

## Architecture

Three processes, one owner of the store:

| process | role |
| --- | --- |
| `reid-core` | Always-on API server. **The only process that touches the store** (`/data/reid.sqlite3`): reads for the UI and every write (confirmations, pins, ignores, write decisions). It also owns the job manager: it runs `reid-batch` as a subprocess, one run at a time, and records progress in the store's `run_status` table. |
| `reid-ui` | Serves the browser assets and proxies every `/api/*` request to the core, injecting the core token server-side. It has **no store volume** and holds no state. |
| `reid-trigger` | One-shot client used by the nightly systemd timer: asks the core to start a run (`--wait` follows it to completion). |

The core is reachable only through `X-API-Key` when `[core] api_key` (or
`REID_CORE_API_KEY`) is set — do set it on a shared network, because the core
can start pipeline runs. The browser talks only to the UI, which is still
unauthenticated (full user auth is future work), so keep the UI on your LAN.

Confirmations pause while a pipeline run is active: the core refuses them
(the proposals it would write are about to be rebuilt) and the UI disables the
buttons and shows the run's progress.

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
3. Run the core and the UI (the image is ARM64; build on the Pi 5 itself or
   with `docker buildx`):

   ```bash
   docker compose --profile local build
   docker compose --profile local up -d reid-core reid-ui
   ```

   `reid-core` listens on loopback `:8090` (only the UI and the trigger need
   it), `reid-ui` on `:8080` (open `http://<pi>:8080`). Both use host networking and run as your UID/GID so the
   bind-mounted store stays yours. Set `REID_DATA_DIR=/srv/reid-data` in `.env`
   if the store should live outside the repo, and export `REID_CORE_API_KEY`
   (or set `[core] api_key` in `reid.toml`) to require a token between the UI
   and the core.

   The nightly timer triggers a run on the core instead of running the
   pipeline itself:

   ```bash
   docker run --rm --network host --entrypoint reid-trigger \
     -v "$PWD/reid.toml:/etc/reid/reid.toml:ro" reid-batch:local \
     --config /etc/reid/reid.toml --trigger timer --wait
   ```

   `scripts/install.sh` (mode-aware, idempotent) installs and enables
   `reid-core.service`, `reid-ui.service` and the `reid-batch.timer` that
   runs the trigger nightly. "Run now" in the UI starts the same run on demand.

   ```bash
   # On the Pi: auto-detect compose vs systemd, bring services up
   ./scripts/install.sh --mode auto --up

   # Or explicitly:
   ./scripts/install.sh --mode systemd --up   # systemd units
   ./scripts/install.sh --mode compose --up   # docker compose
   ```

   Re-running `scripts/install.sh` is safe: it converges whatever is installed
   and is a no-op when nothing changed. In compose mode it stops/disables any
   colliding systemd core/UI units (and vice versa), so the two modes never
   fight over ports 8080/8090.

   The installer writes a `reid.toml` with a generated core API key and sets
   `[core] host = "0.0.0.0"` so the core is reachable from another machine.
   Edit `reid.toml` to add your Immich read/write API keys.

   `scripts/install-systemd.sh` is a thin wrapper for
   `scripts/install.sh --mode systemd`.

4. `reid-batch` remains the ops CLI for one run by hand (it needs the store
   volume and should not run while the core is up — stop the core first, or
   use "Run now"). To reset the store, run the wipe entry point with the same
   config:

   ```bash
   docker run --rm --entrypoint reid-wipe --user "$(id -u):$(id -g)" \
     -v "$PWD/reid.toml:/etc/reid/reid.toml:ro" \
     -v /srv/reid-data:/data \
     reid-batch --config /etc/reid/reid.toml
   ```
5. Sanity-check the image end-to-end with the stub Immich server used by the
   test suite: `scripts/container-smoke.sh`.

## Run the UI on another machine (laptop → Pi core)

The UI is a thin proxy: it holds no store and no Immich client, so it can run
anywhere and point at the core on the Pi. The core binds loopback by default,
so on the Pi set `[core] host = "0.0.0.0"` (and `[core] api_key`) in `reid.toml`,
then restart `reid-core`. The core reports its own Immich `base_url`, which on
the Pi is usually `localhost` and useless on a laptop — override it with
`[ui] immich_url` (or `--immich-url`) so "open in Immich" links resolve there.

On the laptop:

```bash
# minimal reid.toml: only [core] + [ui] are read
cat > reid.toml <<EOF
[core]
url = "http://<pi>:8090"
api_key = "<same key as on the Pi>"
host = "127.0.0.1"
port = 8090

[ui]
immich_url = "http://<pi>:2283"
EOF

uv run reid-ui --config reid.toml   # binds 0.0.0.0:8080
# or without any config file, purely via flags/env:
uv run reid-ui \
  --core-url http://<pi>:8090 \
  --core-key "<same key>" \
  --immich-url http://<pi>:2283 \
  --host 127.0.0.1
```

Open `http://<laptop>:8080`. The UI proxies `/api/*` to the Pi's core, injects
the API key server-side, and shows the same queue. `reid.toml` (which holds the
core key) is gitignored — never commit it.

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
