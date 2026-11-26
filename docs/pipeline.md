# How it works: from new photo to Immich label

> **Diagram**: [`diagrams/pipeline.puml`](diagrams/pipeline.puml) — a swimlane
> activity diagram of one run plus the write path (Immich / trigger-UI / core /
> batch / store / operator). GitHub does not render PlantUML inline, so open the
> `.puml` file in PlantUML/Kroki (or render it locally) to view it.

One pipeline run takes new Immich photos to stored **proposals**, and a human
in the confirmation UI turns confirmed proposals into real Immich writes.
Nothing is ever written to Immich without that human step. Terms are defined
in the root `GLOSSARY.md`.

## Processes

| process | role |
| --- | --- |
| `reid-core` | Always-on API server and the single owner of the local store (see [data-model.md](data-model.md)). Owns the job manager: runs the pipeline as a subprocess, one run at a time, recording progress in `run_status`. |
| `reid-ui` | Serves the browser assets and proxies `/api/*` to the core, injecting the core token server-side. Holds no state. |
| `reid-trigger` | One-shot client for the nightly systemd timer: asks the core to start a run (`--wait` follows it to completion). |
| `reid-batch` | One pipeline run by hand (stop the core first, or use "Run now" in the UI). |
| `reid-wipe` | Erases pipeline state and resets the checkpoint so the next run reprocesses everything. Review decisions survive. |

## Run lifecycle

A run starts either from the nightly timer (`reid-trigger` → `POST /api/jobs`)
or from "Run now" in the UI. The core's `JobManager` (`src/reid/core.py`)
refuses a second run while one is active and launches
`python -m reid.cli --config reid.toml` as a subprocess with `REID_RUN_ID` set,
so a pipeline crash can never take the core down. Each run gets a `run_status`
row (`running` → `done` / `failed` / `cancelled`); the batch updates
`windows_done` as it finishes each timespan window, and the UI shows that
progress. Rows left `running` by a crashed core are marked `interrupted` at
core startup.

While a run is active the core refuses confirmations and writes
(`_require_idle`): the proposals they would act on are about to be rebuilt.

## Phase 1 — the checkpointed new-asset feed

`run_batch` (`src/reid/batch.py`) reads the checkpoint from the store and
fetches the **new-asset feed**: `POST /api/search/metadata` with
`createdAfter` = checkpoint, photos only (videos are out of scope in v1), with
the read-scoped API key. The checkpoint advances only after the *whole* run
succeeds, to the newest asset `createdAt` seen — stored verbatim and sent back
as the next run's `createdAfter`. A crashed run therefore resumes instead of
reprocessing the library.

## Phase 2 — per asset: detect → gate → embed

For each fetched asset (`src/reid/pipeline.py`):

1. Fetch pixels — the preview thumbnail (`/api/assets/{id}/thumbnail?size=preview`)
   by default, or the original (`pixel_source` in `[detection]`).
2. Detect persons with a vendored YOLO11n model (`src/reid/detector.py`).
3. Run the ordered **quality gates** before embedding: the confidence floor
   (`confidence_floor`, default 0.35) drops weak detections first, then the
   minimum person-box height (`min_box_height`, default 100 px) drops
   barely-visible ones. Edge-clipped boxes are deliberately kept. A detection
   failing both gates is claimed by the first one only, so per-gate drop
   counts always add up; they are reported in the run summary.
4. Crop the **full-person crop** (box clamped to the image bounds) and embed it
   with OSNet-AIN x1.0 combineall (`src/reid/embedder.py`): 512-dim,
   L2-normalized, stored as float32 bytes.
5. Persist one `appearance` row per kept detection (`store.add_appearances`
   replaces the asset's rows wholesale, so reprocessing is idempotent).

## Phase 3 — cluster and anchor, over the whole store

After embedding, the run rebuilds proposals from **all** stored appearances,
not just the new ones — a changed threshold or a freshly named anchor
re-proposes the whole window (`src/reid/batch.py::_cluster_and_anchor`,
`src/reid/clustering.py`, `src/reid/anchoring.py`):

1. **Excludes** operator-ignored appearances (never proposed, never pollute
   clusters) and appearances with no usable capture time (stored, never
   surfaced, until a wipe rebuilds the store).
2. **Tombstone detection**: for every `write_ledger` pair, if that person is no
   longer among the asset's Immich faces, record a `tombstone` — never
   re-propose a label the user removed in Immich.
3. **Timespan windows**: each asset's capture day (EXIF time in the local
   timezone, falling back to upload time) is assigned to a window — a greedy
   block of at most `window_days` consecutive calendar days, keyed by its first
   day. Default `window_days = 1`: each calendar day is its own window.
   Clusters are rebuilt from scratch every run and never merged across
   windows.
4. **Clustering** per window: cosine distance over the L2-normalized
   embeddings, agglomerative average-linkage clustering cut at
   `distance_threshold` (default 0.5). Appearances that join no
   multi-appearance **cluster** remain **singletons** — stored, never surfaced.
5. **Anchoring** per cluster: an Immich face detection *claims* an appearance
   when its face box is at least `face_overlap` (default 0.5) contained in the
   person box (intersection / face-box area; best match wins, operator-rejected
   names excluded). Then:
   - one unanimous named **anchor** → `proposed` that name for every
     unlabeled member;
   - two or more distinct anchor names → the whole cluster is `conflicted`
     (v1 never guesses between conflicting truth);
   - no named anchor and size ≥ `min_cluster_size` (default 2) → `ask_name`;
   - an appearance Immich already labeled (any assigned person) → `labeled`,
     never re-proposed.
   - Operator **pins** override all of this for their appearance: it moves to
     the pinned person's group (or its own group), the pin name anchors it, and
     the proposal is the pinned name — through the normal proposal path.
6. Operator **rejections** drop matching `(asset_id, appearance, name)`
   proposals, then `store.replace_proposals` swaps in the run's conclusions
   wholesale.

Each window's summary (clusters, anchored / conflicted / anchor-less /
singleton counts, proposals, ask-for-name entries) lands in the run log and
the UI.

## Phase 4 — human review (no Immich writes)

The UI's Proposals tab shows the queue; the Final check tab groups evidence
cards by person. The operator can confirm or dismiss a cluster, reject an
appearance-name pair, pin an appearance to a person, ignore a detection
(`ignore` vs `not_a_person`), or retarget ("write as X", which pins). All of
these persist as **review decisions** in the store and shape the next run —
none of them writes to Immich.

## Phase 5 — the write path (the only Immich writes)

Final check's write bars call `POST /api/write-faces` on the core
(`src/reid/core.py::write_faces`). It requires the core to be idle and a
**write-scoped** API key (`[immich] write_api_key`) to be configured. Per item
(`asset_id`, `appearance`, `name`):

1. Skip if the appearance is ignored or the pair was rejected by the operator.
2. Resolve the person by name — find or create in Immich (write-scoped key).
3. Resolve the **head box**: prefer an Immich *unassigned* face detection
   contained in the person box (≥ `face_overlap`); otherwise fall back to the
   vendored SCRFD-500M head detector (`src/reid/head_detector.py`) on the
   person crop.
4. Checks (`_write_appearance_checked`): already in the **write ledger** →
   skip (idempotent); **tombstoned** → skip; an overlapping Immich face labeled
   with the same person → journal + skip; an overlapping face labeled with a
   *different* person → **conflict**, skip, never overwrite; no head found →
   skip.
5. `POST /api/faces` with the head box, then journal the pair in the
   `write_ledger` → `written`.

The next run's tombstone detection closes the loop: if the user later removes
that label in Immich, the pair becomes a tombstone and is never re-proposed.

## Configuration

`reid.toml` (see `config/reid.example.toml`): `[immich]` connection and keys,
`[store]` path, `[detection]` pixel source + the two quality gates,
`[clustering]` window size, distance threshold, ask-for-name minimum, face
containment, `[core]` / `[ui]` server settings. Operators can also override
the detection/clustering knobs per run from the UI; those overrides live in
the store's `tuning` table and are applied on top of the TOML on every run.

## Reset

`reid-wipe` erases pipeline state (appearances, proposals, write ledger,
tombstones, checkpoint) and keeps review decisions, so the next run
re-fetches and re-embeds the whole library and re-anchors from scratch. Run
it at batch/configuration boundaries — after changing detection settings or
the embedding model — never as part of a per-asset confirmation flow.
