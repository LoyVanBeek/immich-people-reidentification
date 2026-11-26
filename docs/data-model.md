# Data model: the local store

> **Diagram**: [`diagrams/data-model.puml`](diagrams/data-model.puml) — an ER
> diagram of every table. GitHub does not render PlantUML inline, so open the
> `.puml` file in PlantUML/Kroki (or render it locally) to view it.

Everything the pipeline knows that Immich does not is persisted in one SQLite
file — the **local store** (`[store] path` in `reid.toml`, default
`/data/reid.sqlite3`). All access goes through the `Store` class in
`src/reid/store.py`.

In production `reid-core` is the single long-lived owner of the store. The
pipeline itself runs as a subprocess of the core (the job manager starts
`reid-batch`, one run at a time), and `reid-batch` may also be run by hand —
but only while the core is stopped. `reid-ui` has no store volume at all; it
proxies every `/api/*` request to the core.

The store splits into three kinds of tables: **pipeline state** (rebuilt or
erased by runs and wipes), **review decisions** (the operator's voice; they
survive runs *and* wipes), and **ops/UI state**.

## Pipeline state

Erased by `reid-wipe`. See [pipeline.md](pipeline.md) for how each table is
written.

| table | key | contents |
| --- | --- | --- |
| `checkpoint` | `feed` | The **new-asset feed** checkpoint: the newest asset `createdAt` of the last fully completed run, stored verbatim and sent back as `createdAfter`. The only feed is `new-assets`. Advances only after a whole run succeeds, so a crashed run resumes instead of reprocessing the library. |
| `appearance` | `(asset_id, appearance)` | One **person appearance**: the 512-float32 L2-normalized embedding (BLOB), detector confidence, person box (`box_left/top/right/bottom`), `pixel_source` (`preview`/`original`), and `captured_at` (ISO; empty when the asset has no usable capture time). Replaced wholesale per asset, so reprocessing an asset is idempotent. |
| `proposal` | `(asset_id, appearance)` | The latest run's conclusion for one appearance: `window` (timespan-window key), `cluster` (index within the window), `status` (`ProposalStatus`), and `proposed_name` when there is one. Replaced wholesale every run — clusters are rebuilt from scratch, so this table always mirrors the most recent run. |
| `write_ledger` | `(asset_id, person_id)` | A confirmed write to Immich: cluster, window, and the wall-clock UTC timestamp. Drives idempotency (a pair is never written twice) and tombstone detection. |
| `tombstone` | `(asset_id, person_id)` | A label the user removed in Immich directly. Such pairs are never re-proposed or re-written. Detected each run by comparing ledger pairs against the asset's current Immich faces. |

## Review decisions

Written only through the confirmation UI (via the core). They survive batch
runs **and** `reid-wipe` — they are the operator's voice, not pipeline state.

| table | key | contents |
| --- | --- | --- |
| `appearance_rejection` | `(asset_id, appearance, proposed_name)` | "This appearance is not this person." Matching proposals are never created again and matching writes are skipped. |
| `appearance_pin` | `(asset_id, appearance)` | "This appearance is this person" (`person_name`). Anchoring always groups the appearance with the pinned person, whatever the embeddings say; the assignment still flows through the normal proposal path — a pin never writes to Immich directly. |
| `appearance_ignore` | `(asset_id, appearance)` | A wrong detection, with `reason`: `ignore` (detection noise) or `not_a_person`. Excluded from clustering entirely — never proposed, never ask-for-name, skipped by writes. |
| `cluster_dismissal` | `(window, cluster)` | "I don't care about this cluster." |
| `cluster_confirmation` | `(window, cluster)` | "This proposal is confirmed": the cluster leaves the open queue and its appearances surface in the Final check tab for the actual write. |

## Ops / UI state

| table | key | contents |
| --- | --- | --- |
| `run_status` | `id` | One row per pipeline run, owned by the core's job manager: `state` (`running`, `done`, `failed`, `cancelled`, `interrupted`), `trigger`, `pid`, `started_at`/`finished_at`, per-window progress (`windows_done`), and `error`. Runs left `running` by a crashed core are marked `interrupted` at startup. |
| `tuning` | `key` | Operator tuning overrides saved from the UI, applied on top of the TOML config on every run: `confidence_floor`, `min_box_height`, `window_days`, `distance_threshold`, `face_overlap`. |

## Enums

Status values are never raw strings in code — `src/reid/enums.py`:

- `ProposalStatus`: `proposed`, `ask_name`, `conflicted`, `labeled`, `rejected`
  (operator: this appearance is not this person).
- `ClusterOutcomeStatus`: `anchored`, `conflicted`, `anchor_less`, `singleton`.
- `ClusterRowStatus` (UI only): `open`, `confirmed`, `skipped`.
- `EvidenceCardStatus` (UI only): `will_write`, `already_labeled`,
  `conflict_skip`, `no_head_skip`, `ask_for_name`, `rejected`.
- `WriteResultStatus`: `skipped`, `written`, `error`.

## Lifecycle notes

- **Datetimes** are `datetime` objects in memory; they are parsed at the
  boundary (`fromisoformat`) and serialized at egress (`isoformat`, with `Z`
  for values sent to Immich).
- **Migrations** are additive `ALTER TABLE`s in `Store._migrate` (e.g.
  `appearance.captured_at`, `appearance_ignore.reason`); retired tables
  (`dry_run_batch`, `dry_run_action`, `write_decision`) are dropped, not
  migrated — their contents were ephemeral approvals, and the proposals they
  were built from survive in `proposal`.
- **Wipe** (`reid-wipe`) deletes `appearance`, `proposal`, `write_ledger`,
  `tombstone`, and `checkpoint`, so the next run re-fetches and re-embeds the
  whole library. Review decisions, tuning, and run history survive. A model
  or threshold change therefore means: wipe, then let the next run rebuild.
