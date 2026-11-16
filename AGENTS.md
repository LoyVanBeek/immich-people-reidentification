## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

Current feature: `.scratch/person-reid-v1/`
- Issue map: `.scratch/person-reid-v1/map.md`
- Spec: `.scratch/person-reid-v1/spec.md`
- Issues: `.scratch/person-reid-v1/issues/*.md`

### Triage labels

Default canonical triage label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.

---

## Navigation pointers

### Source layout
- `src/reid/` — main package
  - `batch.py` — batch run entry point (checkpointed feed, embed, cluster, anchor)
  - `cli.py` — `reid-batch`, `reid-wipe`, `reid-ui` Click commands
  - `server.py` — confirmation UI HTTP API (`/api/*`)
  - `store.py` — SQLite persistence (checkpoint, embeddings, proposals, write ledger, tombstones, dry-run batches)
  - `immich.py` — Immich API client (read + write scoped)
  - `detector.py` — YOLO11n person detection
  - `embedder.py` — OSNet-AIN x1.0 embeddings
  - `anchoring.py` — cluster anchoring with Immich face labels
  - `clustering.py` — cosine distance, average linkage, window partitioning
  - `pipeline.py` — detect → gate → embed per asset
  - `head_detector.py` — SCRFD face detection fallback
  - `config.py` — TOML config + env overrides
  - `enums.py` — `ProposalStatus`, `EvidenceCardStatus`, `DryRunActionStatus`, `WriteResultStatus`
  - `wipe.py` — store wipe CLI

### Tests
- `tests/` — pytest suite, stub Immich server in `tests/stub_immich.py`
- `tests/library.py` — fixture photo registry and people registry
- `tests/conftest.py` — shared fixtures (`ui_stub`, `config_with_store`, etc.)

### Key conventions
- **Datetimes**: `datetime` objects in dataclasses; parse at boundaries (`fromisoformat`), serialize at egress (`isoformat` with `Z`).
- **Status enums**: `ProposalStatus`, `EvidenceCardStatus`, `DryRunActionStatus`, `WriteResultStatus` — never raw strings.
- **Pre-commit**: `ruff` (lint/format), `mypy` (strict), `pytest` — see `.pre-commit-config.yaml`.
- **Config**: `reid.toml` with env override `IMMICH_API_KEY` — see `config.py`.

### Coding standards
See `CODING_STANDARDS.md` for full rules (datetimes, enums, no duplicate validation, data clumps, etc.).
