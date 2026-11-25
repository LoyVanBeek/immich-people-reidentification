# Coding Standards

This document captures the project's coding conventions and standards. These rules are enforced during code review (see `code-review` skill) and should be followed when writing new code.

---

## Type System

### Datetimes
- **Use `datetime` objects internally** — never pass ISO strings between modules. Parse at boundaries (API, DB, config) and serialize only at the last moment.
- **Parse at ingestion**: `datetime.fromisoformat(string)` when reading from Immich API, SQLite, or config.
- **Serialize at egress**: `dt.isoformat()` (with `Z` suffix for UTC) when writing JSON responses, CLI output, or DB.
- **Never store ISO strings in dataclasses** — the `Appearance.captured_at`, `WriteLedgerEntry.timestamp`, `DryRunBatch.timestamp` fields are `datetime`, not `str`.

### Enums over Strings
- **Status fields must use `str, Enum`** — e.g., `ProposalStatus`, `EvidenceCardStatus`, `WriteResultStatus`.
- **Never use raw strings for status** — `status: ProposalStatus` not `status: str`.
- **Enum values are the wire format** — `ProposalStatus.PROPOSED == "proposed"` works for JSON serialization.

### Type Annotations
- **All public functions must have type hints** — including return types.
- **Use `from __future__ import annotations`** in every module.
- **Prefer concrete types over `Any`** — if you must use `Any`, add a comment explaining why.

---

## Code Structure

### No Duplicate Validation Logic
- **Extract shared validation into helpers** — the `confirm_cluster` (dry-run) and `promote_batch` paths must not duplicate appearance lookup, write-ledger check, tombstone check, or head-box resolution.
- **Single source of truth** — if two code paths do the same checks, factor them into a shared function or class.

### Data Clumps → Dataclasses
- **Group related fields** — `(asset_id, appearance, head_box, status, message)` belongs in `DryRunAction`, not passed as 5 separate parameters.
- **API response shapes are dataclasses** — `EvidenceCard`, `ClusterRow`, `AssignmentPane`, `StatsPane` are dataclasses, not ad-hoc dicts.

### Primitive Obsession
- **No raw `dict` for domain objects** — use dataclasses with typed fields.
- **No `str` for datetime** — see "Datetimes" above.
- **No `str` for status** — see "Enums over Strings" above.

### Imports
- **Standard library first**, then third-party, then local (`reid.*`).
- **Group with blank lines** between sections.
- **`from __future__ import annotations`** at top of every file.

---

## Database

### Schema
- **Foreign keys with `ON DELETE CASCADE`** for child tables (`dry_run_action` → `dry_run_batch`).
- **`ORDER BY ROWID`** when insertion order matters (e.g., dry-run action replay).
- **JSON columns for structured data** — `head_box` stored as JSON array, parsed on read.

### Migrations
- **Additive only** — `_migrate()` adds columns, never drops.
- **Default values** for new columns so old rows remain valid.

---

## Testing

### Test Isolation
- **Each test creates its own store/config** — use `tmp_path` fixture.
- **Stub Immich server per test** — `ui_stub` fixture provides fresh instance.
- **No shared mutable state** between tests.

### Assertions
- **Check behavior, not implementation** — assert on API responses, DB state, stub request logs.
- **Format-agnostic datetime checks** — compare base time, not exact string format (`.000` vs no milliseconds).

---

## Error Handling

### Fail Fast, Informative Messages
- **Config validation** raises `ConfigError` with section/key context.
- **Immich API errors** wrapped in `ImmichError` with status code and message.
- **Store errors** wrapped in `StoreError` with path context.

---

## Tooling

### Pre-commit (enforced)
- **ruff** — lint + format (`ruff check`, `ruff format`)
- **mypy** — strict type checking (see `pyproject.toml` for overrides)
- **pytest** — full test suite

### Local Development
```bash
uv run pre-commit run --all-files  # lint + format + types
uv run pytest                      # tests
uv run mypy src/                   # type check only
```

---

## Code Review Checklist

When reviewing a PR, verify:

- [ ] No datetime strings in dataclasses — all `datetime` objects
- [ ] No raw string status fields — all `Enum` types
- [ ] No duplicate validation logic between dry-run and promotion paths
- [ ] No data clumps passed as separate parameters — use dataclasses
- [ ] All public functions have type hints
- [ ] Tests assert behavior, not implementation details
- [ ] Pre-commit passes (`ruff`, `mypy`, `pytest`)
