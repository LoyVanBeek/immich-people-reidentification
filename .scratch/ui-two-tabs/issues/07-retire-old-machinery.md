# Retire the old write/review machinery (execute ticket 05's death list)

Part of: [UI Two Tabs — Map](../map.md)
Type: task
Status: resolved
Blocked by: 06

## Question

Execute the death list recorded in [Retire the dry-run batch / promote machinery](05-retire-dry-run-machinery.md)
— lands together with ticket 06 so the app never sits broken:

- Core: remove GET/POST handlers for `/api/write-review`, `/api/dry-run-batches`,
  `/api/promote/<id>`, `/api/promote-all`, `/api/write-pending`, `/api/confirm-all`,
  `/api/write-decision`; methods `get_write_review`, `get_dry_run_batches`, `confirm_all`,
  `promote_batch`, `promote_all`, `write_pending`; `confirm_cluster`'s dry-run mode;
  `_dry_run` / `UIState.dry_run`.
- Store: drop `dry_run_batch`, `dry_run_action`, `write_decisions` tables + their methods;
  `_migrate` auto-drops the three tables on next start (DROP TABLE IF EXISTS) and drops
  its dry-run ALTER branches.
- UI: dry toggle, PREVIEW/LIVE banners, review view, pending tab, dry-batch list, promote
  buttons, write-decision calls, keyboard handler (j/k/c/s dropped per ticket 01).
- Tests: remove/replace the dry-run/write-review/confirm-all/promote/write-pending tests
  listed in ticket 05's audit.
- `src/reid/server.py` unchanged (generic `/api/*` proxy).

Done when: no references to the dead endpoints/tables remain (grep clean), pre-commit
passes, and the only write path is `POST /api/write-faces`.

## Answer

**Executed and committed as `53a1d69` on main (2026-11-25).** Grep-clean (the only
remaining mention is the intentional `DROP TABLE IF EXISTS` in `_migrate`), pre-commit
all green: ruff, ruff-format, mypy strict, pytest **150 passed** (26 dead tests removed),
plus the new jsdom two-tab UI smoke suite 5/5 (`tests/ui/app.test.mjs`, `npm test`).

Removed: endpoints `/api/write-review`, `/api/dry-run-batches`, `/api/promote/<id>`,
`/api/promote-all`, `/api/write-pending`, `/api/confirm-all`, `/api/write-decision`, and
the old write-path `/api/confirm`; core methods `get_write_review`, `confirm_all`,
`promote_all`, `promote_batch`, `write_pending`, `confirm_cluster`, `get_dry_run_batches`,
`set_write_decision`, the `_dry_run` state / `UIState.dry_run`, and the dry-run branches
of `_write_appearance(_checked)`; store tables `dry_run_batch`, `dry_run_action`,
`write_decision` (auto-dropped in `_migrate` per ticket 05) with their methods and the
`DryRunAction`/`DryRunBatch` dataclasses; `DryRunActionStatus` and
`WriteResultStatus.WOULD_WRITE` from enums; `AGENTS.md`/`CODING_STANDARDS.md` enum lists
updated. Tests referencing the dead API deleted; `tests/ui/app.test.mjs` rewritten for
the two-tab UI. The only write path is now `POST /api/write-faces`; development safety =
no write-scoped API key configured.
