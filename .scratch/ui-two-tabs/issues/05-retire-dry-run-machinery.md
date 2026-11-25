# Retire the dry-run batch / promote machinery

Part of: [UI Two Tabs — Map](../map.md)
Type: task
Status: resolved
Blocked by: 04

## Question

With the dry-run toggle removed (ticket 01, decision 7), the dry-run batch/promote flow is
dead. Do the retirement: remove the "review before writing" toggle and PREVIEW/LIVE banners
from the UI, the `/api/dry-run-batches`, `/api/promote/<id>`, `/api/promote-all` endpoints
and their handlers in `core.py`, the dry-run mode of `/api/confirm` and `/api/confirm-all`,
and the `dry_run_batch` / `dry_run_action` tables. Decide what happens to existing rows on
upgrade (wipe vs migrate) and record it. Confirm nothing else depends on those endpoints
(grep the UI and tests). Final check (ticket 04's write endpoints) becomes the only write
path; development safety = no write-scoped API key configured.

## Answer

**Retirement plan — audited and settled 2026-11-25; code removal graduates as build
ticket 07, landing together with the new two-tab UI (ticket 06) so the app never sits
broken.**

**Dependency audit (what references the dying surface):**
- UI `src/reid/ui/app.js`: `dryRun` state + toggle (top bar, both views), PREVIEW/LIVE
  banners, the whole "Write review" view (`/api/write-review`, `/api/write-pending`,
  `/api/confirm-all`, `/api/write-decision` calls), dry-run batch list +
  `/api/dry-run-batches` + `/api/promote/<id>` + `/api/promote-all`, and the `dry_run`
  flag in `/api/confirm`.
- Core `src/reid/core.py`: GET handlers for `/api/dry-run-batches`, `/api/write-review`,
  `/api/crop` (kept; gains `full=1`); POST handlers for `/api/write-decision`,
  `/api/confirm-all`, `/api/write-pending`, `/api/promote/<id>`, `/api/promote-all`;
  methods `get_write_review`, `get_dry_run_batches`, `confirm_all`, `promote_batch`,
  `promote_all`, `write_pending`; `confirm_cluster`'s `dry_run` param; `_dry_run` state;
  `UIState.dry_run`.
- Store `src/reid/store.py`: `dry_run_batch`/`dry_run_action` tables + their methods
  (`add_dry_run_batch`, `get_dry_run_batch`, `get_dry_run_batches`, `mark_dry_run_*`,
  `close_dry_run_batch_if_complete`), `write_decisions`/`set_write_decision`, and the
  `_migrate` branches that ALTER the dry-run tables.
- Tests: `tests/test_store.py::test_dry_run_action_written_tracking`;
  `tests/test_server.py` — dry_run-in-state, confirm dry_run, write-review grouping,
  write-decision persistence, confirm-all, promote-all, write-pending tests.
- `src/reid/server.py` needs **no** change — it proxies `/api/*` generically.

**Death list (final)**: endpoints `/api/write-review`, `/api/dry-run-batches`,
`/api/promote/<id>`, `/api/promote-all`, `/api/write-pending`, `/api/confirm-all`,
`/api/write-decision`; the dry-run mode of `/api/confirm` (confirm becomes
mark-cluster-confirmed per ticket 04); tables `dry_run_batch`, `dry_run_action`,
`write_decisions` (+ their store methods and `_migrate` branches); UI: dry toggle,
PREVIEW/LIVE banners, review view, pending tab, dry-batch list, promote buttons,
write-decision calls. After retirement the only write path is `POST /api/write-faces`.

**Existing rows on upgrade**: auto-drop — `_migrate` runs `DROP TABLE IF EXISTS` on
`dry_run_batch`, `dry_run_action`, `write_decisions` on next start. Dry-run batches are
ephemeral approvals; the proposals they were built from survive in the `proposal` table,
so nothing is lost — the operator re-selects in the new Final check. No manual wipe.
