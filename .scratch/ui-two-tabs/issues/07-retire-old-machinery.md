# Retire the old write/review machinery (execute ticket 05's death list)

Part of: [UI Two Tabs — Map](../map.md)
Type: task
Status: claimed
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
