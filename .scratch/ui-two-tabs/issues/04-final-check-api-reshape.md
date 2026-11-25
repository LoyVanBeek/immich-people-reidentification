# Final check API reshape (person-centric) and bulk endpoints

Part of: [UI Two Tabs — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: 03

## Question

How does the write-review API reshape into the person-centric Final check, and which new
endpoints does the UI need? Settle:

1. Replace `/api/write-review`'s tab structure with a **person-centric payload**: one entry
   per person (name, counts) and, per person, all appearances assigned to them — ReID
   proposals, pins, pending writes (with a written/approved marker). Decide whether this is
   a new endpoint (`/api/final-check`) or a reshape of `/api/write-review`.
2. **Bulk action endpoints**: bulk ignore / unignore, bulk pin ("Pin as X for next run"),
   bulk not-this-person, scoped server-side to an explicit list of `(asset_id, appearance)`
   pairs (the UI sends the visible-set selection, never "everything").
3. **Retarget-write-now**: the Final check "Write as X" (per-item and bulk) changes the
   pending write's target person immediately — decide the endpoint and semantics: resolve X
   against existing Immich people, else create on write; update the write ledger so
   idempotency holds; never touch an existing Immich label.
4. **Write bar endpoints**: per-person and global "Write N selected to Immich" — decide
   whether `/api/write-pending` is reused or replaced.
5. Produce the **death list** of endpoints/tables the retirement ticket (05) removes.

## Answer

**Final check API contract** (grilled 2026-11-25; card fields per ticket 03):

1. **New endpoint `GET /api/final-check`** — person-centric payload:
   `persons: [{name, counts, items: [...]}]`, each item a ticket-03 evidence card plus
   `source` (`proposal` | `pin` | `conflict` | `written`) and `written` (from the write
   ledger). Persons are derived from proposals + pins + write ledger. Unnamed
   (ask-for-name, unpinned) clusters stay in the Proposals tab only. The old
   `/api/write-review` is left untouched here and dies in ticket 05.
2. **Bulk actions: separate endpoint per action** (not one `/api/bulk`):
   `POST /api/bulk-ignore`, `POST /api/bulk-unignore`, `POST /api/bulk-pin {items, name}`,
   `POST /api/bulk-reject {items, name}` — each takes an explicit
   `items: [{asset_id, appearance}...]` list (the UI's visible-set selection, never
   "everything"), validates every item is a stored appearance, applies, and returns
   per-item results. `pin`/`reject` require `name`. Used in Proposals; pin semantics =
   next pipeline run.
3. **Retarget-write-now** ("Write as X" in Final check, per-item and bulk): two effects —
   `POST /api/retarget {items, name}` sets the pin to X (persists; next run's proposal
   agrees), and the write itself carries explicit per-item names so the item is written as
   X in this action, not as its original proposal.
4. **Write bars** (per-person "Write N selected as “X” to Immich" + global write-all):
   both call `POST /api/write-faces {writes: [{asset_id, appearance, name}]}`. Per item,
   server-side: resolve person by name (existing → id; else create at write time with the
   write-scoped key), ledger/tombstone check (idempotent; tombstoned pairs never written),
   conflict check (a different person's overlapping face → skip + flag, never overwrite),
   lazy head box, `POST /faces`, journal the write ledger. `/api/write-pending` dies in 05.
5. **"Confirm proposal" in Proposals no longer writes**: it marks the cluster
   **confirmed** — a new review-decision state alongside dismissed — so the cluster leaves
   the open queue and its appearances surface in Final check for the actual write.

**Death list for ticket 05**: endpoints `/api/write-review`, `/api/dry-run-batches`,
`/api/promote/<id>`, `/api/promote-all`, `/api/write-pending`, `/api/confirm-all`; the
dry-run mode of `/api/confirm`; tables `dry_run_batch`, `dry_run_action`,
`write_decisions` (include checkbox dropped per ticket 02 verdict); UI: dry-run toggle +
PREVIEW/LIVE banners, dry-run batch list, promote/promote-all buttons, the Pending writes
tab.
