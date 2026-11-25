# Build the two-tab UI and the new API

Part of: [UI Two Tabs — Map](../map.md)
Type: task
Status: resolved
Blocked by: 02, 03, 04, 05

## Question

Implement the decided design in `src/reid/ui/` + `src/reid/core.py` + `src/reid/store.py`,
following the prototype on branch `prototype/ui-two-tabs` (`a2978ac`, `fbb9a92`) as the
primary visual/behavioral source:

1. **Two top-level tabs** "Proposals" / "Final check" (replaces the "Write review" button),
   dark theme, no keyboard shortcuts. Per-tab top bar: Proposals gets ⚙ settings,
   show-already-labeled, Run now; Final check gets Run now; banners global.
2. **Proposals tab**: days → clusters → evidence grid → actions pane. Evidence card:
   clickable crop linking to `${immich_url}/photos/<asset_id>`, inline expand to the full
   photo (`/api/crop?full=1&asset_id=…`, SVG overlay, self highlighted) with
   Immich/ReID/You labeled rows, date/confidence/URL text, status chip, per-card actions,
   and ONLY the "select" checkbox (no write checkbox — ticket 02 verdict). Actions pane:
   cluster controls on top (name prompt, Confirm proposal → marks cluster `confirmed`,
   Skip), "N selected" bulk bar below calling `/api/bulk-*`.
3. **Final check tab**: person pane → grid of all appearances per person
   (`GET /api/final-check`), items default OFF, per-person "Write N selected as “X” to
   Immich" + global write-all (both `POST /api/write-faces`), per-item and bulk
   "Write as…" (`POST /api/retarget` + name in the write payload).
4. **API**: ticket 03 payload fields on evidence cards (`siblings`, `sibling_count`,
   `rejected_name`, `dismissed`); `/api/crop?full=1`; ticket 04 endpoints
   `/api/final-check`, `/api/write-faces`, `/api/retarget`, `/api/bulk-ignore|unignore|pin|reject`;
   `/api/confirm` becomes mark-confirmed (new cluster state alongside dismissed).
5. Tests updated for all of the above (stub Immich already serves faces/people/faces-write).

Done when: pre-commit (ruff, mypy strict, pytest) passes and the UI matches the
prototype's structure and semantics.

## Answer

**Built and committed as `806879e` on main (2026-11-25).** All pre-commit hooks pass
(ruff, ruff-format, mypy strict, pytest: 176 passed = 156 existing + 20 new in
`tests/test_two_tabs.py`).

What landed:

- `src/reid/store.py`: `cluster_confirmation` table + `confirm_cluster` /
  `confirmed_clusters` (review decisions — survive `wipe()` like dismissals).
- `src/reid/core.py`:
  - `EvidenceCard` gains `siblings` (all stored appearances of the asset, pixel-space
    boxes), `sibling_count`, `rejected_name`, `dismissed` — populated in
    `_build_evidence_for_cluster` (ticket 03 contract).
  - `/api/crop?full=1&asset_id=…` serves the whole pixel-source photo at native
    resolution, JPEG q85, no outlines (UI overlays boxes via SVG viewBox = natural size).
  - `GET /api/final-check`: person-centric payload (`persons` from proposals + pins +
    ledger; items carry card fields + `source` + `written` + `name`; ask-for-name without
    a pin stays in Proposals).
  - `POST /api/write-faces {writes: [{asset_id, appearance, name}]}` — the only write
    path: per-item person resolution (create if new, write-scoped key), ignore/reject
    respected, ledger/tombstone/conflict checks, lazy head box, ledger journal. Errors
    when no write-scoped key is configured (dev safety).
  - `POST /api/retarget {items, name}` (pins, persists); `POST /api/bulk-ignore |
    /api/bulk-unignore | /api/bulk-pin | /api/bulk-reject` with server-side validation
    that every item is a stored appearance.
  - `POST /api/confirm-cluster {window, cluster, name?, confirmed}` — marks the cluster
    confirmed (leaves the open queue; `ClusterRowStatus.CONFIRMED`); a name given at
    confirm time pins the cluster's appearances (ask-for-name answer).
  - Cluster rows reflect confirmed/skipped status.
- `src/reid/ui/`: rewritten `app.js` — two tabs (Proposals / Final check), new evidence
  card (clickable crop, inline full-photo expand with sibling outlines, Immich/ReID/You
  rows, select-only checkbox, per-card actions), selection-driven bulk bar in the actions
  pane, Final check person pane → grid with default-off checkboxes, per-person "Write N
  selected as “X” to Immich" + global write-all, per-item and bulk "Write as…".
  Dry-run toggle, review view, promote, keyboard shortcuts removed from the UI. New CSS.
- `tests/test_two_tabs.py`: 20 tests covering all of the above.
- `.pre-commit-config.yaml`: mypy hook aligned with pyproject (dropped extra
  `--disable-error-code` flags and `types-click`, which made HEAD fail the hook).

Not done here (ticket 07): the old endpoints (`/api/write-review`, `/api/dry-run-batches`,
`/api/promote/*`, `/api/write-pending`, `/api/confirm-all`, `/api/write-decision`), the
old `/api/confirm` write path, and the `dry_run_*`/`write_decisions` tables.
