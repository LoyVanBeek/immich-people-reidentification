# Build the two-tab UI and the new API

Part of: [UI Two Tabs — Map](../map.md)
Type: task
Status: claimed
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
