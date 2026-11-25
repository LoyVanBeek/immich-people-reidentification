# Prototype the two-tab UI

Part of: [UI Two Tabs — Map](../map.md)
Type: prototype
Status: resolved

## Question

Build a cheap, rough, concrete prototype of the redesigned UI on a throwaway branch
(`prototype/ui-two-tabs`), against mock/stub data (no Immich, no core needed), covering
exactly:

1. The top-level tab bar with the exact labels **"Proposals"** and **"Final check"**, plus
   the per-tab top-bar controls decided in ticket 01 (⚙ settings + show-already-labeled +
   Run now in Proposals; Run now in Final check; global banners).
2. The Proposals tab: days pane → clusters pane → evidence grid → actions pane, using the
   **new evidence card**: clickable crop (links to the Immich photo URL), expandable
   full-photo overlay with all sibling appearances outlined (current highlighted, others
   dimmed), the Immich / ReID / You labeled rows + date/confidence/URL text + status chip,
   per-card actions, and the selection checkbox (separate from include-in-write).
3. The selection-driven actions pane: cluster controls on top, "N selected" bulk bar
   (Ignore, Undo ignore, Assign to person X…, Not this person) scoped to the visible set.
4. The Final check tab: Person pane (one row per person with counts) → grid of all that
   person's appearances (proposals, pins, pending writes marked), default-off checkboxes,
   per-person "Write N selected to Immich" bar + global write-all, per-item and bulk
   "Write as X" reassign.

The prototype exists to react to the look and the interaction — especially the visual design
direction (light/dark, density) and whether the two-checkboxes-per-card pattern reads clearly.

## Comments

- 2026-11-25: Prototype built and committed to throwaway branch `prototype/ui-two-tabs`
  (commit `a2978ac`) as `prototype/ui-two-tabs.html` — single self-contained file, mock
  data, open directly in a browser (`git show prototype/ui-two-tabs:prototype/ui-two-tabs.html`).
  One structural variant (decisions already settled in ticket 01) in the app's existing dark
  theme. Covers: tab bar + per-tab top bar; Proposals with days→clusters→evidence→actions and
  the new card (clickable crop, full-photo overlay with sibling outlines, Immich/ReID/You
  rows, select + write checkboxes); selection bulk bar with semantics hint; Final check
  person pane → grid, default-off checkboxes, per-person + global write bars, "Write as…"
  reassign. Verdict pending — awaiting the user's reaction.

## Answer

**Verdict (2026-11-25): the decided structure wins, with two amendments.**

1. Two-tab structure (Proposals / Final check, panes as prototyped): **yes, as prototyped**.
2. **The per-card "write" (include-in-write) checkbox is dropped** — Proposals cards keep
   only the "select" checkbox for bulk actions; write-selection happens only in Final check
   (default-off checkboxes there). Amends ticket 01 decisions 3/5.
3. Theme: **keep the current dark theme and density**.
4. Full-photo overlay: **inline expand on the card** (not a modal).
5. Final check: per-person write button must **name the person** —
   "Write N selected as “Alice” to Immich" (plus the global write-all across people).

Primary source: branch `prototype/ui-two-tabs` — `a2978ac` (initial prototype) + `fbb9a92`
(verdict fix: drop write checkbox, person-named write button). File:
`prototype/ui-two-tabs.html`, mock data, open in a browser directly.
