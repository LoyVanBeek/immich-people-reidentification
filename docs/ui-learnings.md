# UI learnings — hints and decisions from the user

Every UI-related hint/decision the user gave, in the order they were settled.
Sources: conversation history, `.scratch/person-reid-v1/issues/06` (prototype verdict),
`.scratch/ui-two-tabs/issues/01` (two-tab grilling session, 2026-11-25), and post-redesign
fix sessions (2026-11-26).

## Pre-redesign foundations (person-reid-v1 era)

**Overall shape — prototype verdict (2026-11-15), ticket 06:** variant **B, the 3-pane
workspace** wins: left sidebar queue → center evidence grid → right assignment pane.
Variant A's config drawer folded into the top bar as a ⚙ dropdown. Prototype first on a
throwaway branch, then build.

- Evidence card = crop with head-box overlay + asset, date, confidence, per-appearance
  status chip (`will write` / `already labeled` / `conflict — skip` / `no head — skip`) +
  include/exclude checkbox.
- Name prompt inline in the right pane, autocomplete over existing Immich people; existing
  name attaches, new name creates on write. Anchor-less clusters get a prominent
  "Name this cluster" first.
- **Dry-run = top-bar toggle + full-width mode banner + relabeled confirm**
  ("Record (dry-run)" vs "Write N faces"); dry-run batches listed with Promote.
  *(Superseded by the two-tab redesign — see below.)*
- **Store wipes deliberately NOT in the confirmation UI** — they live at batch/config level
  (`reid-wipe` CLI).
- Config surface: the five tuning knobs (window days, distance threshold, confidence floor,
  min box height, face overlap) in the ⚙ dropdown; everything else in the config file.
- Keyboard `j/k/c/s` (move/confirm/skip). *(Dropped in the two-tab redesign.)*

**Direct user hints (original requirements):**

- **Plain language, no jargon** — copy must be usable by a non-technical user.
- **Fully-labeled clusters hidden from the queue by default**, behind a
  "show already labeled" toggle.
- **Pending writes:** unselected items stay pending; the Pending writes tab is 2nd and
  always rendered.
- **Pinned-elsewhere appearances leave the cluster evidence view immediately** —
  presentation only; store rows unchanged until the next run. Pins take effect in
  presentation immediately, in the store at next run.
- **Write-review page layout:** top-bar toggle, person tabs, day sub-groups,
  already-labeled cards green-bordered, an unnamed tab, a Pending writes tab, global
  approve/write buttons, "Write all to Immich (N groups)".
- **Photos are private** — fixture photos must never be committed.
- **Architecture A:** separate core and UI processes; the systemd timer only triggers a
  job on the core. The UI must be runnable independently (e.g. on a laptop) against the
  core on the Pi.

## The two-tab redesign — grilling session 2026-11-25

Source: `.scratch/ui-two-tabs/issues/01-destination-and-ui-decisions.md`.

1. **Two top-level tabs replace the "Write review" button**: tab 1 "Proposals" = the
   existing queue (days → clusters → evidence grid → actions pane); tab 2 "Final check" =
   the restructured write-review surface. UI-only redesign; core API contract and store
   schema unchanged (payloads may grow).
2. **Evidence card**: crop stays the main image, image itself clickable to Immich;
   expandable full-photo view with **all stored appearances of the asset outlined**
   (current highlighted, others dimmed); labeled rows **Immich / ReID / You** (all
   overrides: pin, ignore, reject, dismissal) + date, confidence, Immich URL as text;
   status chip stays. Per-card actions stay: Ignore/Undo, Assign to person…, Not this
   person, resolve-conflict pins, Wrong label. A **selection checkbox separate from
   include-in-write** — *amended by the prototype verdict: the per-card include-in-write
   checkbox is dropped from Proposals cards; cards keep only the select checkbox;
   write-selection happens exclusively in Final check.*
3. **Bulk bar**: appears when ≥1 card selected, scoped to the visible set only (one
   cluster in Proposals, one person in Final check), count shown ("N selected"): Ignore /
   Undo ignore / Assign selected / Not this person.
4. **Actions pane**: cluster controls on top (name prompt, Confirm, Skip cluster),
   "N selected" bulk section below.
5. **Final check**: person pane left (one row per person + counts); grid shows **all
   appearances per person** (proposals, pins, pending writes marked approved/written);
   unnamed clusters stay in Proposals (they need a name first). Item checkboxes **default
   OFF**, Select all / Deselect all per person and globally. Per-person write bar
   ("Write N selected as “X” to Immich") plus one global write-all. **Reassign = "Write as
   X" retargets the pending write now**, not pin-for-next-run.
6. **Dry-run toggle removed** — Final check is always the write gate; dev safety = no
   write-scoped API key configured. Dry-run batch/promote machinery retired (tables
   auto-dropped in store migration).
7. **Labels state semantics**: Proposals = "Pin as X for next run…", Final check =
   "Write as X"; a hint line under the actions and a toast after each action restate which
   semantics applied.
8. **Top bar per tab**: Proposals gets ⚙ settings, "show already labeled", Run now;
   Final check gets Run now. Core-down and run-progress banners stay global on both tabs.
9. **No keyboard shortcuts** in the first version of the new UI; selection is
   checkbox / shift-click.
10. Dark theme and current density kept; full-photo overlay is an inline expand.

## Post-redesign hints (fix sessions, 2026-11-26 — now shipped)

- **"Not a person" vs "Ignore this person" are two distinct buttons** — stored with a
  `reason` (`not_a_person` / `ignore`); both are excluded from clustering and survive
  batch runs and wipes.
- **"Pin as…" must show in Final check immediately** — pin-only appearances (pinned but
  no proposal row) are included in `GET /api/final-check` under the pinned person; the UI
  refreshes Final check after pinning.
- **Select all / Deselect all in the cluster pane bulk bar too** (not just Final check).
- **Button state must match the write path** — "Write N selected" counts only unwritten
  items; already-written items get a disabled checkbox; an empty write shows "all selected
  are already written to Immich", not "nothing selected".
- **Viewer links:** a `localhost`/`127.0.0.1` Immich URL is rewritten client-side to the
  host the page was opened from (port kept); an explicit `[ui] immich_url` /
  `--immich-url` override wins and is never rewritten. `[ui] immich_url` falls back to
  `[immich] base_url`, so the URL is not entered twice when UI and core share a host.
- **The UI runs anywhere:** `reid-ui` takes an optional `[core]`-only config plus
  `--core-url` / `--core-key` / `--immich-url` flags (and `REID_CORE_URL` /
  `REID_CORE_API_KEY` env); it injects `window.__REID_UI__` into `index.html`; it has no
  store volume and holds no state.
