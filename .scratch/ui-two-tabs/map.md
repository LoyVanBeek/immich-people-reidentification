# UI Two Tabs — Map

Labels: wayfinder:map

Related: [Person Re-ID v1 — Map](../person-reid-v1/map.md).

## Destination

A two-tab `reid-ui`: **Proposals** (see and check proposals from anchored clusters, make
review decisions — the editing) and **Final check** (a person-centric last look before
anything is written to Immich). Prototype first on a throwaway branch, then the real build.
The core API contract and SQLite schema stay unchanged; `/api/evidence` and the
write-review payloads may grow fields. The dry-run toggle is retired — Final check is the
human write gate, and development safety comes from not configuring a write-scoped API key.

## Notes

- Single user, Pi 5, Python/uv. UI is stdlib `http.server` static assets + `/api/*` proxy:
  `src/reid/server.py`, `src/reid/core.py`, `src/reid/ui/`.
- The current UI already has both views behind a "Write review" button; this effort promotes
  them to real tabs and redesigns the evidence card and final-check surfaces (ticket 01).
- Supersedes two person-reid-v1 decisions: the dry-run toggle + promote flow (map note
  "dry-run mode exists from day one") is replaced by the Final check gate + no-write-key dev
  mode; ticket 06's variant-B panes stay as the Proposals tab skeleton.
- Vocabulary per `GLOSSARY.md`: proposal, cluster, person appearance, pin, ignore, reject,
  review decision. "Label as person X" in Proposals = pin (next run); in Final check =
  retarget the pending write now.
- Skills per ticket type: prototype → `prototype`; grilling → `grilling` + `domain-modeling`;
  task → none. Tickets are decisions; build work graduates after tickets 02–05 resolve.

## Decisions so far

- [Destination and UI decisions (grilling session 2026-11-25)](issues/01-destination-and-ui-decisions.md):
  two tabs "Proposals" / "Final check"; new evidence card (clickable crop + expandable
  full-photo overlay of stored sibling appearances, Immich/ReID/You labeled rows);
  selection checkbox separate from include-in-write, bulk bar scoped to the visible set;
  actions pane = cluster controls on top + "N selected" bulk section; Final check = person
  pane → grid of all appearances per person with default-off checkboxes, per-person +
  global write bars, reassign retargeting the write now; no keyboard shortcuts; dry-run
  toggle removed; core contract stays, payloads may grow; prototype first, then build.
- [Prototype the two-tab UI](issues/02-prototype-two-tab-ui.md): the decided structure won
  as prototyped (branch `prototype/ui-two-tabs`, `a2978ac` + verdict commit). Amendments:
  per-card write checkbox dropped (write-selection only in Final check, default off);
  dark theme kept; full-photo overlay = inline expand; per-person write button names the
  person ("Write N selected as “X” to Immich").
- [Evidence API payload growth for the new card](issues/03-evidence-payload-growth.md):
  sibling outlines ship **in the JSON payload** — each card gains `siblings` (all stored
  appearances of the asset, pixel-space boxes), `sibling_count` (badge), `rejected_name`,
  and `dismissed`; the full photo is served by a new `/api/crop?full=1` mode at native
  pixel-source resolution and the UI overlays SVG (viewBox = natural size). Immich URL
  stays a single `immich_url` base in `/api/state`, links built client-side. No other
  endpoint changes shape here.
- [Final check API reshape (person-centric) and bulk endpoints](issues/04-final-check-api-reshape.md):
  new `GET /api/final-check` (persons from proposals+pins+ledger, items carry source +
  written); bulk actions as separate endpoints `/api/bulk-ignore|unignore|pin|reject` with
  explicit item lists; retarget = `POST /api/retarget` (pin) + per-item names in
  `POST /api/write-faces`; writes resolve/create the person at write time with
  ledger/tombstone/conflict checks; "Confirm proposal" in Proposals becomes a confirmed
  cluster state (no write). Death list for 05 recorded in the ticket.
- [Retire the dry-run batch / promote machinery](issues/05-retire-dry-run-machinery.md):
  audited (UI/core/store/tests references enumerated), death list finalized, existing
  `dry_run_batch`/`dry_run_action`/`write_decisions` rows auto-drop in `_migrate` (DROP
  TABLE IF EXISTS; proposals survive, so nothing is lost). Code removal graduates as
  ticket 07, landing with the new UI (ticket 06) so the app never sits broken.

## Not yet specified

- (empty — the map is fully walked. All tickets resolved: 01–05 decisions, 06 build
  (`806879e`), 07 retirement (`53a1d69`).)

## Status

**Done.** The two-tab UI (Proposals / Final check) is built, the old write/review
machinery is retired, and the only write path is `POST /api/write-faces` from the Final
check tab. Verified: pre-commit green on `53a1d69` (ruff, mypy strict, pytest 150 passed),
jsdom UI smoke suite 5/5. Prototype primary source: branch `prototype/ui-two-tabs`
(`a2978ac`, `fbb9a92`).

## Out of scope

- Store schema changes; endpoint removals beyond the dry-run retirement (ticket 05).
- Keyboard shortcuts (j/k/c/s) in the first version of the new UI.
- Gate-dropped detections in the full-photo overlay — stored appearances only.
- Any change to pipeline behavior: detection, gates, clustering, anchoring, head boxes,
  write-back semantics, write ledger, tombstones.
- Historical backfill, video assets, multi-user (still out of scope from person-reid-v1).
