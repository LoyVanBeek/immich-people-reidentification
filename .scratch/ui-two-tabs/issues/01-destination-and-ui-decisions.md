# Destination and UI decisions (grilling session 2026-11-25)

Part of: [UI Two Tabs — Map](../map.md)
Type: grilling
Status: resolved

## Question

What is the destination of the two-tab UI effort, and which UI decisions are settled
before any prototyping starts? (Grilled with the user in one session; every question below
was answered, most with the recommended option.)

## Answer

**Destination**: redesign the `reid-ui` confirmation interface as two top-level tabs —
"Proposals" and "Final check" — prototype first, then real build. UI-only; core API
contract and store schema unchanged (payloads may grow).

Settled decisions:

1. **Tabs** — a top-level tab bar replaces the "Write review" button. Tab 1 "Proposals" =
   today's queue (days pane → clusters pane → evidence grid → actions pane). Tab 2
   "Final check" = today's write-review surface, restructured (see 6).
2. **Scope** — visual/UX redesign of the whole UI on top of the existing structure and
   data; not a backend rewrite. `/api/evidence` and the write-review payloads may grow
   fields (sibling detections with boxes, explicit override fields); no endpoint removals
   or store schema changes beyond the dry-run retirement (7).
3. **Evidence card** — keep the appearance crop as the main image, make the image itself
   clickable to Immich; add an expandable full-photo view with **all stored appearances of
   the asset** outlined (current appearance highlighted, others dimmed). Gate-dropped
   detections are not stored, so they cannot appear — accepted. Card also shows explicit
   labeled rows **Immich:** (face label if any) · **ReID:** (proposal/status) · **You:** (all
   overrides — pin, ignore, reject, dismissal) plus date, confidence, and the Immich URL
   as text; the status chip stays. Per-card actions stay: Ignore/Undo, Assign to person…,
   Not this person, resolve-conflict pins, Wrong label. A **selection checkbox** is added,
   separate from the existing include-in-write checkbox.
4. **Bulk actions** — when ≥1 card is selected, a bulk bar appears scoped to the currently
   visible set only (one cluster in Proposals, one person in Final check), with the count
   shown: Ignore selected, Undo ignore, Assign selected to person X…, Not this person.
5. **Actions pane (Proposals)** — cluster-level controls stay on top (name prompt for
   unnamed clusters, Confirm/Approve, Skip cluster); below them a "N selected" section with
   the bulk actions. The per-card include-in-write checkbox stays.
6. **Final check** — a Person pane on the left (one row per person with counts); clicking a
   person shows a grid of **all appearances assigned to that person**: ReID proposals,
   human pins/corrections, and pending writes (marked as approved/written). Pending writes
   fold into the person's grid; unnamed clusters stay in Proposals (they need a name first).
   Each grid item has a checkbox, **default OFF**, with Select all / Deselect all per
   person and globally. Writes happen via a per-person bar ("Write N selected to Immich")
   plus one global write-all-across-people button. Reassigning an item (per-item **and**
   bulk) **retargets the pending write now** ("Write as X") — not pin-for-next-run.
7. **Dry-run toggle removed** — Final check always writes; development safety = no
   write-scoped API key configured. This supersedes the person-reid-v1 note "dry-run mode
   exists from day one": the two-tab Final check is now the human-confirmation gate.
   The dry-run batch/promote machinery is retired (ticket 05).
8. **Semantics clarity** — button labels state the semantics: Proposals uses "Pin as X for
   next run…", Final check uses "Write as X"; a one-line hint under the actions and a toast
   after each action restate which semantics applied.
9. **Top-bar controls per tab** — Proposals: ⚙ settings, "show already labeled", Run now.
   Final check: Run now. Core-down banner and run-progress banner stay global (both tabs).
10. **No keyboard shortcuts** (j/k/c/s dropped) in the first version of the new UI;
    selection is checkbox / shift-click.
11. **Process** — rough prototype of both tabs + the new card against mock data on a
    throwaway branch first (ticket 02), then the real implementation.

## Comments

- 2026-11-25 — Amended by the [Prototype the two-tab UI](02-prototype-two-tab-ui.md) verdict:
  the per-card include-in-write checkbox is **dropped** from Proposals cards (decisions 3
  and 5 said it stays). Cards keep only the "select" checkbox for bulk actions;
  write-selection happens exclusively in Final check (default-off checkboxes per person).
  Also settled by the verdict: dark theme + current density kept; full-photo overlay is an
  inline expand; the per-person write button names the person ("Write N selected as “X” to
  Immich").
