# Confirmation UI shape

Part of: [Person Re-ID v1 — Map](../map.md)
Type: prototype
Status: resolved

## Question

What does the confirmation UI look like and how does it behave? Make a cheap, rough, concrete artifact to react to, covering:

1. The review queue: proposals as cluster → proposed name, with evidence (person crops from the cluster plus the anchor face).
2. The name prompt for anchor-less clusters (new-person onboarding).
3. Dry-run mode toggle and how dry-run results are shown.
4. Store management buttons: wipe embeddings, wipe clusters.
5. Config surface worth exposing (timespan window, thresholds) vs hidden in a config file.

Built against fake data; no Immich connection needed.

## Comments

- 2026-11-15: Prototype built at `prototype/confirmation-ui.html` — single self-contained HTML file (no app exists yet; repo is spec-stage), open directly in a browser. Three variants switchable via `?variant=A|B|C` + floating bottom bar (←/→ keys work). Bakes in ticket 07's settled semantics: conflicts skip+flag, "no head found" skip list, dry-run batches with promotion, existing-person autocomplete in the name prompt. Verdict pending — no variant promoted, nothing committed yet.
- 2026-11-15: Bug fix after first look — `modeLabel` lookup key mismatch (`no-head` vs `nohead`) killed variants A/B at render, and variant C's CSS anchor (`#C`) didn't wrap its body so its cards rendered unstyled. Both fixed; all three variants verified rendering with zero console errors.
- 2026-11-15: **Verdict**: variant **B** wins, with **A's config drawer folded into B's top bar** as a ⚙ dropdown. Merged into the prototype before capture; committed to throwaway branch `prototype/confirmation-ui` (4d529a2) as the primary source; working tree on main keeps an untracked copy for browsing.

## Answer

**Winner: variant B — the 3-pane workspace — with variant A's config drawer folded into the top bar (⚙ dropdown).** Variants A (queue list) and C (triage grid + admin tab) are rejected as home shapes but contributed the config-drawer pattern.

What the confirmation UI is, as settled by the prototype:

1. **Review queue**: a left sidebar of mini rows (crop pair, name-or-"unnamed", cluster id + confidence, status) selects the cluster; the **center evidence grid** shows one card per appearance — crop with head-box overlay, asset, date, confidence, per-appearance write-status chip (`will write` / `already labeled` / `conflict — skip` / `no head — skip`, per ticket 07) and the Immich face label — with **per-asset include/exclude checkboxes** over the writable ones. The right pane holds the assignment: name, stats, confirm/skip.
2. **Name prompt** is inline in the right pane: an "Assign to" input with autocomplete over existing Immich people; an existing name attaches the cluster to that person, a new name creates the person on write (the ticket 07 contract). Anchor-less clusters surface a prominent "Name this cluster" button first.
3. **Dry-run** is a top-bar toggle with a full-width mode banner ("DRY-RUN — nothing will be written to Immich" vs "LIVE"); the confirm button relabels accordingly ("Record (dry-run)" vs "Write N faces"). Dry-run batches are listed (sidebar in the prototype) with a **Promote** action per batch.
4. **Store management is deliberately not in the confirmation UI** — wipes live at batch-run/config level. (Variant C's admin tab demonstrated the alternative; rejected for the home shape.)
5. **Config surface**: the five tuning knobs (timespan window, cluster distance threshold, detector confidence floor, min box height, face→person containment) live in the ⚙ top-bar dropdown; everything else stays in the config file.

Keyboard: `j`/`k` move between clusters, `c` confirm, `s` skip.

Primary source: branch `prototype/confirmation-ui` (all three variants + switcher). Main keeps only this decision.
