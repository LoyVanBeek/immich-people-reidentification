# 12: Confirmation UI read path over the backing HTTP API

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** The curator opens the confirmation workspace in a browser and reviews the latest run's proposals: the queue, the evidence, the stats. Read-only — no confirmation or writing yet. Layout and behavior follow the prototype on branch `prototype/confirmation-ui` (commit 4d529a2) as the primary source.

**Blocked by:** [11: Cluster appearances and anchor them with Immich face labels](11-clustering-and-anchoring.md)

**Status:** resolved

- [x] The left sidebar lists clusters as a review queue
- [x] The center evidence grid shows one card per appearance: crop, head-box overlay, asset, date, confidence, Immich face label, and a per-appearance status chip (`will write` / `already labeled` / `conflict — skip` / `no head — skip`)
- [x] Include/exclude checkboxes are available per asset over writable appearances
- [x] The right assignment pane shows the name prompt with autocomplete over existing Immich people
- [x] The stats pane shows the latest per-run window summary
- [x] Keyboard navigation works: `j`/`k` move, `c` confirm, `s` skip
- [x] The ⚙ top-bar dropdown exposes the tuning knobs that exist so far (timespan window, cluster distance threshold, detector confidence floor, minimum box height); everything else stays in the config file
- [x] The workspace is served by the container over a small HTTP API; no Immich writes occur from this ticket's behavior
- [x] Tests assert user-visible UI/API states against the prototype as the behavioral reference
