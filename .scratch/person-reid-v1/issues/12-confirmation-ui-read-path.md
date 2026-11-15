# 12: Confirmation UI read path over the backing HTTP API

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** The curator opens the confirmation workspace in a browser and reviews the latest run's proposals: the queue, the evidence, the stats. Read-only — no confirmation or writing yet. Layout and behavior follow the prototype on branch `prototype/confirmation-ui` (commit 4d529a2) as the primary source.

**Blocked by:** [11: Cluster appearances and anchor them with Immich face labels](11-clustering-and-anchoring.md)

**Status:** ready-for-agent

- [ ] The left sidebar lists clusters as a review queue
- [ ] The center evidence grid shows one card per appearance: crop, head-box overlay, asset, date, confidence, Immich face label, and a per-appearance status chip (`will write` / `already labeled` / `conflict — skip` / `no head — skip`)
- [ ] Include/exclude checkboxes are available per asset over writable appearances
- [ ] The right assignment pane shows the name prompt with autocomplete over existing Immich people
- [ ] The stats pane shows the latest per-run window summary
- [ ] Keyboard navigation works: `j`/`k` move, `c` confirm, `s` skip
- [ ] The ⚙ top-bar dropdown exposes the tuning knobs that exist so far (timespan window, cluster distance threshold, detector confidence floor, minimum box height); everything else stays in the config file
- [ ] The workspace is served by the container over a small HTTP API; no Immich writes occur from this ticket's behavior
- [ ] Tests assert user-visible UI/API states against the prototype as the behavioral reference
