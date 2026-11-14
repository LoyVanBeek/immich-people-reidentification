# Confirmation UI shape

Part of: [Person Re-ID v1 — Map](../map.md)
Type: prototype
Status: open

## Question

What does the confirmation UI look like and how does it behave? Make a cheap, rough, concrete artifact to react to, covering:

1. The review queue: proposals as cluster → proposed name, with evidence (person crops from the cluster plus the anchor face).
2. The name prompt for anchor-less clusters (new-person onboarding).
3. Dry-run mode toggle and how dry-run results are shown.
4. Store management buttons: wipe embeddings, wipe clusters.
5. Config surface worth exposing (timespan window, thresholds) vs hidden in a config file.

Built against fake data; no Immich connection needed.
