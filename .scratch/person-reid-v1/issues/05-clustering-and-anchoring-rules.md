# Clustering and anchoring rules

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: open
Blocked by: [Model and detector selection](04-model-and-detector-selection.md)

## Question

Design the rule set so all thresholds are tunable config:

1. How clusters form: distance metric and threshold over the chosen embeddings, within the configurable timespan window (default same calendar day).
2. Cluster granularity: one cluster per person per window, or other shapes?
3. Anchor selection: when multiple Immich-recognized faces appear in one cluster and disagree on the name, which anchor wins and when does disagreement block a proposal?
4. Anchor-less clusters: the ask-for-name fallback flow.
5. Plan to validate the assumption "every cluster contains at least one recognized face" against real data.
