# Clustering and anchoring rules

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: [Model and detector selection](04-model-and-detector-selection.md)

## Question

Design the rule set so all thresholds are tunable config:

1. How clusters form: distance metric and threshold over the chosen embeddings, within the configurable timespan window (default same calendar day).
2. Cluster granularity: one cluster per person per window, or other shapes?
3. Anchor selection: when multiple Immich-recognized faces appear in one cluster and disagree on the name, which anchor wins and when does disagreement block a proposal?
4. Anchor-less clusters: the ask-for-name fallback flow.
5. Plan to validate the assumption "every cluster contains at least one recognized face" against real data.

## Comments

- Round 1: all six recommendations accepted ("yes to all"). Metric/algorithm (Q1), granularity + day boundary (Q2), anchor matching + conflict policy (Q3), named-anchors-only (Q4), ask-for-name queue with min size (Q5), live validation via run summary (Q6). Full detail in `## Answer`.

## Answer

All thresholds below are tunable config; values are initial defaults explicitly marked for calibration against real data.

**1. Cluster formation** — **cosine distance** over the normalized 512-dim `osnet_ain_x1_0` embeddings, clustered with **agglomerative hierarchical clustering, average linkage, distance threshold**: deterministic, no cluster count needed, instant at tens of appearances per day. Threshold is the primary tuning knob (initial default **0.5** cosine distance — *to calibrate*). Single-link/connected-components rejected (chains A~B~C into one cluster even when A≠C); DBSCAN rejected (extra knob, no gain at this scale). Appearances that join no cluster remain **singletons**: stored, but never generate proposals.

**2. Granularity & day boundary** — a **cluster is one person's appearance-group within one window**, deliberately *not* a persistent identity. Splitting within a day (clothes change) is legitimate: two clusters may both anchor "Albert" and both propose Albert. Clusters are never merged across windows. Window membership by **asset capture time (EXIF) in the Pi's local timezone**, upload-time fallback when EXIF is missing.

**3. Anchor matching & conflict policy** — an Immich face detection claims a person-appearance when its face box **overlaps our person box** on the same asset (intersection / face-box-area ≥ 0.5, tunable — the same containment rule ticket 07 uses for head-box reuse; best match wins if several faces qualify). Rules per cluster:
- All named anchors **unanimous** (any count, one name) → propose that name for every *unlabeled* appearance in the cluster.
- **Any disagreement** (≥ 2 distinct names among anchors) → **block the entire cluster's proposals** and surface it as conflicted in the UI. v1 never guesses between conflicting ground truth.

**4. Unnamed Immich faces** — v1 uses **named anchors only**. Matching on unnamed person-IDs (must-link constraints or validation signals) is deferred as a tuning knob until real data shows whether it's needed.

**5. Anchor-less clusters** — clusters with **≥ 2 appearances** and no named anchor enter the **ask-for-name queue** in the UI (visual shape is issue 06's). Minimum size tunable, default **2**. Lone unlabeled singletons are indistinguishable from detector noise: stored, never surfaced.

**6. Validation of "every cluster contains an anchor"** — no dedicated exercise; every nightly run emits a **window summary**: cluster count, anchored %, conflicted %, anchor-less %, singleton count — plus per-cluster detail in dry-run mode. Observed live over the first weeks of real use; a persistently high anchor-less % is the trigger to revisit (e.g. enable person-ID constraints from 4).
