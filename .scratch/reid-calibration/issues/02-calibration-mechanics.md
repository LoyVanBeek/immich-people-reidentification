# Calibration mechanics

Part of: [Re-ID calibration v1 — Map](../map.md)
Type: grilling
Status: open
Blocked by: 01

## Question

For each knob in scope — `distance_threshold`, `min_cluster_size`, `face_overlap`, `window_days`, `confidence_floor`, `min_box_height` — what does the `reid-calibrate` pass compute, from what data, and what does it output? Expected shape: sweep candidate values, re-run the affected stages over stored appearances, and score each candidate against the human labels from ticket 01; the report shows precision (fewer wrong proposals) and manual touches side by side.

Pin down:

1. The concrete sweep per knob: which knobs need re-clustering only, which need re-anchoring (and therefore Immich face data — re-fetch per sweep, or persist face detections at run time so sweeps are offline), and which need the gate spot-check labels.
2. What "score" means per knob against the pair labels, and how a suggested value is picked from the curve without silently choosing the trade-off (the human picks the point; what does the pass highlight?).
3. Cold-start behavior: minimum label/pair volumes before a knob gets a suggestion at all, and what the report shows before then.
4. Where suggestions persist and how apply updates the config (the existing `reid.toml` sections and the ⚙/`/api/config` surface), including whether an audit trail of applied changes is kept so they can be reverted.
5. How the six knobs interact when applied together (sweep order, which suggestion is computed against which baseline values).

Grilling + domain-modeling; cross-reference with `clustering.py`, `anchoring.py`, `pipeline.py`, and the store schema before designing anything.
