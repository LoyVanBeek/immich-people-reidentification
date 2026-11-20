# Calibration mechanics

Part of: [Re-ID calibration v1 — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: 01

## Question

For each knob in scope — `distance_threshold`, `min_cluster_size`, `face_overlap`, `window_days`, `confidence_floor`, `min_box_height` — what does the `reid-calibrate` pass compute, from what data, and what does it output? Expected shape: sweep candidate values, re-run the affected stages over stored appearances, and score each candidate against the human labels from ticket 01; the report shows precision (fewer wrong proposals) and manual touches side by side.

Pin down:

1. The concrete sweep per knob: which knobs need re-clustering only, which need re-anchoring (and therefore Immich face data — re-fetch per sweep, or persist face detections at run time so sweeps are offline), and which need the gate spot-check labels.
2. What "score" means per knob against the pair labels, and how a suggested value is picked from the curve without silently choosing the trade-off (the human picks the point; what does the pass highlight?).
3. Cold-start behavior: minimum label/pair volumes before a knob gets a suggestion at all, and what the report shows before then.
4. Where suggestions persist and how apply updates the config (the existing `reid.toml` sections and the ⚙/`/api/config` surface), including whether an audit trail of applied changes is kept so they can be reverted.
5. How the six knobs interact when applied together (sweep order, which suggestion is computed against which baseline values).
6. How labels re-bind to appearances after a wipe + re-detection (box-coord IoU, per ticket 01) and what happens to labels that match nothing (orphans: skip, surface, or prune). Label sources are settled: `appearance_label` + pins are positives, `appearance_rejection` is the negative, `gate_verdict` feeds the gates; the `tuning` table is the existing apply surface for suggested values.

Grilling + domain-modeling; cross-reference with `clustering.py`, `anchoring.py`, `pipeline.py`, and the store schema before designing anything.

## Comments

- 2026-11-20: Resolved by grilling against the pipeline code; the pure functions and the threshold-independent merge order turned most sweep mechanics into reuse questions.

## Answer

The sweep reuses the pure pipeline functions entirely in memory — it never touches real proposals.

1. **Sweep mechanics per knob:**
   - `distance_threshold`: one average-linkage merge sequence per window (merge order is threshold-independent — the sequence only stops when the next merge exceeds the threshold), clusters read off at every candidate. Nearly free after the first pass.
   - `window_days`: re-partition + re-cluster per candidate (the greedy partition is cheap).
   - `face_overlap`: re-anchoring only, over fixed clusters. Immich faces are fetched once per calibration run (all appearance-bearing assets) and cached in memory — never persisted, no stale mirror table.
   - `min_cluster_size`: re-anchoring only; it is a single comparison in `build_window`.
   - `confidence_floor` / `min_box_height`: scored from accumulated `gate_verdict` records; no pipeline re-run. Spot-check collection each calibration run re-detects a bounded sample (≤ ~300 assets sampled across the store's date range, detector at `conf=0.01` as the pipeline does), shows a capped near-gate queue across a wide band, and verdicts accumulate across runs. Detection knobs affect newly fetched assets only — the report says so.
2. **Scores per candidate:** wrong proposals (would-be `PROPOSED` rows contradicted by an appearance label or matching a rejection), negative violations (rejected / cross-person pairs merged into one cluster), missed positives (labeled same-person pairs split apart), and manual touches (ask-for-name + conflicted counts). Suggested value = the lexicographic best: among candidates with zero wrong proposals and zero negative violations, the fewest manual touches; if none is clean, the best trade-off point is marked and flagged as such. The full curve is always stored; the highlight is a default the human may override.
3. **Cold start:** the curve always renders with its data volumes; a suggested mark requires ≥ 30 positive pairs and ≥ 10 negative pairs (clustering knobs) or ≥ 20 spot-check verdicts with both verdict classes present (gates). Staleness is visible, never automatic: the report and UI show labels-since-last-calibration; no auto-trigger exists.
4. **Persistence and apply:** a calibration table persists the full curve per knob (candidate → metrics + volumes + computed-at); apply writes the picked value through the existing `tuning`-override path (⚙ / `/api/config`, `_apply_tuning_overrides`); an append-only `tuning_history` journal (knob, old, new, source calibration|manual, timestamp) gives one-click revert.
5. **Joint application:** any subset applies in one action (independent overrides; the next batch run applies them together), with a report note that applying several shifts the baselines the others were computed against — recalibration afterwards is cheap. Knobs are global values only; per-window breakdowns are diagnosis, never separate values.
6. **Re-binding and orphans:** labels re-bind to appearances by box-coord IoU after a wipe + re-detection; orphans (labels matching no current appearance) are counted and surfaced in the report, skipped in scoring, never pruned.
