# Re-ID calibration v1 — Map

Labels: wayfinder:map

## Destination

A buildable spec for a calibration feature: a `reid-calibrate` command that derives suggested values for the clustering knobs (distance threshold, min cluster size, face overlap, window days) and the detection gates (confidence floor, min box height) from accumulated confirmed human input, surfaced in the confirmation UI with explicit apply, reporting precision and manual-touch trade-offs side by side. Ready for `/to-spec`.

## Notes

- Builds on the Re-ID v1 effort ([../person-reid-v1/map.md](../person-reid-v1/map.md)): same store (embeddings, proposals, write ledger, tombstones), same confirmation UI (3-pane workspace, ⚙ dropdown, `/api/config`). Read `goal.md` at the repo root first.
- Scope: the clustering four (`distance_threshold`, `min_cluster_size`, `face_overlap`, `window_days`) + the two detection gates (`confidence_floor`, `min_box_height`).
- Nothing changes without the human: knob values are suggested, never auto-applied; apply is an explicit UI action.
- Signals: confirmed data only — confirm / ask-for-name answer / pin = same-person positive, reject = different-person negative, exclude = per-cycle write intent with no signal, skip = no signal, conflict / already-labeled / tombstone = background signal. (Revised by ticket 01: the code distinguishes reject from exclude; exclude is not a negative.)
- Suggestion mechanics are a sweep: re-cluster stored embeddings at candidate values and score against human labels. No network training of any kind (see Out of scope).
- Facts carried from v1: the proposal table is replaced wholesale each run (ticket 11), and `wipe()` clears the write ledger and tombstones too (ticket 13) — so nothing of the confirmation history survives a wipe today. Ticket 01 must resolve label survival.
- This effort carries spec-writing into the map (like v1 did): the final ticket drafts `spec.md`, the destination artifact, for the human to review before `/to-spec`.
- Skills per ticket type: grilling tickets run `grilling` + `domain-modeling`; the prototype ticket runs `prototype`.

## Decisions so far

<!-- the index: one line per closed ticket, enough to judge relevance, then zoom the link for the detail the ticket holds -->

- [Calibration signal persistence](issues/01-calibration-signal-persistence.md): reject = the calibration negative (exclude is per-cycle write intent, no signal); a new durable `appearance_label` table records confirm-time positives — decision-based, write-outcome-independent, box coords stored so labels re-bind by IoU after a wipe + re-detection; a new durable `gate_verdict` table records spot-check person/not-person verdicts; both survive wipes as operator-voice state; curated external ground truth stays out. Recording only — nothing consumes the tables until ticket 02.
- [Calibration mechanics](issues/02-calibration-mechanics.md): in-memory sweeps reuse the pure pipeline (the threshold-independent merge order makes threshold sweeps nearly free); Immich faces fetched once per calibration run, never persisted; gate sweeps score accumulated spot-check verdicts (sampled re-detection ≤ ~300 assets per run); per-candidate scores are wrong proposals / negative violations / missed positives / manual touches, suggested = lexicographic best (clean first, then fewest touches), full curve always stored, highlight overridable; cold start at ≥ 30 pos / ≥ 10 neg pairs or ≥ 20 verdicts; staleness visible (labels since last calibration), never automatic; curves persisted per knob, apply via the tuning-override path with a `tuning_history` journal; any subset applies at once; global values only; orphans surfaced + counted, never pruned.

## Not yet specified

<!-- see "Fog of war": in-scope fog you can't ticket yet; graduates as the frontier advances -->

- Whether Immich's own face-recognition labels may serve as calibration evidence beyond the operator-voice records — the store keeps the two apart by design (ticket 01) and cold-start minimums are now defined (ticket 02), so this decision hangs on real-world label accumulation after the feature lands, not on anything chartable now.

## Out of scope

<!-- work ruled beyond the destination; closed, never graduates -->

- Retraining or fine-tuning the embedding network (metric learning, projection heads): the human explicitly ruled it out — suggest-and-apply over stored embeddings only.
- Auto-apply or online adaptation of knob values: the ambition ladder was pinned at "suggest, human applies" — knob changes are always explicit human actions.
- Head-detection parameters (SCRFD ratios, input size): plumbing with no honest human signal; the two detection gates in scope are the quality gates only.
- Embedding-model or detector swaps and embedding-store versioning: already fog on the [v1 map](../person-reid-v1/map.md); calibration assumes a stable embedder, since a model change invalidates stored pair distances.
