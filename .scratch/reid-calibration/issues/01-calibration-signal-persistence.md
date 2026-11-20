# Calibration signal persistence

Part of: [Re-ID calibration v1 — Map](../map.md)
Type: grilling
Status: resolved

## Question

What must the store retain so a calibration pass can label appearance pairs — and gate crops — from human input? Today nothing of the confirmation history survives in reusable form: the proposal table is replaced wholesale each run, and `wipe()` clears the write ledger and tombstones too. Exclusions within a confirmed cluster, ask-for-name answers, and gate spot-check verdicts have no persistence at all.

Pin down:

1. The confirmation-record model: what rows, what keys, which events (confirm, exclude, ask-for-name answer, gate spot-check verdict) — new tables, or can the write ledger / proposal table carry it?
2. Whether calibration labels survive `reid-wipe` (today wiping embeddings also erases the write ledger and tombstones), or whether a wipe honestly resets calibration too.
3. The exact signal semantics per event, as pinned by the effort decision: confirm = same-person positive, exclude from a confirmed cluster = different-person negative, skip = no signal, conflict / already-labeled / tombstone = background signal.
4. What the gate spot-check queue must record (which crop was shown, at which gate value, and the person/not-person verdict).

Grilling + domain-modeling; add new glossary terms as they crystallize. Cross-reference with the actual store schema before deciding — don't design tables the code already has.

## Comments

- 2026-11-20: Grilled across sessions; the store's actual vocabulary (reject vs exclude, the durable operator-voice tables) revised the charting-time signal semantics mid-flight. The grilling ran under plan-mode constraints, so the claim was set together with the resolution.

## Answer

The store already keeps most of the operator's voice; calibration needs exactly two new tables and one vocabulary revision.

1. **Negatives: reject, not exclude.** `appearance_rejection` ("this appearance is not this person", name-keyed, durable) is the different-person negative. The include/exclude checkbox (`write_decision`) is per-cycle write intent, cleared every run — no calibration signal. This revises the charting decision "exclude = negative": exclude means "not now", not "different person".
2. **Positives: a new durable `appearance_label` table**, written at confirm time (confirm, confirm-all, dry-run confirm) for every included appearance regardless of write outcome — so no-head and ask-for-name appearances are labeled too. The decision is the signal, not the write; an unpromoted dry-run batch still carries valid labels. Existing pins (`appearance_pin`) remain positives as well. Operator-voice state: survives wipes, like pins/rejections/ignores/tuning already do. Rows key on `(asset_id, appearance)` and also store `person_name` + `person_id` (either may be missing for a brand-new person), box coords, the source event, and a timestamp.
3. **Re-binding:** the stored box coords are the durable identity. After a wipe + re-detection (appearance indexes can shift), the calibration pass re-binds labels to appearances by IoU overlap; the key stays `(asset_id, appearance)`.
4. **Spot-check verdicts: a new durable `gate_verdict` table** — `(asset_id, box, gate_context)` plus confidence, box height, verdict person|not_person, and timestamp, deduped by key. Spot-check crops are detections the gates dropped, so no appearance row exists to key off.
5. **External curated ground truth stays out** — confirmed-input-only. V1's manual eval loop (`data/eval-labels.toml`, ticket 17) remains the interim path until this lands; the store holds operator-voice records only.

Scope: recording only. Nothing consumes these tables yet — the calibration pass that reads them is ticket 02.
