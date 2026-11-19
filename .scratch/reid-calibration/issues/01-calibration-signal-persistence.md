# Calibration signal persistence

Part of: [Re-ID calibration v1 — Map](../map.md)
Type: grilling
Status: open

## Question

What must the store retain so a calibration pass can label appearance pairs — and gate crops — from human input? Today nothing of the confirmation history survives in reusable form: the proposal table is replaced wholesale each run, and `wipe()` clears the write ledger and tombstones too. Exclusions within a confirmed cluster, ask-for-name answers, and gate spot-check verdicts have no persistence at all.

Pin down:

1. The confirmation-record model: what rows, what keys, which events (confirm, exclude, ask-for-name answer, gate spot-check verdict) — new tables, or can the write ledger / proposal table carry it?
2. Whether calibration labels survive `reid-wipe` (today wiping embeddings also erases the write ledger and tombstones), or whether a wipe honestly resets calibration too.
3. The exact signal semantics per event, as pinned by the effort decision: confirm = same-person positive, exclude from a confirmed cluster = different-person negative, skip = no signal, conflict / already-labeled / tombstone = background signal.
4. What the gate spot-check queue must record (which crop was shown, at which gate value, and the person/not-person verdict).

Grilling + domain-modeling; add new glossary terms as they crystallize. Cross-reference with the actual store schema before deciding — don't design tables the code already has.
