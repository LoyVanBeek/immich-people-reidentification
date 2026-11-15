# 13: Confirm-and-write: head boxes, person creation, and the write ledger

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** Confirming a proposal in the workspace writes real labels into Immich: one face entry per asset carrying a head box bound to the target person. Immich's own labels are never overwritten; repeated confirmations cannot duplicate writes; corrections made directly in Immich are respected. At this point Immich's own unassigned face detections are the only head-box source — the detector fallback arrives in ticket 14, so "no head" appearances are skipped and flagged.

**Blocked by:** [12: Confirmation UI read path over the backing HTTP API](12-confirmation-ui-read-path.md)

**Status:** ready-for-agent

- [ ] Confirming resolves each writable appearance's head box lazily: Immich's own unassigned face detection preferred where contained in the person box (≥ 0.5 containment, best match wins); assets with no such face are skipped and flagged "no head found"
- [ ] The chosen name attaches to an existing Immich person when it matches one; a typed new name creates the person in Immich at confirm time
- [ ] Writes go through `POST /faces` as one face entry per cluster asset carrying the head box bound to the target person; existing faces are never reassigned
- [ ] A write-scoped API key is used only where writes happen (confirm/promote); everything else stays read-scoped
- [ ] An asset already carrying a different person's overlapping face label is skipped and flagged as a conflict — never overwritten; non-overlapping different-person faces are just group photos
- [ ] An asset already carrying the target person's face is skipped as already-labeled evidence
- [ ] Every confirmed write is journaled in the write ledger keyed by (assetId, personId); before each write the asset's existing faces are fetched and checked, so a crash between write and journal entry cannot produce duplicates on retry
- [ ] Re-confirming an already-written cluster is a no-op
- [ ] A label the user removed manually in Immich becomes a tombstone for that asset–person pair and is never re-proposed
- [ ] The include/exclude checkboxes determine which appearances are written
- [ ] Tests assert the exact Immich call sequences through the stub server: person creation, `POST /faces` payloads, and idempotent behavior on retry
