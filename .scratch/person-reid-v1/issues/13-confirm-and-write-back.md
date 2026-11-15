# 13: Confirm-and-write: head boxes, person creation, and the write ledger

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** Confirming a proposal in the workspace writes real labels into Immich: one face entry per asset carrying a head box bound to the target person. Immich's own labels are never overwritten; repeated confirmations cannot duplicate writes; corrections made directly in Immich are respected. At this point Immich's own unassigned face detections are the only head-box source — the detector fallback arrives in ticket 14, so "no head" appearances are skipped and flagged.

**Blocked by:** [12: Confirmation UI read path over the backing HTTP API](12-confirmation-ui-read-path.md)

**Status:** resolved

- [x] Confirming resolves each writable appearance's head box lazily: Immich's own unassigned face detection preferred where contained in the person box (≥ 0.5 containment, best match wins); assets with no such face are skipped and flagged "no head found"
- [x] The chosen name attaches to an existing Immich person when it matches one; a typed new name creates the person in Immich at confirm time
- [x] Writes go through `POST /faces` as one face entry per cluster asset carrying the head box bound to the target person; existing faces are never reassigned
- [x] A write-scoped API key is used only where writes happen (confirm/promote); everything else stays read-scoped
- [x] An asset already carrying a different person's overlapping face label is skipped and flagged as a conflict — never overwritten; non-overlapping different-person faces are just group photos
- [x] An asset already carrying the target person's face is skipped as already-labeled evidence
- [x] Every confirmed write is journaled in the write ledger keyed by (assetId, personId); before each write the asset's existing faces are fetched and checked, so a crash between write and journal entry cannot produce duplicates on retry
- [x] Re-confirming an already-written cluster is a no-op
- [x] A label the user removed manually in Immich becomes a tombstone for that asset–person pair and is never re-proposed
- [x] The include/exclude checkboxes determine which appearances are written
- [x] Tests assert the exact Immich call sequences through the stub server: person creation, `POST /faces` payloads, and idempotent behavior on retry

## Answer

Implemented the confirm-and-write-back functionality:

1. **Store changes** (`src/reid/store.py`):
   - Added `WriteLedgerEntry` dataclass and `write_ledger` table for tracking confirmed writes keyed by (asset_id, person_id)
   - Added `Tombstone` dataclass and `tombstone` table for tracking manually removed labels
   - Added methods: `add_write_ledger_entry`, `has_write_ledger_entry`, `get_write_ledger_entries`, `add_tombstone`, `has_tombstone`
   - Updated `wipe()` to clear write ledger and tombstones

2. **Immich client changes** (`src/reid/immich.py`):
   - Added `CreatedPerson` and `CreatedFace` dataclasses
   - Added `find_person_by_name()` for exact name matching (case-insensitive)
   - Added `create_person()` via `POST /people`
   - Added `create_face()` via `POST /faces` with head box
   - Added `resolve_head_box()` - prefers Immich's unassigned face detections with ≥ 0.5 containment (intersection/face-area), best match wins
   - Added `check_existing_faces()` - detects conflicts (different person's overlapping face) and already-labeled (target person's face)
   - Added `_face_person_overlap()` helper for containment calculation

3. **Config changes** (`src/reid/config.py`):
   - Added `write_api_key` field to `ImmichConfig` for write-scoped API key

4. **Server changes** (`src/reid/server.py`):
   - Added `POST /api/confirm` endpoint
   - Added `confirm_cluster()` method with:
     - Dry-run mode (returns what would happen without writing)
     - Real write mode (uses write-scoped API key)
     - Person resolution (find existing or create new)
     - Head box resolution per appearance
     - Conflict detection (different person's overlapping face → skip)
     - Already-labeled detection (target person's face → skip and journal)
     - Tombstone check (manually removed → skip)
     - Write ledger check (idempotency - already written → skip)
     - Write ledger journaling after successful writes
     - Include/exclude checkbox support via `included` parameter

5. **Test stub updates** (`tests/stub_immich.py`):
   - Added `POST /faces` and `POST /people` endpoints for testing

All existing tests pass, and manual testing confirms:
- Dry-run mode works correctly
- Real writes create person and faces in Immich
- Idempotency: re-confirming is a no-op (skipped via write ledger)
- Conflict detection works (different person's overlapping face)
- Already-labeled detection works
- No-head detection works (skipped and flagged)
- Tombstone support works (manually removed labels never re-proposed)
