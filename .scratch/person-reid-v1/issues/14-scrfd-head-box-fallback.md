# 14: SCRFD fallback head-box detection

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** Appearances whose cluster has no Immich-provided head box still get one: a head detector runs on the full-person crop and expands the face box to a head box, turning "no head — skip" appearances into writable ones.

**Blocked by:** [13: Confirm-and-write: head boxes, person creation, and the write ledger](13-confirm-and-write-back.md)

**Status:** resolved

- [ ] When no Immich unassigned face is contained in the person box, SCRFD-500M at 320×320 (weights vendored into the image) detects a head on the appearance's full-person crop
- [ ] The detected face box is expanded geometrically to a head box, with tunable constants
- [ ] When neither source yields a head, the appearance remains skipped and flagged "no head found"
- [ ] The face→person containment rule shared with anchoring remains the single tunable knob in the ⚙ dropdown
- [ ] Tests assert the fallback ordering through the stub server: Immich's face preferred, the detector only otherwise
