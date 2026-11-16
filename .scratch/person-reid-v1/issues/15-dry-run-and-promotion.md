# 15: Dry-run mode and promotion batches

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** The curator can explore and stage proposals with provably zero Immich side effects, then make them real deliberately. Dry-run confirmations collect into batches that are promoted — replayed as real writes — when the curator chooses.

**Blocked by:** [13: Confirm-and-write: head boxes, person creation, and the write ledger](13-confirm-and-write-back.md)

**Status:** resolved

- [ ] Dry-run is a top-bar toggle with an unmistakable full-width mode banner and a relabeled confirm button
- [ ] Confirming in dry-run mode produces zero Immich writes — no faces written, no person creation
- [ ] Dry-run confirmations collect into persisted batches (proposals flagged as dry-run), listed in the UI with a per-batch Promote action
- [ ] Promoting a batch replays its recorded actions in order: create the person if new, then one face entry per asset, journaling each into the write ledger
- [ ] Promotion is idempotent: already-written pairs are no-ops via the write ledger and check-before-write, so promoting twice produces no duplicates
- [ ] Tests assert through the stub server: absence of writes in dry-run, and the replay order on promotion
