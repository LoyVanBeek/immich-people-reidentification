# 11: Cluster appearances and anchor them with Immich face labels

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** Each batch run turns stored embeddings into proposals: clusters rebuilt from scratch within the timespan window, anchored with Immich's own named face labels, and surfaced as proposals plus a window summary. Nothing is written to Immich yet — proposals exist only in the local store and run output.

**Blocked by:** [10: Detect, gate, and embed person appearances into the incremental store](10-person-appearance-embeddings.md)

**Status:** ready-for-agent

- [ ] Clusters are rebuilt from scratch every run; window membership by EXIF capture time in the local timezone with upload-time fallback; the timespan window is configurable (default: same calendar day)
- [ ] Clustering uses cosine distance over normalized embeddings with agglomerative average linkage at a configurable distance threshold
- [ ] An appearance that joined no cluster remains a singleton: stored, but never surfaced in proposals or the ask-for-name queue
- [ ] An Immich face detection claims an appearance when its face box overlaps the appearance's person box (intersection / face-box-area ≥ 0.5, tunable; best match wins if several qualify)
- [ ] A cluster whose named anchors are unanimous (any count, one name) proposes that name for every unlabeled appearance in it
- [ ] A cluster with ≥ 2 distinct anchor names is blocked entirely and surfaced as conflicted
- [ ] Anchor-less clusters with at least the minimum number of appearances (default 2, tunable) enter the ask-for-name queue
- [ ] Appearances Immich has already labeled never receive proposals
- [ ] Every run emits a window summary: cluster count, anchored %, conflicted %, anchor-less %, singleton count
- [ ] The pure clustering/anchoring rules are tested directly with synthetic embedding vectors (unanimity, conflicts, singleton handling, ask-for-name queue sizing, threshold edge cases)
