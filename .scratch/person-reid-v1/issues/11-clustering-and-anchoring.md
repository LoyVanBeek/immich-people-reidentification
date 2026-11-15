# 11: Cluster appearances and anchor them with Immich face labels

Part of: [Person Re-ID v1 — Map](../map.md)

**What to build:** Each batch run turns stored embeddings into proposals: clusters rebuilt from scratch within the timespan window, anchored with Immich's own named face labels, and surfaced as proposals plus a window summary. Nothing is written to Immich yet — proposals exist only in the local store and run output.

**Blocked by:** [10: Detect, gate, and embed person appearances into the incremental store](10-person-appearance-embeddings.md)

**Status:** resolved

- [x] Clusters are rebuilt from scratch every run; window membership by EXIF capture time in the local timezone with upload-time fallback; the timespan window is configurable (default: same calendar day)
- [x] Clustering uses cosine distance over normalized embeddings with agglomerative average linkage at a configurable distance threshold
- [x] An appearance that joined no cluster remains a singleton: stored, but never surfaced in proposals or the ask-for-name queue
- [x] An Immich face detection claims an appearance when its face box overlaps the appearance's person box (intersection / face-box-area ≥ 0.5, tunable; best match wins if several qualify)
- [x] A cluster whose named anchors are unanimous (any count, one name) proposes that name for every unlabeled appearance in it
- [x] A cluster with ≥ 2 distinct anchor names is blocked entirely and surfaced as conflicted
- [x] Anchor-less clusters with at least the minimum number of appearances (default 2, tunable) enter the ask-for-name queue
- [x] Appearances Immich has already labeled never receive proposals
- [x] Every run emits a window summary: cluster count, anchored %, conflicted %, anchor-less %, singleton count
- [x] The pure clustering/anchoring rules are tested directly with synthetic embedding vectors (unanimity, conflicts, singleton handling, ask-for-name queue sizing, threshold edge cases)

## Comments

Implemented on `main` (2026-11-15). All ten acceptance criteria verified:
28 pure-logic tests over synthetic embedding vectors
(`tests/test_clustering.py`) plus 9 new pytest acceptance tests through the
real CLI against the stub Immich (63 tests total, `mypy --strict` clean).

**Windowing.** Each run re-clusters *all* stored appearances, so window
membership needs every appearance's capture time — the appearance row now
persists `captured_at`, resolved by the client as
`exifInfo.dateTimeOriginal`, falling back to `createdAt` (upload time).
Capture instants convert to the container's local timezone (`capture_day`
in `clustering.py`); appearances with unusable capture times cannot be
windowed and are excluded from clustering entirely (stored, never
surfaced, warning logged) until a wipe rebuilds the store. Windows are
blocks of at most `window_days` consecutive calendar days, filled greedily
from the earliest day (`partition_windows`, wired into the run); default
1 = same calendar day. Stores written before this ticket migrate via
`ALTER TABLE` with an empty capture time (wipe + re-embed restores proper
windows).

**Clustering.** Pure module `clustering.py`: cosine distance over the
stored (re-normalized) embeddings, hand-rolled agglomerative average
linkage cut at `distance_threshold` (no scipy dependency; appearances per
window are tens, so O(n³) is fine). Deterministic: ties keep the first
pair in scan order, and clusters are ordered by their smallest member
index for stable cluster ids.

**Anchoring.** Pure module `anchoring.py`. A face claims an appearance at
intersection / face-box-area ≥ `face_overlap` (best match wins, earlier
face on ties); a claim whose Immich person is set makes the appearance
*labeled* — never proposed, and an empty person name contributes no
anchor name. Unanimous named anchors (any count, one name) propose for
every unlabeled member; ≥ 2 distinct names block the whole cluster as
conflicted (labeled members still read labeled); anchor-less clusters
reach the ask-for-name queue only at `min_cluster_size` appearances.
Singletons and below-minimum anchor-less clusters get no proposal rows at
all. Proposal rows carry `(asset, appearance, window, cluster, status,
proposed_name)` with statuses `proposed` / `ask_name` / `conflicted` /
`labeled`, replaced wholesale each run; `reid-wipe` clears them too.

**Run output.** Each run echoes one summary line per window:
`window 2026-11-10: appearances=18 clusters=2 anchored=50.0%
conflicted=0.0% anchor_less=50.0% singletons=8 proposals=4
ask_for_name=5` (percentages over multi-appearance clusters). The stub
Immich now serves `GET /api/faces?id=<assetId>` so acceptance tests can
serve face labels; anchoring tests use `distance_threshold = 0.01`
because identical embeddings sit at float32-epsilon distance (~1e-7)
rather than exactly 0, and the nearest genuinely different fixture pair
sits at 0.295. Knobs live in the new `[clustering]` config section
(`window_days`, `distance_threshold`, `min_cluster_size`,
`face_overlap`).
