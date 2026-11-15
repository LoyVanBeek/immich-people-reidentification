# GLOSSARY.md

Person re-identification for Immich: a Pi 5 container that detects persons in new photos, embeds them, clusters the embeddings within a timespan window, anchors clusters with Immich's face labels, and proposes name assignments through a confirmation UI.

## Language

**Checkpoint**:
The timestamp marking how far the new-asset feed has been consumed: the newest asset createdAt from the last fully completed batch run. Persists in the local store; advances only after the whole feed is read, so a crashed run resumes instead of reprocessing.
_Avoid_: cursor (that's Immich's search pagination token), bookmark

**New-asset feed**:
The stream of photos not yet processed: Immich's metadata search filtered to `createdAfter` = the checkpoint. Photos only (videos are out of scope in v1).
_Avoid_: sync stream (that's Immich's session-token-only sync API), asset list

**Person appearance**:
One detected person instance in one photo. The unit that receives an embedding; clustering operates on person-appearances, not photos. A photo with three detected persons contributes three appearances.
_Avoid_: detection, person (ambiguous between the human and the instance)

**Full-person crop**:
The body-region pixels cropped from the photo for one person appearance; the input to the embedding model. Distinct from Immich's face box, which covers only the face and serves anchoring.
_Avoid_: face crop (that means Immich's face detection region)

**Quality gate**:
One ordered filter between raw person detections and stored appearances, configured in `[detection]`: the confidence floor drops weak detections, the minimum box height drops barely-visible ones. Edge-clipped boxes are deliberately kept (a clipped person is still an appearance). Each gate's drop count is reported per run.
_Avoid_: threshold (there are two gates, and only one is a threshold), filter step

**Cluster**:
One person's appearance-group within one timespan window, formed by clustering embeddings. Not a persistent identity: the same human may legitimately form multiple clusters in a window, and clusters are never merged across windows.
_Avoid_: identity, person

**Timespan window**:
The calendar-day span (configurable, default: the same calendar day) whose appearances cluster together. Window membership follows the asset capture time (EXIF) in the local timezone, falling back to the upload time; appearances with no usable capture time stay unproposed until the store is rebuilt. Rebuilt from scratch every run.
_Avoid_: day (ambiguous about the boundary rule), session

**Singleton**:
A person-appearance that joined no cluster. Stored, but never surfaces in proposals or the ask-for-name queue.
_Avoid_: noise (that's a judgment, not a state)

**Proposal**:
The name assignment a run suggests for one appearance, derived from its cluster's unanimous named anchors, persisted locally with a per-appearance status (proposed, ask-for-name, conflicted, already labeled). Nothing reaches Immich without human confirmation.
_Avoid_: suggestion, write (that's the confirmed Immich side effect)

**Anchor**:
A *named* Immich face label attached to an appearance in a cluster, used as ground truth to propose a name. Anchors within a cluster must be unanimous; disagreement blocks the cluster. Every cluster is predicted to contain at least one anchor; anchor-less clusters fall back to asking the user for a name.
_Avoid_: ground truth label, seed

**Head box**:
The head-region bounding box that represents where a person appearance sits in an asset, written to Immich as the face position. Sourced from Immich's own face detections when one lies within the appearance's full-person box; otherwise produced by the pipeline's head detector.
_Avoid_: face box (reserved for Immich's own detection data), person box

**Dry-run**:
A run mode that produces proposals without writing anything to Immich. All Immich side effects are deferred until promotion.
_Avoid_: preview mode, simulation

**Promotion**:
Converting a dry-run proposal into real Immich writes by replaying its recorded actions.
_Avoid_: apply, commit

**Conflict**:
An asset where an overlapping Immich face label names a different person than the cluster's proposal. The write is skipped and the asset is flagged, never overwritten.
_Avoid_: collision, mismatch

**Write ledger**:
The local record of asset–person label assignments already written to Immich. Drives idempotency and holds tombstones.
_Avoid_: write log, journal

**Tombstone**:
A write-ledger entry marking an asset–person pair the user removed in Immich directly. Such pairs are never re-proposed.
_Avoid_: deletion record
