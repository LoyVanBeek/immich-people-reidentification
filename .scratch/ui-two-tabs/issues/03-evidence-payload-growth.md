# Evidence API payload growth for the new card

Part of: [UI Two Tabs — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: 01

## Question

What exact fields do `/api/evidence` (and the card payloads inside `/api/write-review`)
gain so the new evidence card can render without extra round-trips? Settle:

1. **Sibling appearances**: per card, the asset's other *stored* appearances — appearance
   index, person box (the existing `box_*` columns), confidence — plus which one is the
   card's own appearance, so the full-photo overlay can draw all outlines with the current
   one highlighted. Field name/shape proposal to confirm.
2. **You-row overrides**: explicit fields for pin, ignored, rejected (with the rejected
   name), and cluster dismissal, so the "You:" row renders from data, not derived chips.
3. **Immich URL**: either a per-card `immich_url` or a single `immich_url` base in
   `/api/state` (already exists) — decide which, given the card must be clickable and show
   the URL as text.
4. Confirm no other endpoints change shape in this ticket (write-review reshape is ticket 04).

## Answer

**Payload contract for the new evidence card** (grilled 2026-11-25; applies to
`/api/evidence` cards and to the cards inside the write-review/Final-check payload, which
reuse `_build_evidence_for_cluster` — so they inherit the same fields):

1. **Sibling outlines via JSON** (option (a) chosen): each card gains
   - `siblings: [{appearance: int, box: [l, t, r, b], confidence: float}]` — **all** stored
     appearances of the asset, including the card's own (self = the entry whose
     `appearance` equals the card's). `box` is in pixel coordinates of the configured
     pixel source (same space detection ran in).
   - `sibling_count: int` (= `len(siblings) - 1`) for the "N others in this photo" badge.
   - Full photo image: new `/api/crop?full=1&asset_id=…` mode (no `appearance` needed) returns
     the **entire pixel-source image** as JPEG q85 at **native resolution** — no box drawing
     server-side. The UI draws an SVG overlay with `viewBox` set to the loaded image's
     natural size, so the pixel boxes map 1:1; self box highlighted (accent), others
     contrasting. Fetched lazily only when the card is expanded (inline expand, per the
     ticket 02 verdict).
2. **You-row overrides**: card gains `rejected_name: str | None` and `dismissed: bool`
   (per-card flag that the cluster is dismissed, in addition to the cluster-level "skipped"
   status + reopen). `pin` and `ignored` already exist. You: row renders
   `pin ?? ignored ?? rejected_name`.
3. **Immich URL**: unchanged — single `immich_url` base in `/api/state`; the card builds
   `${immich_url}/photos/${asset_id}` client-side for the clickable crop and the URL text.
   No per-card URL field.
4. **Scope**: no other endpoint changes shape in this ticket. Implementation notes:
   `EvidenceCard` is a frozen dataclass — add the new fields with defaults; boxes come from
   `store.appearances_by_asset()` which `get_evidence` already loads.
