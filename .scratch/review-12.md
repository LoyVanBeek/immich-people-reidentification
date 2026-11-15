# Review: Ticket #12 Confirmation UI Read Path

## (a) Missing / Partial Requirements

| Spec line | Finding |
|-----------|---------|
| L12: "Immich face label" in evidence cards | **Missing.** `server.py:_build_evidence_for_cluster` returns `immich_label=None` with comment "would need to re-fetch faces to get this". Prototype shows `immich: 'Alice'` on labeled appearances. |
| L17: "tuning knobs that exist so far (timespan window, cluster distance threshold, detector confidence floor, minimum box height)" | **Partial.** Knobs exposed but `face_overlap` (face→person containment) is included in UI/config despite not being listed in spec. |
| L16: "Keyboard navigation works: `j`/`k` move, `c` confirm, `s` skip" | **Partial.** Keys wired but `c`/`s` trigger `doConfirm`/`doSkip` which mutate local state — spec says "read-only — no confirmation or writing yet". |
| L13: "Include/exclude checkboxes per asset over writable appearances" | **Partial.** Checkboxes render but changes aren't persisted (TODO comment in `wireEvidenceEvents`). |

## (b) Scope Creep (Behavior Not Asked For)

| Spec line | Finding |
|-----------|---------|
| L5: "Read-only — no confirmation or writing yet" | **Creep.** UI implements full confirm/skip flow: `doConfirm` marks clusters `confirmed`, records dry-run batches, shows "Promote to real write" buttons, `doSkip` marks `skipped`. Prototype variant B/C had these but spec explicitly deferred writes to later tickets. |
| L5: "Read-only" | **Creep.** "Name this cluster" button + modal prompts for name, updates `proposed_name` locally. Spec defers naming to write path. |
| L5: "Read-only" | **Creep.** Dry-run toggle in top bar, dry-run batch sidebar section, "Record (dry-run)" button labels. Spec doesn't mention dry-run UI. |
| — | **Creep.** `/api/proposals` endpoint returns raw proposals — not in spec's API surface. |
| — | **Creep.** `face_overlap` knob exposed in config popover; not in spec's knob list. |

## (c) Implementation Looks Wrong

| Spec line | Finding |
|-----------|---------|
| L12: status chips `will write` / `already labeled` / `conflict — skip` / `no head — skip` | **Wrong mapping.** `server.py` maps `ask_name` → `no_head_skip` with TODO "real no-head detection in #14". But `ask_name` means "no anchor found, ask for name" (prototype: cluster C-03), while `no_head` means detection failed to find a head box (prototype: `IMG_4502.jpg` with `mode: 'no-head'`). These are distinct. |
| L12: "per-appearance status chip" | **Wrong.** `immich_label` always `None`; prototype shows "immich: Alice" on already-labeled appearances. Backend would need to join face labels. |
| L15: "stats pane shows latest per-run window summary" | **Questionable.** `clusters` counts only multi-appearance clusters (size ≥ 2); `singletons` counts size-1 clusters. Spec doesn't define this distinction; prototype doesn't show stats pane. |
| L18: "workspace served by container over small HTTP API" | **Risk.** `serve()` uses stdlib `HTTPServer` (single-threaded, no TLS, no graceful shutdown). Fine for prototype but not "container-served" production path. |
| L19: "Tests assert user-visible UI/API states against prototype as behavioral reference" | **Gap.** Tests verify API shapes and static file serving but don't assert evidence-card fields (crop, head-box overlay, date, confidence, immich label, status chip) match prototype's fake data schema. |