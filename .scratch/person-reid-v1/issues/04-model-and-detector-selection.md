# Model and detector selection

Part of: [Person Re-ID v1 — Map](../map.md)
Type: grilling
Status: resolved
Blocked by: [Re-ID model on ARM CPU](03-reid-model-arm-cpu.md)

## Question

Decide, with the research facts on the table:

1. Which Re-ID embedding model the container runs (OSNet variant vs alternatives), given Pi 5 CPU constraints.
2. Which person detector produces the full-person crops feeding it (nano YOLO-class or otherwise), its input resolution, and its throughput budget.
3. How crops failing quality checks (too small, clipped, low confidence) are handled.

## Comments

- Round 1: Q1–Q4 accepted.
  - **Q1 Embedding model**: `osnet_ain_x0.25` — AIN variant for cross-domain generalization at x0.25 cost (ticket 03's throughput estimate carries); best available non-Market weights (MSMT/combined); OSNet-standard 256×128 input → 512-dim embedding.
  - **Q2 Code source**: vendor `torchreid/models/osnet.py` + weights into the image; no BoxMOT (AGPL not needed).
  - **Q3 Detector**: YOLO11n (Ultralytics, COCO-pretrained), person class only, 640×640 letterboxed input; own detection on **every** asset — Immich face boxes are anchoring material (issue 05), not detection input; background-person misses at 640 acceptable for v1, tuning knob later.
  - **Q4 Source images**: Immich preview thumbnails (~1440px) as pixel source for detection and cropping in v1; originals only as fallback knob. No preview↔original coordinate mapping needed for v1.

Q5 (throughput budget) and Q6 (crop quality policy) still open.
- Round 1 (cont.): Q5–Q6 accepted.
  - **Q5 Throughput budget**: relaxed — a nightly run over one day's new photos must complete in **under 30 minutes** on the Pi 5 (runs at night, wall time not precious). 3× looser than the agent's 10-min proposal; materially increases compute headroom.
  - **Q6 Crop quality policy**: three gates after detection, before embedding: (1) detector confidence floor ~0.35 → drop; (2) person-box height < ~100px in source image → drop (upscaling a sliver would embed interpolation artifacts); (3) edge-clipped persons → **keep** (clothes/pose signal still useful). Rejected crops are never embedded; each run logs drop counts per gate for dry-run visibility. Exact thresholds are tuning knobs, deferred per map fog note.
- Fact-check (subagent, primary sources): **the torchreid model zoo has NO ReID-trained checkpoint for `osnet_ain_x0.25`** — only ImageNet-pretrained weights (for training init). ReID-trained checkpoints exist for `osnet_x0_25` (market1501 91.2 R1 / duke 82.0 / msmt17 61.4) and for `osnet_ain_x1_0` in the cross-domain tables (e.g. MSMT-combineall→market1501 70.1 R1). License confirmed **MIT** (vendoring Q2 stands). Input 256×128, feature dim 512 confirmed for all OSNet variants. - Round 3: Q1 revisited (as Q7) and Q8 asked; both accepted.
  - **Q7 Embedding model (revised)**: `osnet_ain_x1_0` with **combineall** weights — the only AIN variant with ReID training in the zoo (e.g. MSMT-combineall→Market1501 70.1 R1); AIN + multi-dataset training is the domain-generalization setup family photos need. ~10× x0.25 FLOPs is irrelevant: detector is the bottleneck (7–15 img/s) and budget is 30 min. 256×128 input, 512-dim embedding (confirmed variant-independent). Fallback option B was `osnet_x0_25`+MSMT17 weights, not taken.
  - **Q8 Granularity**: confirmed — one embedding per detected person crop; a photo with N detected persons contributes N embeddings; clustering operates on person-appearances, not photos. Same person twice in one photo → two embeddings, likely same cluster (harmless).

## Answer

**1. Embedding model** — `osnet_ain_x1_0` with **combineall** weights from torchreid's model zoo (the only published AIN checkpoint with ReID training; e.g. MSMT-combineall→Market1501 70.1 Rank-1). Input 256×128 → 512-dim embedding (variant-independent, verified). Chosen over the fallback (`osnet_x0_25` + MSMT17 weights) because AIN + multi-dataset combineall training is the right domain-generalization setup for family photos, and with the 30-min budget the detector — not the embedder — is the throughput bottleneck. Code vendored from torchreid (`osnet_ain.py` + weights, MIT) into the image; no BoxMOT. Recorded as [ADR 0001](../../../docs/adr/0001-osnet-ain-x1-over-nano-osnet.md).

**2. Person detector** — YOLO11n (Ultralytics, COCO-pretrained), person class only, 640×640 letterboxed input, run on **every** asset. Immich face boxes are anchoring material (issue 05), not detection input; reusing Immich detections is only a backfill optimization, out of scope for v1. Pixel source: Immich **preview thumbnails** (~1440px); originals only as a fallback knob. Missed far-background persons at 640 are acceptable for v1 (tuning knob later).

**3. Throughput budget** — one day's new photos processed in **< 30 min** on the Pi 5. (Ticket 03's estimate: ~10–30 s for a 50-photo batch; headroom absorbs detector surprises.)

**4. Crop quality gates** (post-detection, pre-embedding):
- Detector confidence below ~0.35 → drop (tuning knob).
- Person-box height below ~100px in the source image → drop — upscaling a sliver to 256×128 would embed interpolation artifacts, not a person (tuning knob).
- Edge-clipped persons → **keep**; clothes/pose signal still useful.
- Rejected crops are never embedded; each run logs drop counts per gate for dry-run visibility.

**5. Granularity** — one embedding per **person-appearance**: a photo with N detected persons contributes N embeddings; clustering operates on person-appearances, not photos. Same person twice in one photo → two embeddings, likely same cluster (harmless).

Glossary terms recorded in `GLOSSARY.md`: *Person appearance*, *Full-person crop*, *Anchor*.
