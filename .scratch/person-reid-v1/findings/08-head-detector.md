# Findings: head/face detector options for person crops on Pi 5 (CPU-only, ONNX Runtime)

Part of: [Person Re-ID v1 — Map](../map.md)
Type: research
Date: 2026-11-14

## Bottom line

**Recommended: SCRFD-500M (`det_500m.onnx`, insightface) run per-person-crop at 320×320 letterbox input via the pipeline's existing ONNX Runtime, with a geometric face-box→head-box expansion.** Reasons:

1. **ONNX-native, zero new runtime deps** — official ONNX exports exist (insightface `tools/scrfd2onnx.py`; ready-made mirrors on HuggingFace: `RuteNL/SCRFD-face-detection-ONNX`, `globalnebula/insightface-buffalo-l-onnx`). Postprocessing is plain NumPy (insightface ships a reference ONNX Runtime implementation in `python-package/insightface/model_zoo/scrfd.py`). Fits the repo's "everything through onnxruntime, vendor weights into the image" pattern exactly (issue 02/03).
2. **Cheapest meaningful compute.** At 640 input, SCRFD-0.5GF is 779 MFLOPs — ~11× lighter than the pipeline's nano-YOLO person detector (~8.7 GFLOPs, measured 7–15 img/s on Pi 5). On person **crops** at 320 input it is essentially free (see throughput below).
3. **Box-style consistency with Immich.** Immich's own face detection is SCRFD-family (antelopev2/buffalo packs ship `det_10g.onnx`), so the boxes we write back via `POST /faces` match the geometry Immich users already see. *(Medium-high confidence: Immich's facial-recognition default model is antelopev2, whose detector component is SCRFD-10GF.)*
4. **Accuracy headroom if needed.** The same ONNX path scales to SCRFD-2.5G/10G without code changes — only the weights file changes.

**License caveat (important, not blocking for this project):** insightface's *code* is MIT, but its model-zoo *weights* are published for "non-commercial research purposes only". For a private single-user project this is acceptable; if this is ever open-sourced/shipped, switch weights to **YuNet** (see below), the clean-license fallback: MIT code + Apache-2.0 ONNX model in OpenCV Zoo, similar latency class.

**Alternative worth real consideration — one-pass pose head (Option C):** swap YOLO11n person detector for YOLO11n-pose (COCO keypoints). It still emits person boxes (replacing the current detector 1:1) *and* nose/eyes/ears keypoints → head box is pure geometry, no second model pass at all. Cost is ~1.1× the current detector (yolov8n-pose 9.0 vs yolov8n 8.7 GFLOPs at 640 — near-zero marginal). Weakness: head box exists only when ≥1 facial keypoint is visible (fails for back-of-head/strong profile — exactly the tolerance the requirement asks about), and landmark-derived boxes need generous inflation (~1.5–2× ear-span, extended upward) to cover skull/hair. Best used as *fallback geometry* for crops where the face detector finds nothing, rather than as the primary source.

**Immich-detections correction (verified):** the premise that `GET /faces?id=<assetId>` only returns faces associated with people is **wrong for current Immich (v2/v3)**. The endpoint "Retrieve[s] all faces belonging to an asset" — unassigned detections included, returned with `person: null`. (Evidence: api.immich.app `getFaces` spec; the v3 migration blog explicitly says the old `unassignedFaces` asset-response property was removed and "use GET /faces to retrieve asset face information"; community tooling since 2024 filters `face['person']` null out of this exact endpoint; the DB table backing it is `asset_faces` with nullable `personId`.) So Immich's raw face detections **are** reachable without person assignment. This does not remove the need for our own detector: Immich detections only exist where Immich's SCRFD fired (no back-of-head, no guarantee the face-detection job ran on the asset), they are face- not head-tight, and they give no person↔face linkage for unassigned faces — we'd have to re-associate by IoU against our person boxes anyway. Treat `GET /faces` as a **free supplement/cross-check**, not the primary head source. (This partially contradicts the map's issue-01 summary phrasing; the endpoint itself was recorded correctly there.)

**Throughput impact on the nightly batch: negligible either way.** ~50 photos → ~150 person crops (assume ~3 persons/photo). Head stage on crops at ~25–100 crop/s adds roughly **2–6 s**. Even the most expensive option (full-frame 640 face detection per photo) adds ~4–7 s. The 30-min budget (issue 04 Q5) is untouched; the batch stays in the ~15–40 s range established by ticket 03/04.

## Calibration anchor used for all Pi 5 estimates

The repo's own measured number — nano-YOLO at 640 → **7–15 img/s** on Pi 5 (issue 03) — is the only on-hardware anchor. Cross-checked against primary benchmarks:

- YuNet paper (Machine Intelligence Research 20(5), 2023, [PDF](https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf)), ONNX Runtime CPU on i7-12700K: SCRFD-0.5g 17.8 ms @640, SCRFD-10g 95 ms @640, YuNet 11.3 ms @640; at 320×320: SCRFD-0.5g 3.4 ms, YuNet 2.2 ms. Per-FLOP efficiency is comparable across these detectors (SCRFD-0.5g 779 MFLOPs@640 ≈ YOLO5Face-n 741 MFLOPs@640, and their measured latencies are ~equal), so scaling the repo's nano-YOLO anchor by FLOPs is defensible.
- Pi 5 measured datapoint (Swallow Incubate blog, OpenCV 4.10 on Cortex-A76): YuNet-class face detection + SFace recognition at 640×480 → 74 ms total (13.5 fps); detection alone at 1280×720 → 204 ms. Confirms the "small face detector ≈ tens of ms per frame on Pi 5" order of magnitude.

All Pi 5 numbers below are **estimates unless marked measured**, with confidence stated. FLOPs→img/s scaling assumes ~60–130 effective GFLOPS (from the 7–15 img/s nano-YOLO anchor) with a pre/post-processing floor of ~15–40 ms/image in Python.

## Options compared

### A. SCRFD (insightface) — **recommended**

| | |
|---|---|
| Sizes | 500M / 1G / 2.5G / 10G / 34G (GFLOPs at VGA 640×480). `det_500m.onnx` 0.57 M params; `det_2.5g` 0.67 M; `det_10g` 3.86 M params |
| License | Code MIT; **weights: insightface model zoo = non-commercial research only** (buffalo_l mirrors on HF inherit this). Clean-license alternative: see YuNet (G) |
| ONNX | **Yes, official** (`tools/scrfd2onnx.py`, dynamic input supported); community mirrors with SHA-256s exist |
| Input | 640×640 default; 320×320 works and divides cost by ~4 |
| Output | Face box + score (+ 5 landmarks in `_KPS`/`bnkps` variants) |
| Head-quality box | **Face box only** (roughly square, includes some forehead — same style as Immich's own). No back-of-head, weak on profile past ~90°. Head box derived by expansion: extend top by ~0.5–0.6× face height, widen ×1.6–2.0, square-ify |
| Pi 5 throughput (est., low confidence) | ~25–80 crop/s @640; **~50–120 crop/s @320** for 500M. 2.5G ≈ 5× that cost. *Derived: FLOP-scaled from the 7–15 img/s nano-YOLO anchor; FLOPs ratio 8.7/0.78 @640.* |
| Notes | `bnkps` variants have lower false-positive rate; landmarks struggle past ~90° rotation (upstream note) |

### B. YOLOv8n-face / YOLO11n-face (WIDERFace-finetuned YOLO)

- Sources: `lindevs/yolov8-face` (repo MIT, ONNX shipped: `yolov8n-face-lindevs.onnx` 11.7 MB, WIDERFace 93.8/91.8/**79.4** Easy/Med/Hard @640); `derronqi/yolov8-face` (GPL-3.0, yolov8n 94.6/92.3/79.6, 12 MB ONNX via `yakhyo/yolov8-face-onnx-inference`); `akanametov/yolo-face` (yolov11n-face.pt, ONNX export documented).
- License: messy — YOLOv8 base is Ultralytics AGPL-3.0; derronqi is GPL-3.0. Fine for private use, worse than SCRFD/YuNet for reuse.
- Input: 640×640 fixed (dynamic export available). Output: face box + 5 landmarks.
- Pi 5 throughput (est., medium confidence): ~7–15 img/s @640 (≈ the person detector, 8.1 vs 8.7 GFLOPs — same class); ~4× that @320. On crops at 640 the cost is per-crop, so a 3-person photo pays 3×.
- Head-quality box: same as SCRFD (face box + landmarks → expand). Slightly better on WIDERFace-hard than SCRFD-500M numbers suggest, but the comparison sets differ.
- Verdict: viable, but strictly heavier than A for the same output semantics; pick only if we want one YOLO toolchain for everything.

### C. YOLOv8n-pose / YOLO11n-pose head keypoints (replace the person detector)

- COCO-17 keypoints; head = indices 0–4 (nose, 2×eyes, 2×ears), confirmed in Ultralytics docs/COCO spec. Pose models also emit person boxes, so one model replaces the current detector.
- ONNX: official Ultralytics export (yolov8n-pose ≈ 9.0 GFLOPs @640 — ~1.04× the current detector).
- License: AGPL-3.0 (Ultralytics).
- Pi 5 throughput (est., medium confidence): ~6–14 img/s @640, **replacing** (not adding to) the person detector pass — net pipeline cost ≈ zero.
- Head-quality box: **the weak point.** No keypoints when the face is not visible (back-of-head, deep profile, heavy occlusion) → no head box at all unless we fall back to "top ~25% of person box" heuristics. Landmark-span boxes also under-cover skull/hair and need inflation. Per-keypoint confidence flags make the "no face visible" case explicitly detectable, which is good for logging.
- Verdict: cheapest wall-clock option and simplest architecture, but worst tolerance to exactly the hard cases named in the requirement. Use as fallback geometry under A, or as the whole detector only if v1 accepts "no face visible → no head box".

### D. MediaPipe BlazeFace (short-range) — **not recommended**

- Model: `blaze_face_short_range.tflite`, 128×128 input, float16, Apache-2.0. Measured 2.94 ms CPU on Pixel 6 — trivially fast; on Pi 5 est. ~5–15 ms/frame (low confidence).
- Problems: ships as **TFLite, not ONNX** (tflite→onnx conversion friction, and the Tasks runtime is a separate non-ONNX stack); the Python `mediapipe` wheel has no official aarch64/Python 3.13 manylinux build (Pi users go through piwheels/source builds) — breaks the repo's onnxruntime-only constraint (issue 02). Short-range variant is tuned for selfie-distance faces filling much of the frame; degraded for small faces in person crops (full-range variant is sparser and also TFLite).
- Head-quality box: face box + 6 keypoints; same expansion caveat.
- Verdict: reject on integration friction, not speed.

### E. RetinaFace (MobileNet0.25) — viable, dominated by A

- ONNX available (community conversions; e.g. memryx pipeline notes, `yuuMQ/Retinaface_WiderFaceDataset`-derived exports). ~0.44 M params, 0.8 GFLOPs @VGA. License: code MIT (dbsuper/retinaface-family), original weights from insightface ecosystem → same research-only caveat as A in practice.
- Pi 5 throughput (est., low confidence): ~15–40 crop/s @640 — the YuNet paper's i7 numbers (RetinaFace-0.25 22.0 ms @640 vs SCRFD-0.5g 17.8 ms) plus insightface-issue datapoints (RetinaFace 22.19 ms vs SCRFD-0.5GF 3.6 ms on RTX3080 CPU) show it ~1.2–6× slower than SCRFD-0.5G at equal-or-worse WIDERFace-hard (47–68 vs 68–77).
- Head-quality box: face box + 5 landmarks; historically the strongest on hard/profile faces of its generation, but SCRFD surpasses it now.
- Verdict: no win over A.

### F. CenterFace — viable, dominated by A

- ONNX available (original repo `StarDu/CenterFace` ships `onnx_models/centerface.onnx`, ~7.6 MB... fp32; dynamic-free 640×480 input). Code MIT; weights free-for-research per README (weaker than MIT, medium confidence on exact terms).
- Pi 5 throughput (est., low confidence): ~20–60 crop/s @640×480 — comparable to SCRFD-0.5G class.
- Head-quality box: face box + 5 landmarks. Effectively unmaintained since ~2021.
- Verdict: fine but no advantage; maintenance risk.

### G. YuNet (ShiqiYu/libfacedetection, OpenCV Zoo) — **clean-license fallback**

| | |
|---|---|
| Size | 75,856 params (~1.1 MB-class model, the smallest in the YuNet paper's comparison — 8.3× fewer than SCRFD-0.5g); 595 MFLOPs @640×640, 149 MFLOPs @320 |
| License | Code **MIT** (libfacedetection); ONNX model `face_detection_yunet_2023mar.onnx` in OpenCV Zoo **Apache-2.0** — the cleanest license chain of all options |
| ONNX | **Yes** (opencv_zoo ships it; OpenCV DNN and ONNX Runtime both run it) |
| Input | 320 or 640, dynamic OK |
| Output | Face box + 5 landmarks + score |
| Head-quality box | Face box → expand (same as A). Landmark quality good |
| Pi 5 throughput | **Partially measured on Pi 5**: detection+SFace recognition 74 ms @640×480 on Cortex-A76 (OpenCV DNN). Detection-only est. ~20–50 ms → ~20–50 img/s @640, ~60–150 @320 (medium-low confidence; the 74 ms includes recognition) |
| Accuracy | YuNet paper: 0.899/0.869/0.691 @640 — between SCRFD-500M (0.907/0.882/0.684) and SCRFD-2.5G; excellent for its size |

- Verdict: **equal-first on technical merit, first on license.** If the insightface weights' research-only clause ever matters, this is the drop-in replacement.

### H. Reuse Immich's own face detections (`GET /faces?id=<assetId>`) — supplement only

- **Verified:** returns **all** face detections on an asset, including unassigned (`person: null`), with pixel boxes (`boundingBoxX1..Y2`, `imageWidth/Height`, `sourceType`). Permission `face.read`. No separate "raw detections" endpoint needed — this *is* it. (Also: `POST /faces` exists for writing manual faces — our write-back path is documented and stable.)
- What it can't do: guarantee a detection for every person box we produce (Immich's SCRFD may miss what our YOLO found, and vice versa); it says nothing about back-of-head; and it only exists if Immich's face-detection job ran for that asset.
- Best use: (a) cross-check/second opinion at zero compute cost, (b) a source of pre-labeled anchor boxes for issue 05 (already the map's plan — anchoring uses Immich face labels as ground truth), (c) reduce head-detector invocations where an Immich detection already covers the person box (IoU gate) — an optimization, not a dependency.

## Design consequences for the pipeline

1. **Two-stage, crop-local:** run the head detector on each person crop (letterboxed to 320×320), not on the full frame. Keeps person↔head association free (assignment problem vanishes) and cuts FLOPs ~4× vs full-frame 640.
2. **Face size floor:** with the existing ≥100 px person-height gate, a face can be ~12–15 px tall — below reliable detection for any model here. Either (a) accept "small person → no head box" as a logged outcome (consistent with the quality-gate philosophy of issue 04), or (b) upscale crops <200 px person height ×2 before the head pass (cheap at 320 input). Decide as a tuning knob, same pattern as issue 04 Q6.
3. **Box convention:** expand face box → head box (top-extension ~0.5–0.6× face height, width ×~1.6–2.0, then square-ify); calibrate against Immich's own SCRFD boxes on a few known assets rather than trusting the constants.
4. **Fallback ladder for a person crop:** SCRFD hit → expanded head box; SCRFD miss but pose keypoints visible (only if option C adopted) → landmark-derived box; neither → log "no head box" and skip the Immich face write for that appearance (embedding still stands on the person crop).

## Source list (primary)

- insightface SCRFD README + model zoo (models, ONNX conversion, CPU timings): github.com/deepinsight/insightface `detection/scrfd`, `model_zoo` — incl. the 28.3 ms/640×480 single-thread x64 number for SCRFD-0.5GF and the "3.6 ms is GPU/optimized, CPU timings differ" issue thread (#1761).
- SCRFD ONNX mirrors: huggingface.co/RuteNL/SCRFD-face-detection-ONNX, huggingface.co/globalnebula/insightface-buffalo-l-onnx.
- lindevs/yolov8-face (MIT repo, ONNX files + WIDERFace evals); derronqi/yolov8-face (GPL-3.0); yakhyo/yolov8-face-onnx-inference (ONNX Runtime reference incl. CPU note); akanametov/yolo-face (yolo11n-face).
- YuNet paper table (ONNX Runtime CPU cross-detector latency+FLOPs): link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf; Pi 5 OpenCV measurements: tech.swallow-incubate.com (Cortex-A76 OpenCV benchmark, 2026).
- MediaPipe Face Detector docs (model card, 128×128, TFLite, Pixel-6 latencies): developers.google.com/mediapipe.
- CenterFace ONNX: github.com/StarDu/CenterFace.
- Immich API: api.immich.app `getFaces` / `AssetFaceResponseDto` / `AssetFaceWithoutPersonResponseDto`; v3 migration blog (unassignedFaces → GET /faces); immich issue #10277 + discussion #10534 (unassigned faces via `GET /faces?id=`, `asset_faces.personId IS NULL`); immich `face.controller.ts` (endpoint semantics).
- Pi 5 face-detection reference point for SCRFD ONNX on Pi (ModelNova card, ONNX/CPUExecutionProvider notes, 82.91 ms @640 — Pi 4/5 card, treat as upper bound / medium-low confidence).
