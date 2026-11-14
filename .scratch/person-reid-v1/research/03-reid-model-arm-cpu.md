# Research: Re-ID model on ARM CPU (Raspberry Pi 5)

Ticket: [.scratch/person-reid-v1/issues/03-reid-model-arm-cpu.md](../issues/03-reid-model-arm-cpu.md)
Date: 2026-11-14
Status: complete

> **Note on method.** The `research` skill asks for a background subagent; since this
> document was produced by that subagent directly, the investigation was done inline.
> Where no published Pi 5 benchmark exists for a specific model, numbers are marked
> **[estimate]** and derived from measured Pi 5 benchmarks plus model FLOPs, as the
> ticket requested. Confidence is stated per claim.

## Primary sources used

| # | Source | What it establishes |
|---|--------|---------------------|
| S1 | Torchreid docs — https://kaiyangzhou.github.io/deep-person-reid/ (v1.4.0) | Official install procedure (conda-based), FeatureExtractor API, project status |
| S2 | Torchreid Model Zoo — https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html | Per-model params, GFLOPs, accuracy, weights hosted on Google Drive |
| S3 | `torchreid/models/osnet.py` (upstream master) — https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py | Weights auto-download via `gdown` from Google Drive at first use |
| S4 | Upstream `requirements.txt` — https://github.com/KaiyangZhou/deep-person-reid/blob/master/requirements.txt | Actual dependency list incl. `tb-nightly`, `isort==4.3.21`, `gdown` |
| S5 | PyPI `torchreid` 0.2.5 — https://pypi.org/project/torchreid/ | Third-party repack, sdist-only, last released Oct 2022 |
| S6 | `torchreid-pip` repo — https://github.com/goksenin-uav/torchreid-pip | setup.py / requirements of the PyPI repack |
| S7 | PyPI JSON API for `torch` (queried 2026-11-14) | aarch64 manylinux wheels exist for torch 2.12.0–2.15.0 (cp310–cp313) |
| S8 | Ultralytics Raspberry Pi guide — https://docs.ultralytics.com/guides/raspberry-pi/ (updated ~Nov 2026, benchmarks w/ Ultralytics 8.4.108) | Measured Pi 5 inference times for nano-detectors across 8 runtimes; Pi 5 CPU specs (BCM2712, Cortex-A76 ×4 @ 2.4 GHz) |
| S9 | PyPI JSON API for `onnxruntime` (1.32.0) and `ncnn` (1.0.20261020) | aarch64 manylinux wheels available for both |
| S10 | BoxMOT — https://github.com/mikel-brostrom/boxmot (master, Nov 2026) | Maintained alternative packaging of OSNet/HACNN/MobileNetV2 Re-ID backbones; AGPL-3.0 license |

---

## 1. Installation path: torchreid + PyTorch on ARM64 under `uv`

### 1.1 PyTorch itself: works out of the box (high confidence)

PyPI publishes official `manylinux_2_28_aarch64` CPU wheels for `torch` for every
recent release — 2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1 and 2.15.0 (latest),
covering CPython 3.10–3.13 (S7). `torchvision` follows the same release matrix.
So on Raspberry Pi OS 64-bit / Debian arm64, plain:

```bash
uv add torch torchvision
```

resolves and installs without any extra index URL. The aarch64 PyPI wheels are
CPU-only builds, which is exactly what we want.

**Caution (medium-high confidence):** on ARM the classic failure mode is a wrong
torch binary ("Illegal instruction" on import) when a CUDA-flavoured or
mismatched-arch wheel sneaks in. This is a recurring support theme in the
Ultralytics Raspberry Pi guide comments (S8), including a Nov 2026 report with
`torch-2.14.1+cu130 CPU (aarch64)` that the Ultralytics team resolved by
force-reinstalling the CPU wheels from the official CPU index. On aarch64 the
PyPI default *is* the CPU build, so the risk mostly bites if someone copies an
x86 `--index-url https://download.pytorch.org/whl/cu...` line into the
Dockerfile. If pinning for Immich-version parity (ticket 02) ever forces the
PyTorch CPU index, use `https://download.pytorch.org/whl/cpu` — it serves aarch64
wheels too.

### 1.2 torchreid: the messy part (high confidence on the facts)

Upstream has **no official PyPI package and no `pyproject.toml`** (S3/S4 — verified
404 for pyproject). The documented install (S1) is conda-era:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
pip install -r requirements.txt   # see S4
python setup.py develop
```

The upstream `requirements.txt` (S4) drags in `tb-nightly` (a TensorFlow *nightly*
build — large, frequently broken for fresh Python versions), an ancient
`isort==4.3.21` pin, `gdown`, `yacs`, etc. The repo was last touched substantively
in 2021–2022; docs version is 1.4.0.

The PyPI package named `torchreid` (S5) is **not** upstream: it is a third-party
repack (`kadirnar` / goksenin-uav/torchreid-pip, S6), published as an **sdist only**
(92.7 kB, Oct 2022, no wheels, no provenance) with the same bloated requirements
including `tb-nightly`. Under `uv` it will build from source and pull the full
legacy dependency tree; `tb-nightly` is the part most likely to fail or bloat an
ARM64 image.

**Practical paths, in order of preference:**

1. **Vendor the OSNet module (recommended).** `torchreid/models/osnet.py` is a
   self-contained PyTorch module definition (model architecture + pretrained-URL
   table). For *embedding-only* use we don't need the training engine, datamanager,
   or the Cython-accelerated CMC evaluation (that Cython build is only used for
   mAP/rank metrics during evaluation). Copy `osnet.py` + the small
   `load_pretrained_weights` helper from `torchreid.utils`, depend only on
   `torch`/`torchvision`, and load weights from a file we ship in the image. This
   gives full control and zero legacy deps. (High confidence — verified against
   S3 and the `FeatureExtractor` source, which only uses `torchreid.models.build_model`
   + `load_pretrained_weights`.)
2. **Install from the upstream git repo with a trimmed dependency set** —
   `uv add torchreid@git+https://...` style, but curate deps (skip
   `tb-nightly`/lint pins) if we want the full torchreid API (e.g. the
   `FeatureExtractor` convenience class). More moving parts, still fine.
3. **Avoid `pip install torchreid` from PyPI** unless we accept the
   TensorFlow-nightly dep tree (S5/S6).
4. **BoxMOT's Re-ID module (S10)** ships the same OSNet/HACNN/MobileNetV2
   backbones as a maintained pip package (`pip install boxmot`) with model
   variants like `osnet_x0_25_msmt17` — this sidesteps torchreid packaging
   entirely, but note **BoxMOT is AGPL-3.0**; fine for a self-hosted personal
   container, a real consideration if the code is ever distributed.

**Weight download risk (high confidence on mechanism, medium on frequency):** at
first use with `pretrained=True`, torchreid downloads weights via
`gdown.download('https://drive.google.com/uc?id=...')` from Google Drive (S3, line
~480). Google Drive anonymous downloads are quota-limited and break regularly for
high-traffic files; the Model Zoo notes weights were moved to Google Drive in 2021
(S2). Recommendation regardless of path: **bake the chosen `.pth`/`.onnx` into the
Docker image** and set `model_path` explicitly; don't rely on runtime gdown.

### 1.3 Runtime alternatives on aarch64

`onnxruntime` (1.32.0) and `ncnn` (1.0.20261020) both publish aarch64 manylinux
wheels (S9). Since NCNN is the fastest Pi 5 runtime by a wide margin for
Ultralytics models (S8, below) and OSNet is a plain CNN that exports to ONNX
cleanly, the "torch once → ONNX → onnxruntime/ncnn at runtime" shape is viable and
removes torch from the hot path entirely.

---

## 2. Expected throughput on Pi 5 (CPU-only)

### 2.1 Measured anchor points (published, S8)

Ultralytics' Pi 5 benchmark (FP32, 640×640 input, inference time excludes
pre/post-processing; Ultralytics 8.4.108, Nov 2026):

| Runtime | YOLO26n ms/img | → img/s |
|---|---|---|
| PyTorch (eager) | 299.09 | 3.3 |
| TorchScript | 353.20 | 2.8 |
| ONNX (onnxruntime) | 125.99 | 7.9 |
| OpenVINO | 104.55 | 9.6 |
| NCNN | **67.03** | **14.9** |

Also (S8): YOLO26n ONNX = 128.42 ms/img vs YOLO11n ONNX = 147.3 ms/img (6.79 FPS)
— i.e. **YOLO11n-class nano detector ≈ 7 img/s ONNX, ~150 ms/img** on Pi 5 at 640.
NCNN for a nano model ≈ 67–90 ms/img ≈ **11–15 img/s** (67 ms is YOLO26n; YOLO11n
NCNN is somewhat slower — **[estimate]**).

The Ultralytics table also confirms Pi 5 silicon: BCM2712, Cortex-A76 ×4 @ 2.4 GHz,
LPDDR4X-4267.

### 2.2 FLOPs of the Re-ID candidates (published, S2 — computed at each model's own
input size, which is 256×128 for OSNet and 160×64 for HACNN)

| Model | Params | GFLOPs | Market1501 rank-1 (mAP) |
|---|---|---|---|
| osnet_x1_0 | 2.2 M | 0.98 | 94.2 (82.6) |
| osnet_x0_75 | 1.3 M | 0.57 | 93.7 (81.2) |
| osnet_x0_5 | 0.6 M | 0.27 | 92.5 (79.8) |
| **osnet_x0_25** | **0.2 M** | **0.08** | 91.2 (75.0) |
| hacnn* | 4.5 M | 0.5 (at 160×64) | 90.9 (75.6) |
| mobilenetv2_x1_0 | 2.2 M | 0.2 | 85.6 (67.3) |
| resnet50_fc512 (reference) | 24.6 M | 4.1 | 90.8 (75.3) |

Both torchreid's `compute_model_complexity` and Ultralytics' GFLOPs figures count
multiply-accumulates, so **ratios are directly comparable**.

### 2.3 Derived throughput **[estimate — the core estimate of this ticket]**

Calibrate ms-per-GFLOP from the measured nano-detector (6.5 GFLOPs at 640):

- ONNX: 126 ms / 6.5 ≈ **19 ms/GFLOP**
- PyTorch eager: 299 ms / 6.5 ≈ **46 ms/GFLOP**
- NCNN: 67 ms / 6.5 ≈ **10 ms/GFLOP**

Apply to the Re-ID models at 256×128, then add a small-model overhead floor
(≈5–10 ms for preprocessing/Python/dispatch — tiny convs are latency-, not
FLOP-, bound):

| Model (256×128) | PyTorch eager | ONNX runtime | Comment |
|---|---|---|---|
| osnet_x1_0 (0.98 GF) | ~45 ms → **12–20 img/s** | ~19 ms → **25–45 img/s** | |
| osnet_x0_75 (0.57 GF) | ~26 ms → **25–40 img/s** | ~11 ms → **50–90 img/s** | |
| osnet_x0_5 (0.27 GF) | ~12 ms → **40–70 img/s** | ~5 ms → **70–100+ img/s** | |
| **osnet_x0_25 (0.08 GF)** | ~4 ms compute → **~50–100+ img/s** | ~2 ms compute → **100+ img/s** | overhead-dominated |

Uncertainty: ±2× either way. The FLOP-scaling assumption (linear in FLOPs) holds
reasonably for conv nets of the same family at the same input; the overhead floor
is the dominant unknown for x0.25/x0.5. These are single-stream numbers; batching
crops (e.g. 16 crops/batch) improves small-model throughput further.

**Nano-class detector (YOLOv8n/YOLO11n class), per full photo at 640:** measured
~126–147 ms ONNX (S8) and ~67–90 ms NCNN **[part estimate]** → **~7–11 img/s
ONNX, ~11–15 img/s NCNN**. This is the expensive half of the pipeline per photo.

### 2.4 Does the nightly batch fit the "minutes" budget?

Yes, with a large margin (high confidence). Per photo, worst-plausible shape:

- 1 detector pass on a 640-letterboxed photo: 70–150 ms
- 1–4 person crops embedded: 3×20 ms ≈ 6–80 ms depending on variant/runtime
- decode/resize/IO: ~50–150 ms

≈ **0.15–0.4 s per photo** with an ONNX/NCNN runtime → **~3–6 photos/s**.
A nightly batch of 50 photos ≈ **10–20 s**; even the slowest PyTorch-eager path
stays under ~1 min. The "minutes" budget is comfortably met.

---

## 3. Credible lightweight alternatives (before committing)

| Option | Why consider | Why hesitate |
|---|---|---|
| **OSNet-AIN x0.25 / x0.5** (torchreid) | Instance-normalised OSNet is the *domain-generalization* variant; `osnet_ain_x1_0` is the strongest cross-dataset model in the zoo (73.3 rank-1 MS+D+C→M vs 66.6 for plain x1_0, S2). ImageNet-pretrained `ain` x0.75/x0.5/x0.25 released Aug 2021 (S1). Our task is exactly cross-domain (Market-style training → family photos). | Slightly less battle-tested than plain osnet weights; same install story. |
| **HACNN** | 0.5 GFLOPs at 160×64, 4.5 M params, 90.9 (75.6) Market rank-1 — best accuracy-per-FLOP in the zoo (S2); supported by both torchreid and BoxMOT (S10). | Trained from scratch (asterisk in zoo); older architecture; 160×64 input gives less spatial detail. |
| **MobileNetV2 x1.0 Re-ID head** | 0.2 GFLOPs, 2.2 M — the fastest family in the zoo. | Accuracy clearly below OSNet (85.6/67.3 vs 94.2/82.6, S2). Only if throughput were the binding constraint — it isn't at our scale. |
| **BoxMOT Re-ID module (S10)** | Maintained pip package, same OSNet/HACNN/MLFN/LMBN/ResNet50 backbones with per-dataset weights (e.g. `osnet_x0_25_msmt17`), no torchreid install needed, actively developed (Nov 2026). | **AGPL-3.0**; pulls a larger dependency surface (ultralytics); heavier than vendoring one file. |
| **CLIP/ViT-class Re-ID (TransReID, CLIP-ReID, FastReID SBS-R50)** | Higher cross-domain accuracy in the literature. | 10–100× the compute of OSNet; wrong tool for a Pi 5 CPU. Not credible here. |
| **Ultralytics built-in BoT-SORT Re-ID (`yolo26n-cls` as embedder)** | Zero extra model. | Classification features, not Re-ID-trained; S8 support threads show it's fragile for exactly this use. |

**Recommendation-shaped summary:** start with `osnet_x0_25` or `osnet_ain_x0_25`
(embedding-only cost is negligible on Pi 5; x1.0 also affordable via ONNX), keep
`osnet_x0_5` as the accuracy step-up, and prefer vendored OSNet over the PyPI
repack. Accuracy headroom between x0.25 (75.0 mAP Market) and x1.0 (82.6) matters
more for cross-domain family photos than the ~10× FLOP gap, given we have minutes
of budget.

---

## 4. Backfill cost estimate (tens of thousands of photos)

**[Estimate]** — anchored on the measured S8 numbers and the FLOP scaling above.
Assume a 30,000-photo library, ~2–4 person crops per photo (family photos skew
higher), photo decode + resize ~50–150 ms.

| Path | Per photo | 30,000 photos |
|---|---|---|
| Detector (ONNX 640) + 3 crops × osnet_x0_25 (PyTorch) | ~0.15–0.3 s | **1.5–2.5 h** |
| Detector (ONNX) + 3 crops × osnet_x0_25 (ONNX) | ~0.2–0.4 s (incl. IO) | **2–3.5 h** |
| Same with osnet_x1_0 ONNX | ~0.25–0.5 s | **2.5–4.5 h** |
| PyTorch-eager everything (no runtime work) | ~0.6–1.2 s | **5–10 h** |
| **Embedding-only** (reuse Immich's existing detections/boxes; no detector pass) | ~0.05–0.1 s | **~30–90 min** |

Round-number answer: **an overnight job** — roughly **2–10 hours** for a full
backfill on a Pi 5 depending on runtime and model size; the embedding-only variant
(reusing detections Immich has already computed) is **under ~2 hours**. Per the
goal, backfill is a one-off decision input: the nightly steady-state (dozens of
photos) is seconds-to-minutes either way, so the backfill cost alone should not
force a bigger machine.

Caveats: thermal throttling on an uncooled Pi 5 (S8 recommends cooling for
sustained loads); SD-card read throughput argues for running the backfill against
an SSD or the Immich API's already-served thumbnails rather than raw file reads.

---

## Uncertainty summary

- Measured, high confidence: PyTorch aarch64 wheel availability (S7); Pi 5
  nano-detector runtimes (S8); torchreid packaging facts (S1–S6); OSNet FLOPs and
  accuracies (S2); runtime wheel availability (S9); BoxMOT model list & license (S10).
- Derived, medium confidence: per-model embedding ms/img (FLOP-scaled from S8 with
  an overhead floor); backfill totals (assumed crops/photo dominate).
- Unknown / not verified here: actual gdown failure frequency in 2026 (mechanism
  verified in S3); exact YOLO11n-vs-YOLO26n NCNN delta; real-world crops-per-photo
  for the user's library (measure during ticket 04/07 design).
