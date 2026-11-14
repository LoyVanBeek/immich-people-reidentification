# Research: Immich ML stack pinning

Ticket: [.scratch/person-reid-v1/issues/02-immich-ml-stack-pinning.md](../issues/02-immich-ml-stack-pinning.md)
Date: 2026-11-14
Sources are the `immich-app/immich` GitHub repository at tag `v3.4.1` (the latest stable release at the time of research, per the [releases API](https://api.github.com/repos/immich-app/immich/releases)) unless noted otherwise.

## TL;DR

**Immich's machine-learning service contains no PyTorch at all.** Every model — CLIP, facial recognition, OCR — runs through **ONNX Runtime** (`onnxruntime`), selected per device via execution providers. `"use the same pytorch version as Immich itself uses"` (goal.md line 26) is therefore not a real constraint: there is no such version to match. The constraints that actually matter for embedding compatibility are the **`immich-model` package revision** (pinned by git rev) and the ONNX model artifacts + preprocessing, not any framework version.

## 1. What the ML service pins, and why

`machine-learning/pyproject.toml` ([v3.4.1](https://github.com/immich-app/immich/blob/v3.4.1/machine-learning/pyproject.toml)) has **no torch/torchvision/transformers dependency in the main dependency list**. The runtime-relevant pins:

| Dependency | Pin | Note |
|---|---|---|
| `onnxruntime` (cpu/armnn/rknn extras) | `>=1.23.2,<2` | resolves to **1.31.0** in [`uv.lock`](https://github.com/immich-app/immich/blob/v3.4.1/machine-learning/uv.lock) (1.31.1 on `main`) |
| `onnxruntime-gpu` (cuda extra) | `>=1.27.1,<1.28` | comment in the file: *"last ORT release supporting CUDA 12.2, also need to handle CUDA 13"* |
| `onnxruntime-openvino` | `>=1.24.1,<2` | replaced at image build time by a wheel **built from source at ONNX Runtime `v1.30.0`** against OpenVINO `2026.5.0.dev20261005` (renovate-managed, see Dockerfile) |
| `onnxruntime-migraphx` (rocm) | `>=1.23.2,<2` | |
| `onnx` | `>=1.22.0` | model format library |
| `immich-model` | git rev `a1eeff2c2118d76067efecf4cc4630998eac519a` ([v3.4.1](https://github.com/immich-app/immich/blob/v3.4.1/machine-learning/pyproject.toml)) | pinned via `[tool.uv.sources]` to `https://github.com/immich-app/ml-models.git` |

There is no documented "why" for the CPU floor beyond being a plain SemVer range; the only in-file justification comment is the CUDA one (CUDA 12.2 support ceiling). The renovate bot keeps these ranges moving with releases.

**Torch absence is confirmed at the lock level:** `uv.lock` at v3.4.1 contains no `torch` package at all, and the model-support package [`immich-app/ml-models`](https://github.com/immich-app/ml-models) (`immich_model`) describes its runtime core as *"The torch-free core: `immich_model.runtime` plus the onnx lowerings"* — its `pyproject.toml` puts `torch` only behind the `export`/`rknn` **optional** extras, used when *exporting* models to ONNX, never at inference time ([ml-models pyproject.toml](https://github.com/immich-app/ml-models/blob/main/pyproject.toml)).

## 2. Python version

- `requires-python = ">=3.12,<4.0"` in `pyproject.toml`; mypy/ruff target `py312`.
- The shipped images run **Python 3.13**: the Dockerfile's cpu/openvino/armnn/rocm builder and all `prod-*` CPU-family stages use `python:3.13-slim-trixie` ([machine-learning/Dockerfile](https://github.com/immich-app/immich/blob/v3.4.1/machine-learning/Dockerfile)). The rknn builder/prod stage uses `python:3.12-slim-trixie`.

## 3. ARM64 images and what runs inference on ARM

**Yes, working ARM64 images are published, natively built.** The Docker workflow ([`.github/workflows/docker.yml`](https://github.com/immich-app/immich/blob/v3.4.1/.github/workflows/docker.yml), `ml-variants` matrix) builds:

| Image variant | Platforms |
|---|---|
| `immich-machine-learning` (default, device `cpu`) | `linux/amd64` + `linux/arm64` (default when `platforms` is unset; `linux/arm64` is built natively on `ubuntu-24.04-arm` runners per [multi-runner-build.yml](https://github.com/immich-app/devtools/blob/main/.github/workflows/multi-runner-build.yml)) |
| `-cuda`, `-openvino`, `-rocm` | `linux/amd64` only |
| `-armnn`, `-rknn` | `linux/arm64` only |

What actually executes inference on ARM:

- **Default (cpu) image on arm64: ONNX Runtime, `CPUExecutionProvider`** — a plain PyPI `onnxruntime` manylinux aarch64 wheel; the code explicitly treats CPU as the provider that handles dynamic shapes (`sessions/ort.py`, `DYNAMIC_PROVIDERS = frozenset({"CPUExecutionProvider"})`) and per-machine prepacked graphs keyed on `platform.machine()`.
- **`-armnn` (arm64-only): ArmNN via a drop-in session wrapper** — the ArmNN libraries are fetched from [`ARM-software/armnn` release `v26.07` (aarch64 tarball)](https://github.com/ARM-software/armnn) and `immich_ml/sessions/ann/__init__.py` wraps it as *"drop-in replacement for ONNX session"* (Mali GPU; per [docs](https://github.com/immich-app/immich/blob/v3.4.1/docs/docs/features/ml-hardware-acceleration.md), ARM NN is *not* used for Smart Search due to model compatibility, only face/other jobs).
- **`-rknn` (arm64-only): Rockchip NPU** — `rknn-toolkit2 v2.3.0` runtime (`librknnrt.so`), `.rknn` compiled models, only RK3566/68/76/88 SoCs.

So on generic ARM64 (e.g. Raspberry Pi 5, generic aarch64 VPS), inference is **ONNX Runtime on CPU** — no PyTorch, and no ArmNN unless the user opts into the `-armnn` image with a Mali GPU.

## 4. Is "same PyTorch version as Immich" the right constraint?

**No — it is vacuous.** Findings to surface to the spec:

1. **There is no PyTorch in Immich's ML service.** It is not merely "Immich uses an older torch"; the dependency does not exist in `pyproject.toml` or `uv.lock`. Inference is `onnxruntime.InferenceSession`-based for every device (`immich_ml/sessions/ort.py`), with ArmNN and RKNN as alternative session backends behind the same interface.
2. The analogous (and satisfiable) constraint is: **run inference with the same ONNX Runtime major series and, more importantly, the same model artifacts and preprocessing** as Immich. Embedding compatibility depends on:
   - the pinned `immich-model` git rev (model zoo + per-EP graph rewrites/lowerings, `immich_model.runtime`),
   - the ONNX model files Immich downloads (InsightFace antelopev2/buffalo_* etc., CLIP),
   - the transforms/preprocessing in `immich_ml/models/transforms.py`.
3. A Re-ID container using OSNet/torchreid will necessarily pull in PyTorch — that torch version is **free choice**, since it runs in a separate process/container and nothing about it must match Immich. Only the *output contract* matters (see ticket 03/04 for model choice and whether an ONNX-exportable Re-ID model is preferable to a torch one).

## Where this file lives

No prior convention for research notes existed in the repo, so findings live next to their tickets under `.scratch/<feature-slug>/research/`.
