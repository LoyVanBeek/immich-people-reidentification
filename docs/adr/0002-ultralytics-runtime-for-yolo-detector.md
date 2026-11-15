# Ultralytics as the runtime for the YOLO11n person detector

Ticket 10 needs a person detector running inside the inference-only container
on the Pi 5. The model itself was fixed by the spec: YOLO11n, the official
Ultralytics release weights (`yolo11n.pt`, vendored so the image needs no
network). The remaining decision is the runtime that executes it.

We use the `ultralytics` package (AGPL-3.0) directly: `YOLO("yolo11n.pt")`
with `imgsz=640`, `classes=[0]` (person only), and a low internal confidence
cutoff — our own ordered quality gates (confidence floor, minimum box
height) are applied to the raw detections afterwards. The PyTorch weights are
committed under `src/reid/weights/` and are picked up by the wheel, so the
image needs no network and no runtime download.

## Considered Options

- Ultralytics package (AGPL-3.0) — chosen. It is the reference implementation
  of the model we vendored, executes on the Pi 5's CPU, and its detection API
  is stable. Its AGPL license is inherited by our distribution of the
  container image; this project is a private deployment (the "private v1"
  scope agreed for ticket 03's SCRFD question), so AGPL obligations are
  accepted for now.
- Export YOLO11n to ONNX and run it with `onnxruntime` (MIT): avoids AGPL and
  would be the natural route if the image is ever distributed publicly, at
  the cost of an export step and a second runtime dependency to keep aligned
  with the weights. Rejected for now as premature; recorded as a migration
  path, not a dead end.
- Face-detection style models (e.g. SCRFD) as detectors: out of scope — the
  spec deliberately detects whole bodies, not faces, so that people facing
  away from the camera still get an appearance.

## Consequences

- The built image distributes AGPL-3.0 code (ultralytics) alongside
  MIT-licensed vendored torchreid model definitions and the vendored
  `yolo11n.pt` weights (AGPL). Source disclosure obligations apply to the
  image distribution; the repo is the source.
- If the project is ever offered to third parties under a permissive license,
  the detector must move to an ONNX/onnxruntime runtime (weights stay AGPL
  unless retrained or replaced) — revisit this ADR then.
- Swapping the detector or its weights does not invalidate stored embeddings
  directly, but detection changes (conf thresholds, model version) change
  which appearances get stored; treat a detector change like a `[detection]`
  config change and wipe the store at that boundary.
