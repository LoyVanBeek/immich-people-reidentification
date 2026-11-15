# OSNet-AIN x1.0 (combineall) over nano OSNet for Re-ID embeddings

Ticket 03's viability estimate was built on `osnet_x0.25` (~50–100+ embeddings/s on the Pi 5), and the natural continuation was a nano variant. But a fact-check of torchreid's model zoo found **no ReID-trained checkpoint for `osnet_ain_x0.25`** — only ImageNet-pretrained init weights, useless for an inference-only container — while `osnet_ain_x1_0` ships combineall (multi-dataset) weights, the strongest domain-generalization artifact in the zoo. Since the ticket-04 grilling relaxed the nightly budget to 30 minutes for a day's photos, the YOLO11n detector (7–15 img/s) is the bottleneck either way, so embedding compute is effectively free. We therefore run `osnet_ain_x1_0` + combineall weights (256×128 input → 512-dim embedding), vendored from torchreid (MIT).

## Considered Options

- `osnet_x0_25` + MSMT17-trained weights (fallback): ~10× cheaper, but a much weaker embedder and no AIN domain robustness — rejected because embedding quality drives cluster quality, the core value of the pipeline.
- BoxMOT's Re-ID module (AGPL-3.0): rejected; vendoring torchreid's model definition is simpler and permissive.

## Consequences

Switching embedding models later invalidates all stored embeddings: a model change requires wiping the store and re-embedding (clusters are rebuilt from scratch each run, so only the embedding store is affected). This is accepted; see the map's "Embedding/model versioning" fog item.
