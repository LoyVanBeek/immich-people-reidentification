# Ground Truth Fixtures for Person Re-Identification Evaluation

## Problem

The current test fixtures use the same JPEG repeated (e.g., 5x `reid-fixture-indoor.jpg` for the indoor cluster). This means:
- Embeddings are identical (distance = 0)
- Clustering tests don't validate real-world embedding similarity
- No signal on whether OSNet-AIN handles pose, lighting, occlusion variation

We need real photo sets of the same person (and different people) where only **one photo per person has an Immich face label**, to validate:
1. **Anchor propagation**: Does the labeled photo's name propagate to unlabeled cluster members?
2. **Clustering quality**: Do same-person embeddings actually cluster at our `distance_threshold`?
3. **Threshold tuning**: Are different people correctly separated?

## Scope

### Fixture photos
Add to `tests/fixtures/photos/`:
- **Person A (Alice)**: 3–5 photos, different conditions (indoor, outdoor, pose variation), only 1 labeled
- **Person B (Bob)**: 3–5 photos, different conditions, only 1 labeled
- **Person C (Carol)**: 3–5 photos, different conditions, only 1 labeled
- **Negative pairs**: 2–3 photos of different people that should *not* cluster together

Each photo needs:
- A unique asset ID (ULID format matching `library.py` convention)
- A fixture JPEG in `tests/fixtures/photos/`
- Entry in `library.PHOTO_FILE_NAMES`
- Entry in a new registry function (e.g., `ground_truth_registry()`)

### Test acceptance criteria
New test in `tests/test_anchoring.py` (or new file `tests/test_ground_truth.py`):
- Run batch with the ground truth registry
- Assert each person's unlabeled photos get `proposed` status with the correct name
- Assert different people do not cluster together (or if they do, the cluster is `conflicted`)
- Log the embedding distances for threshold analysis

### Configuration
- Use a `distance_threshold` that reflects production intent (not 0.01 which only clusters identical embeddings)
- May need to adjust `min_cluster_size` for the fixture counts

## Out of Scope
- Automated precision/recall metrics (can be a follow-up ticket)
- CI integration for regression detection (follow-up)
- Photo sourcing — user provides photos manually
