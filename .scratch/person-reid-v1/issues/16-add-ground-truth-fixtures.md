# 16-add-ground-truth-fixtures

Type: task
Status: ready-for-human
Blocked by:

## Summary

Add real photo fixtures for ground truth evaluation of clustering and anchor propagation.

## Acceptance Criteria

- [ ] 3–5 photos per person (Alice, Bob, Carol) in `tests/fixtures/photos/` with pose/lighting variation
- [ ] 2–3 negative-pair photos (different people, should not cluster)
- [ ] Each person has exactly **one** photo with a face label in the stub Immich
- [ ] New registry function `ground_truth_registry()` in `tests/library.py`
- [ ] New acceptance test asserting anchor propagation works on real embeddings
- [ ] Test logs pairwise embedding distances for threshold analysis

## Implementation Notes

1. **Photo naming**: `reid-gt-alice-1.jpg`, `reid-gt-alice-2.jpg`, etc.
2. **Asset IDs**: Continue ULID sequence from `PHOTO_13_LATE_ID` (`000000000011` onward)
3. **Stub faces**: Only the first photo of each person gets a `person_name` in `stub_immich.faces`
4. **Test config**: Use a realistic `distance_threshold` (e.g., 0.4–0.6 based on OSNet-AIN literature)
5. **Test output**: Print distance matrix for manual threshold tuning

## Photo Requirements (for the human collecting fixtures)

### Quantity & People
- **3 people** (Alice, Bob, Carol) × **3–5 photos each** = 9–15 photos
- **2–3 negative photos** of different people who should NOT cluster

### File Placement
```
tests/fixtures/photos/
├── reid-gt-alice-1.jpg
├── reid-gt-alice-2.jpg
├── reid-gt-alice-3.jpg
├── reid-gt-bob-1.jpg
├── reid-gt-bob-2.jpg
├── reid-gt-bob-3.jpg
├── reid-gt-carol-1.jpg
├── reid-gt-carol-2.jpg
├── reid-gt-carol-3.jpg
├── reid-gt-neg-1.jpg
└── reid-gt-neg-2.jpg
```

### Labeling Rule (Critical)
**Only ONE photo per person gets a face label in Immich:**
| Photo | Immich Face Label |
|-------|-------------------|
| `alice-1.jpg` | ✅ "Alice" (anchor) |
| `alice-2.jpg` | ❌ No label |
| `alice-3.jpg` | ❌ No label |
| `bob-1.jpg` | ✅ "Bob" |
| `bob-2.jpg` | ❌ No label |
| `carol-1.jpg` | ✅ "Carol" |
| `carol-2.jpg` | ❌ No label |
| `neg-1.jpg` | ❌ No label |
| `neg-2.jpg` | ❌ No label |

The test stub simulates this — you only provide the photo files.

### Photo Quality
- **Resolution**: ≥ 640px on longest side (matches Immich preview)
- **Face visibility**: At least one clear face per photo (for Immich face detection)
- **Variation**: Different clothes, lighting, pose, background — tests Re-ID robustness
- **Format**: JPEG

## Test Sketch

```python
def test_ground_truth_anchor_propagation(reid_stub, make_config):
    # Use ground truth registry with real photos
    # Configure stub faces: only alice-1, bob-1, carol-1 have person_name
    # Run batch
    # Assert: alice-2, alice-3... get proposed "Alice"
    # Assert: bob-2, bob-3... get proposed "Bob"
    # Assert: carol-2, carol-3... get proposed "Carol"
    # Assert: negative pairs don't cluster (or are conflicted if they do)
    # Log: embedding distances for all pairs
```

## Comments