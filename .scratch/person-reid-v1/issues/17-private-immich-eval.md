# 17-private-immich-eval

Type: task
Status: ready-for-agent
Blocked by:

## Summary

Evaluate the pipeline on a private set of photos that already live in the user's
Immich instance, without ever committing the photos, their labels, or the
resulting store to git. This is the **manual path**: zero code changes required.
A scoring CLI (`reid-eval`) is recorded as an optional later slice.

Related: [16-add-ground-truth-fixtures.md](16-add-ground-truth-fixtures.md) —
this ticket provides the real-data counterpart for threshold calibration.

## Privacy model

- `data/` is git-ignored ("local stores (contain proposals with real names —
  never commit)") — the eval store and the labels file live there.
- The Immich API key comes from the `IMMICH_API_KEY` env var, never from a
  committed config file.
- The eval config deliberately omits `write_api_key` → the UI's promote /
  real-write path fails safely; dry-run confirmation still works (read key only).
- Photos stay in Immich; nothing photographic enters the repo.

## Manual runbook

### 1. Curate the eval set in Immich

- Upload the selected photos (duplicates are fine) into a **dedicated album**
  so the real library stays untouched.
- In Immich, label **exactly one photo per person** — that photo is the anchor.
  Leave every other photo in the set unlabeled.
- This reproduces the target state from the original idea: same-person photo
  sets where only one photo carries an Immich label.

### 2. Write `reid-eval.toml`

Copy of `reid.toml` with deliberate differences:

```toml
[store]
path = "data/eval.sqlite3"        # separate eval store

[immich]
base_url = "http://<pi>:2283"     # real Immich; key via IMMICH_API_KEY env
# NO write_api_key → promotion fails safely during evaluation
```

Clustering knobs (`[clustering]`) start at production defaults; they are the
thing being tuned.

### 3. Run the evaluation

```bash
reid-batch --config reid-eval.toml    # detect → gate → embed → cluster → anchor
reid-ui   --config reid-eval.toml     # full 3-pane review on the private photos
```

- The UI reads the eval store and fetches crops from real Immich (read key).
- Window summaries print per-window cluster stats as usual.
- Reviewing in the UI exercises the normal dry-run confirm flow; promotion
  errors out safely (no write key).

### 4. Threshold-tuning loop

- Changing `distance_threshold` (config or ⚙ dropdown tuning knob) and
  re-running `reid-batch --config reid-eval.toml` **re-clusters the eval store
  from scratch without re-embedding** (ticket-11 design: clusters rebuilt from
  stored embeddings every run).
- Adding NEW photos later: they must be newer than the store's checkpoint
  (`createdAfter` feed). For an arbitrary new set, wipe the eval store first
  (`reid-wipe --config reid-eval.toml` equivalent: point the wipe at the eval
  store path) and re-run — eval-scale sets re-embed quickly.

### 5. What to observe (manual ground truth)

Keep the expected truth outside git (e.g. on paper / local notes): for each
person, which photos should cluster together. Then check:

- Did all of a person's photos land in the anchor's cluster? (recall)
- Is the anchor's cluster free of other people's photos? (precision)
- Did the anchor's name propagate to every unlabeled member? (proposed status)
- Do different people stay separated, or merge? (threshold too high)
- Are same-person photos stranded as singletons? (threshold too low, or
  embedding robustness problem: pose/lighting/occlusion)

## Guardrails

- Labels file, report, store: only under `data/` (already git-ignored).
- Never point the eval config's store at the production store path.
- The eval store contains real names — treat it like `data/` already does.
- Optional future hardening: pre-commit check that `git ls-files` shows nothing
  under `data/` (insurance against `git add -f`).

## Optional later slice: `reid-eval` scoring command

Not needed for the manual path; recorded here so the idea survives:

- `data/eval-labels.toml` keyed by file name (not opaque asset IDs):

  ```toml
  [people.albert]
  photos = ["IMG_1001.jpg", "IMG_1002.jpg"]

  [extra]
  photos = ["IMG_9001.jpg"]   # distractors
  ```

- `reid-eval --config reid-eval.toml [--fresh]`: fetch feed, filter to the
  labels' file names client-side, run the batch, then score per person:
  cluster recall/precision, anchor-propagation correctness, pairwise distance
  matrix (per expected person + nearest different-person distances) for
  threshold calibration; optional report file under `data/`.
- Guardrail: `git check-ignore` on the labels/report paths; refuse when not
  ignored (or require an explicit override flag).

## Acceptance Criteria

- [ ] `reid-eval.toml` committed WITHOUT secrets or names (no `write_api_key`,
      no `api_key`)
- [ ] Eval store and any labels/notes live under `data/`; `git status` is clean
      after a full eval run
- [ ] A full eval run produces window summaries; the UI shows the private
      photos' clusters with anchor propagation statuses
- [ ] Confirming a cluster in dry-run works; promote fails safely with a clear
      error (no write key)
- [ ] Threshold sweep documented: at least two `distance_threshold` values
      re-run from the store without re-embedding
- [ ] Manual observations recorded (per person: clustered / separated /
      stranded) — used as calibration input for the production threshold

## Comments
