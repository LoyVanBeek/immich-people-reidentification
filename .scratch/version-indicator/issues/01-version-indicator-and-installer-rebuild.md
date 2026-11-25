# Version indicator and installer rebuild

Part of: [Version indicator in the UI — Map](../map.md)
Type: task
Status: resolved

## Question

(1) Show the running git commit(s) in the UI so a deploy is visibly different
after `git pull`. (2) Make `scripts/install.sh` build/use fresh images: does it
today? (Answer: no — it never built; `reid-batch:local`, once present, was reused
as-is by compose `up -d` and by the systemd units, which only `docker pull` the
ghcr image.)

## Answer

Implemented and committed (see map):

- `src/reid/version.py`: `git_commit()` — baked `GIT_COMMIT.txt` (Dockerfile
  build-arg, written into site-packages), else `git rev-parse --short HEAD` in a
  dev checkout, else "unknown".
- `Dockerfile`: `ARG GIT_COMMIT=unknown`, baked at build; `docker-compose.yml`
  passes `GIT_COMMIT: ${GIT_COMMIT:-unknown}` to all build blocks.
- Core: `/api/state` gains `core_commit`. UI server: injects its own commit as
  `window.__REID_UI__.commit` (config script now always emitted). Top bar shows
  `ui <short> · core <short>` (`.ver` chip, full value in the tooltip).
- `scripts/install.sh`: `ensure_local_image()` rebuilds `reid-batch:local` with
  `--build-arg GIT_COMMIT=<head>` when the image is missing or HEAD differs from
  the `.reid-image-commit` marker; compose mode uses `up -d --build` for the
  local profile; usage text documents the pull→install flow.

Verified: ruff/mypy clean, pytest (incl. new `tests/test_version.py` and state/
index assertions), jsdom UI suite 5/5 (asserts the `ui … · core …` chip),
`bash -n` + installer dry-run idempotency diff clean.
