# Version indicator in the UI — Map

Labels: wayfinder:map

## Destination

The UI shows which git commit is running — both the UI's own commit and the
core's — in the top bar, and `scripts/install.sh` rebuilds the local image
(`reid-batch:local`) when the repo's git HEAD changed since the last build, so
`git pull && install.sh` on the Pi actually deploys the new code.

## Notes

- The commit is baked into the image via Dockerfile build-arg `GIT_COMMIT`
  (written to site-packages `reid/GIT_COMMIT.txt`); in a dev checkout
  `reid/version.py` falls back to `git rev-parse --short HEAD`, else "unknown".
- Core reports it as `core_commit` in `/api/state`; the UI server injects its
  own as `window.__REID_UI__.commit`; the top bar renders `ui X · core Y`.
- Previously `ui_version()` was only a content hash for cache-busting and no
  git commit was baked anywhere; install.sh never built images, so a stale
  `reid-batch:local` kept running after `git pull` (compose `up -d` without
  `--build` reuses the existing image; systemd units run the named image and
  only `docker pull` the ghcr one).

## Decisions so far

- [Version indicator + installer rebuild](issues/01-version-indicator-and-installer-rebuild.md) (`5e48d74`):
  baked GIT_COMMIT build-arg + `version.git_commit()` fallback chain; `core_commit`
  in /api/state; `__REID_UI__.commit` injection; top-bar chip; install.sh
  `ensure_local_image` rebuilds reid-batch:local on HEAD change (marker
  `.reid-image-commit`) and compose `up -d --build` for the local profile.

## Not yet specified

- (empty)

## Out of scope

- Publishing commit-tagged ghcr images from CI (the docker-build pipeline is
  separate; images there get :latest + tags). The build-arg default "unknown"
  covers builds without it.
