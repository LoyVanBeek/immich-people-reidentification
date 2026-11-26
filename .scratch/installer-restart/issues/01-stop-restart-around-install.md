# Stop/restart services around install

Part of: [Installer stop/restart — Map](../map.md)
Type: task
Status: resolved

## Question

When installing, stop the current services being installed and restart them
when done installing & building — because today a rebuilt `reid-batch:local`
does not change the systemd unit file, so "No systemd units changed" meant no
restart and the old container kept serving.

## Answer

Implemented in `scripts/install.sh` (committed as `24cb75c` on main):

- `active_services()` — detects running reid-core/reid-ui (compose names or
  systemd units) per mode; empty in dry-run.
- `stop_services()` — stops them before the build (`compose stop` /
  `systemctl stop`); dry-run logs `Would stop …`.
- `start_services()` — after install, starts the previously-running units;
  `--up` enables+starts both (unchanged `--up` semantics).
- `main()` order: detect mode/image → record+stop → `ensure_local_image`
  (build) → reid.toml/data dir → mode install → start → health check.
- Compose mode starts via `up -d [--build] ${COMPOSE_TARGETS}` where targets =
  previously-running services (or both with `--up`); if nothing was running,
  it does not start them.
- `reid-batch.timer` untouched; oneshot batch service needs no restart.

Verified: `bash -n` clean, `--dry-run` (systemd and compose) exits 0 and two
consecutive runs diff clean (CI's idempotency check), dry-run logs show the
Would stop → build → install → start order.
