# Installer stop/restart — Map

Labels: wayfinder:map

## Destination

`scripts/install.sh` stops the running reid-core/reid-ui before building and
installing, and starts them again afterwards (only those that were running,
unless `--up`) — so a rebuilt `reid-batch:local` after `git pull` always
replaces the old containers instead of leaving them bound to the ports.

## Notes

- Gap being fixed: systemd mode only restarted *changed* unit files; a rebuilt
  image doesn't change the unit, so the old container kept running.
- `reid-batch.timer` is never stopped; the oneshot batch service needs no restart.
- Dry-run stays read-only and idempotent (CI diffs two dry-run logs).

## Decisions so far

- [Stop/restart services around install](issues/01-stop-restart-around-install.md):
  scope = core+ui in both modes; restart = previously-running only, `--up`
  starts all; order = stop → build → install → start; dry-run logs
  "Would …" only.

## Not yet specified

- (empty)

## Out of scope

- Zero-downtime blue/green deploys (single Pi, nightly installs — downtime is fine).
- Touching the batch timer's schedule.
