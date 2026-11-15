"""The reid-batch entry point: one batch run, suitable for cron or systemd."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .batch import run_batch
from .config import ConfigError, load_config
from .immich import ImmichError
from .store import StoreError

_FATAL_ERRORS = (ImmichError, ConfigError, StoreError)


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the reid.toml config file.",
)
def batch(config_path: Path) -> None:
    """Run one batch: fetch new Immich photos and advance the checkpoint."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    level = getattr(logging, cfg.run.log_level.upper())
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)

    try:
        summary = run_batch(cfg)
    except _FATAL_ERRORS as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    for asset in summary.assets:
        click.echo(f"asset {asset.id} {asset.original_file_name} created_at={asset.created_at}")
    click.echo(
        f"batch run complete: assets fetched={len(summary.assets)} "
        f"checkpoint={summary.checkpoint_before} -> {summary.checkpoint_after}"
    )
    click.echo(
        f"appearances embedded={summary.appearances} "
        f"edge_clipped_kept={summary.edge_clipped_kept} "
        f"assets_without_persons={summary.assets_without_persons}"
    )
    counts = summary.gate_counts
    click.echo(f"gate drops: low_confidence={counts.low_confidence} below_min_height={counts.below_min_height}")


if __name__ == "__main__":
    batch()
