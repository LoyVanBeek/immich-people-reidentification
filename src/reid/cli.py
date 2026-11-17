"""The reid-batch entry point: one batch run, suitable for cron or systemd."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

from .batch import run_batch
from .config import ConfigError, load_config
from .immich import ImmichError, isoformat_z
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
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    try:
        summary = run_batch(cfg)
    except _FATAL_ERRORS as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    def _fmt_iso(dt: datetime | None) -> str:
        """Format datetime as ISO string with Z suffix for UTC."""
        if dt is None:
            return ""
        return isoformat_z(dt)

    for asset in summary.assets:
        click.echo(
            f"asset {asset.id} {asset.original_file_name} created_at={_fmt_iso(asset.created_at)}"
        )
    # Format datetimes as ISO strings with Z suffix for consistent output
    cp_before = _fmt_iso(summary.checkpoint_before)
    cp_after = _fmt_iso(summary.checkpoint_after)
    click.echo(
        f"batch run complete: assets fetched={len(summary.assets)} "
        f"checkpoint={cp_before} -> {cp_after}"
    )
    click.echo(
        f"appearances embedded={summary.appearances} "
        f"edge_clipped_kept={summary.edge_clipped_kept} "
        f"assets_without_persons={summary.assets_without_persons}"
    )
    counts = summary.gate_counts
    click.echo(
        f"gate drops: low_confidence={counts.low_confidence} below_min_height={counts.below_min_height}"
    )
    for window in summary.windows:
        clusters = window.clusters

        def pct(count: int, clusters: int = clusters) -> str:
            return f"{100.0 * count / clusters:.1f}%" if clusters else "0.0%"

        click.echo(
            f"window {window.window}: appearances={window.appearances} clusters={clusters} "
            f"anchored={pct(window.anchored)} conflicted={pct(window.conflicted)} "
            f"anchor_less={pct(window.anchor_less)} singletons={window.singletons} "
            f"proposals={window.proposals} ask_for_name={window.ask_for_name}"
        )


if __name__ == "__main__":
    batch()
