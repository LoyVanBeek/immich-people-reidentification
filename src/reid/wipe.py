"""The reid-wipe entry point: erase the local store at batch-run level.

Destructive on purpose and deliberately not reachable from the confirmation
UI: wiping the store erases every stored embedding and resets the new-asset
feed checkpoint, so the next batch run starts clean and re-embeds the whole
library. Operators run it after changing detector or embedding settings.
"""

from __future__ import annotations

from pathlib import Path

import click

from .config import ConfigError, load_config
from .store import Store, StoreError


@click.command()  # type: ignore
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the reid.toml config file.",
)  # type: ignore
def main(config_path: Path) -> None:
    """Wipe the local store: embeddings and checkpoints."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    try:
        removed = Store(cfg.store_path).wipe()
    except StoreError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    click.echo(f"store wiped: {removed} embeddings removed, checkpoint reset")


if __name__ == "__main__":
    main()
