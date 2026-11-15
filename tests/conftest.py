"""Shared fixtures for the container-boundary test seam."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from reid.cli import batch
from . import library
from .stub_immich import API_KEY, StubImmich


@pytest.fixture
def stub() -> Iterator[StubImmich]:
    server = StubImmich(library.asset_registry())
    server.start()
    yield server
    server.stop()


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[[StubImmich, int], Path]:
    """Write a reid.toml pointing a run at a stub Immich with a fresh store path."""

    def _make(server: StubImmich, page_size: int = 1000) -> Path:
        config_path = tmp_path / "reid.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[immich]",
                    f'base_url = "{server.base_url}"',
                    f'api_key = "{API_KEY}"',
                    f"page_size = {page_size}",
                    "",
                    "[store]",
                    f'path = "{tmp_path / "reid.sqlite3"}"',
                    "",
                ]
            )
        )
        return config_path

    return _make


def run_batch(config_path: Path) -> Result:
    """Invoke the real batch entry point the way an operator schedules it."""
    return CliRunner().invoke(batch, ["--config", str(config_path)], catch_exceptions=False)
