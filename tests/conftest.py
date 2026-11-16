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
def detection_stub() -> Iterator[StubImmich]:
    """The full fixture registry: every photo the detection tests need."""
    server = StubImmich(library.detection_registry())
    server.people = library.people_registry()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def ui_stub() -> Iterator[StubImmich]:
    """Stub with people for UI tests."""
    server = StubImmich(library.detection_registry())
    server.people = library.people_registry()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[[StubImmich, int], Path]:
    """Write a reid.toml pointing a run at a stub Immich with a fresh store path."""

    weights_dir = Path(__file__).parent.parent / "src" / "reid" / "weights"

    def _make(
        server: StubImmich,
        page_size: int = 1000,
        detection: dict | None = None,
        clustering: dict | None = None,
        name: str = "reid",
    ) -> Path:
        config_path = tmp_path / f"{name}.toml"
        lines = [
            "[immich]",
            f'base_url = "{server.base_url}"',
            f'api_key = "{API_KEY}"',
            f"page_size = {page_size}",
            "",
            "[store]",
            f'path = "{tmp_path / f"{name}.sqlite3"}"',
            "",
            "[head_detection]",
            f'weights_path = "{weights_dir / "scrfd_500m.onnx"}"',
            "",
        ]
        if detection is not None:
            lines += ["[detection]"]
            lines += [f"{key} = {value!r}" for key, value in detection.items()]
            lines += [""]
        if clustering is not None:
            lines += ["[clustering]"]
            lines += [f"{key} = {value!r}" for key, value in clustering.items()]
            lines += [""]
        config_path.write_text("\n".join(lines))
        return config_path

    return _make


def run_batch(config_path: Path) -> Result:
    """Invoke the real batch entry point the way an operator schedules it."""
    return CliRunner().invoke(
        batch, ["--config", str(config_path)], catch_exceptions=False
    )
