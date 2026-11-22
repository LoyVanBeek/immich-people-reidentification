"""Tests for `reid-trigger`: the nightly timer's one-line job trigger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from reid.trigger import main


def _config(tmp_path: Path, api_key: str = "secret") -> Path:
    path = tmp_path / "trigger.toml"
    path.write_text(
        f"""
[immich]
base_url = "http://immich.test"
api_key = "read-key"

[store]
path = "{tmp_path / "trigger.sqlite3"}"

[core]
url = "http://127.0.0.1:9999"
api_key = "{api_key}"
"""
    )
    return path


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError("boom")


def test_trigger_posts_a_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_post(
        url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> FakeResponse:
        calls.append((url, json, headers))
        return FakeResponse(200, {"job_id": "run-1", "pid": 42})

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "started run run-1" in result.output
    url, body, headers = calls[0]
    assert url == "http://127.0.0.1:9999/api/jobs"
    assert body == {"trigger": "timer"}
    assert headers == {"x-api-key": "secret"}


def test_trigger_without_api_key_sends_no_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, str]] = []

    def fake_post(
        url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> FakeResponse:
        seen.append(headers)
        return FakeResponse(200, {"job_id": "run-1", "pid": 42})

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path, api_key=""))])

    assert result.exit_code == 0, result.output
    assert seen == [{}]


def test_trigger_reports_a_refused_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(409, {"error": "a pipeline run is already in progress"})

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path))])

    assert result.exit_code == 1
    assert "already in progress" in result.output


def test_trigger_reports_an_unreachable_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path))])

    assert result.exit_code == 1
    assert "unreachable" in result.output
    assert "reid-core" in result.output


def test_wait_follows_the_run_to_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    states = [
        {"id": "run-1", "state": "running", "windows_done": 1},
        {"id": "run-1", "state": "running", "windows_done": 2},
        {"id": "run-1", "state": "done", "windows_done": 2},
    ]

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"job_id": "run-1", "pid": 42})

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        return FakeResponse(200, {"active": len(states) > 1, "last_run": states.pop(0)})

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    monkeypatch.setattr("reid.trigger.httpx.get", fake_get)
    monkeypatch.setattr("reid.trigger._POLL_SECONDS", 0.0)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path)), "--wait"])

    assert result.exit_code == 0, result.output
    assert "windows done: 1" in result.output
    assert "windows done: 2" in result.output
    assert "finished: done" in result.output


def test_wait_exits_nonzero_when_the_run_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"job_id": "run-1", "pid": 42})

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "active": False,
                "last_run": {
                    "id": "run-1",
                    "state": "failed",
                    "windows_done": 0,
                    "error": "pipeline exited with code 1",
                },
            },
        )

    monkeypatch.setattr("reid.trigger.httpx.post", fake_post)
    monkeypatch.setattr("reid.trigger.httpx.get", fake_get)
    result = CliRunner().invoke(main, ["--config", str(_config(tmp_path)), "--wait"])

    assert result.exit_code == 1
    assert "pipeline exited with code 1" in result.output
