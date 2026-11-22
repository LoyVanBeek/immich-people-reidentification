"""Acceptance tests for ticket 09: the batch container and checkpointed feed.

All tests drive the real batch entry point (`reid-batch --config …`) against
the stub Immich HTTP server, which records every request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from click.testing import CliRunner, Result

from reid.cli import batch
from reid.config import API_KEY_ENV, load_config
from reid.store import Store

from . import library
from .conftest import run_batch
from .stub_immich import StubImmich


def test_first_run_fetches_all_new_photos_and_advances_the_checkpoint(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    result = run_batch(config)
    assert isinstance(result, Result)

    assert result.exit_code == 0, result.output
    for asset_id in library.photo_ids():
        assert asset_id in result.output, f"missing {asset_id} in run output"
    # Check for checkpoint with milliseconds (isoformat_z includes .000Z)
    assert library.PHOTO_MAX_CREATED_AT in result.output, (
        "checkpoint not reported in run output"
    )

    first_request = stub.search_requests()[0]
    body = first_request.body
    assert body["type"] == "IMAGE", "feed must fetch photos only"
    assert body["order"] == "asc"
    assert "createdAfter" not in body or body["createdAfter"] is None, (
        "first run has no checkpoint"
    )


def test_second_consecutive_run_processes_zero_assets(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    first = run_batch(config)
    assert first.exit_code == 0, first.output
    second = run_batch(config)
    assert second.exit_code == 0, second.output

    first_request, second_request = stub.search_requests()
    assert first_request.body.get("createdAfter") is None
    assert second_request.body["createdAfter"] == library.PHOTO_MAX_CREATED_AT, (
        "second run must resume from the persisted checkpoint"
    )

    assert "assets fetched=0" in second.output, second.output
    for asset_id in library.photo_ids():
        assert asset_id not in second.output, f"{asset_id} was reprocessed"


def test_only_photos_are_fetched_videos_are_excluded(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    result = run_batch(config)

    assert result.exit_code == 0, result.output
    assert library.VIDEO_1_ID not in result.output
    assert "assets fetched=3" in result.output, result.output


def test_every_request_carries_the_read_scoped_api_key(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    result = run_batch(config)

    assert result.exit_code == 0, result.output
    for recorded in stub.requests:
        assert recorded.headers.get("x-api-key") == "reid-read-scoped-test-key", (
            f"request {recorded.method} {recorded.path} did not carry the read-scoped key"
        )


def test_crashed_run_resumes_from_the_checkpoint(
    stub: StubImmich, make_config: Any
) -> None:
    server = StubImmich(library.extended_asset_registry())
    server.start()
    try:
        config = make_config(server, page_size=2)  # seven photos -> four pages

        first = run_batch(config)
        assert first.exit_code == 0, first.output
        # Check for checkpoint with milliseconds (isoformat_z includes .000Z)
        assert library.EXTENDED_PHOTO_MAX_CREATED_AT in first.output

        # Three new photos land after the first run; then the feed dies mid-run.
        server.assets.extend(library.late_photos())
        server.fail_search_pages = {2}

        crashed = run_batch(config)
        assert crashed.exit_code != 0, crashed.output
        assert "batch run complete" not in crashed.output

        server.fail_search_pages = set()

        resumed = run_batch(config)
        assert resumed.exit_code == 0, resumed.output
        for late in library.late_photos():
            assert late["id"] in resumed.output
            assert late["id"] not in first.output

        again = run_batch(config)
        assert again.exit_code == 0, again.output
        assert "assets fetched=0" in again.output
    finally:
        server.stop()

    resumed_request = server.search_requests()[-2]
    assert (
        resumed_request.body["createdAfter"] == library.EXTENDED_PHOTO_MAX_CREATED_AT
    ), "the crashed run must resume from the checkpoint, not from scratch"
    again_request = server.search_requests()[-1]
    assert again_request.body["createdAfter"] == library.PHOTO_10_CREATED_AT


def test_api_key_env_var_overrides_the_config_file(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    result = CliRunner().invoke(
        batch,
        ["--config", str(config)],
        env={API_KEY_ENV: "env-supplied-key"},
        catch_exceptions=False,
    )

    assert result.exit_code != 0, "the stub rejects a key that is not registered"
    for recorded in stub.search_requests():
        assert recorded.headers.get("x-api-key") == "env-supplied-key"


def test_missing_config_file_fails_cleanly(stub: StubImmich, make_config: Any) -> None:
    missing = make_config(stub)
    missing.unlink()

    result = run_batch(missing)

    assert result.exit_code != 0
    assert "error" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.output


def test_fixture_photos_are_served_through_the_asset_endpoints(
    stub: StubImmich, make_config: Any
) -> None:
    config = make_config(stub)

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    import httpx

    for asset_id in library.photo_ids():
        expected = library.asset_bytes(asset_id)
        for endpoint in (
            f"/api/assets/{asset_id}/original",
            f"/api/assets/{asset_id}/thumbnail",
        ):
            response = httpx.get(
                f"{stub.base_url}{endpoint}",
                headers={"x-api-key": "reid-read-scoped-test-key"},
            )
            assert response.status_code == 200, endpoint
            assert response.headers["content-type"] == "image/jpeg"
            assert response.content == expected, f"fixture bytes differ for {endpoint}"


def test_run_reports_window_progress_to_the_job_manager(
    stub: StubImmich, make_config: Any, monkeypatch: Any
) -> None:
    """A core-launched run records how many windows it has finished.

    The UI polls this to show progress; without REID_RUN_ID the same code
    path does nothing (ops runs keep working unchanged).
    """
    config = make_config(stub)
    cfg = load_config(config)
    Store(cfg.store_path).start_run("run-1", "timer", None, datetime.now(UTC))
    monkeypatch.setenv("REID_RUN_ID", "run-1")

    result = run_batch(config)
    assert result.exit_code == 0, result.output

    windows = [
        line for line in result.output.splitlines() if line.startswith("window ")
    ]
    assert windows, "run reported no windows"
    run = Store(cfg.store_path).latest_run()
    assert run is not None
    assert run["windows_done"] == len(windows)
