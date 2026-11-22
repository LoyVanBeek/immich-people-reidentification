"""The `reid-trigger` entry point: ask the core to start one pipeline run.

The nightly systemd timer calls this instead of running the pipeline
directly: the core owns the store and the job manager, so it is the only
process that may start a run. This command is a thin HTTP client — it
posts to the core's /api/jobs endpoint, optionally waits for the run to
finish, and reports the outcome to the journal.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import httpx

from .config import ConfigError, load_config

_POLL_SECONDS = 5.0


def _post_job(base_url: str, api_key: str, trigger: str) -> dict[str, object]:
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/jobs",
            json={"trigger": trigger},
            headers=headers,
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise click.ClickException(
            f"core is unreachable at {base_url}: {exc} — is reid-core running?"
        ) from exc
    if response.status_code != 200:
        raise click.ClickException(
            f"core refused the job (HTTP {response.status_code}): {response.text}"
        )
    return response.json()


def _wait_for_run(
    base_url: str, api_key: str, job_id: str, timeout: float
) -> dict[str, object]:
    """Poll /api/jobs until this run leaves the running state."""
    headers = {"x-api-key": api_key} if api_key else {}
    deadline = time.monotonic() + timeout
    last_windows = -1
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}/api/jobs", headers=headers, timeout=30.0
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise click.ClickException(
                f"lost contact with the core while waiting: {exc}"
            ) from exc
        run = response.json().get("last_run") or {}
        if run.get("id") != job_id:
            # The core restarted and no longer knows this job.
            raise click.ClickException(
                "the core restarted before the run finished — check the UI"
            )
        if run.get("windows_done", 0) != last_windows:
            last_windows = int(run.get("windows_done", 0))
            click.echo(f"windows done: {last_windows}")
        if run.get("state") != "running":
            return run
        time.sleep(_POLL_SECONDS)
    raise click.ClickException(f"run {job_id} did not finish within {timeout:.0f}s")


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the reid.toml config file.",
)
@click.option(
    "--trigger",
    default="timer",
    show_default=True,
    help="Recorded on the run (timer, manual, ...).",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for the run to finish and exit non-zero if it failed.",
)
@click.option(
    "--timeout",
    default=6 * 60 * 60.0,
    show_default=True,
    help="Seconds to wait for the run when --wait is set.",
)
def main(config_path: Path, trigger: str, wait: bool, timeout: float) -> None:
    """Ask the core to start one pipeline run."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    base_url = cfg.core.url
    api_key = cfg.core.api_key

    result = _post_job(base_url, api_key, trigger)
    if "error" in result:
        raise click.ClickException(str(result["error"]))
    job_id = str(result["job_id"])
    click.echo(f"started run {job_id} (pid {result.get('pid')})")

    if not wait:
        return

    run = _wait_for_run(base_url, api_key, job_id, timeout)
    state = str(run.get("state"))
    click.echo(f"run {job_id} finished: {state}")
    if state != "done":
        if run.get("error"):
            click.echo(f"error: {run['error']}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
