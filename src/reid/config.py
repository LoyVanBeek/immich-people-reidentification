"""Config file support: reid.toml, TOML-parsed, with env overrides for secrets.

UI tuning knobs arrive in later tickets; for now the config file holds the
Immich connection, the local store location, and run log settings.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

API_KEY_ENV = "IMMICH_API_KEY"

MAX_PAGE_SIZE = 1000

LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(Exception):
    """The config file is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class ImmichConfig:
    base_url: str
    api_key: str
    page_size: int = MAX_PAGE_SIZE
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class DetectionConfig:
    """Quality gates and the pixel source: spec tuning knobs."""

    pixel_source: str = "preview"  # Immich preview thumbnail, or "original"
    confidence_floor: float = 0.35
    min_box_height: float = 100.0


PIXEL_SOURCES = ("preview", "original")


@dataclass(frozen=True)
class RunConfig:
    log_level: str = "info"


@dataclass(frozen=True)
class Config:
    immich: ImmichConfig
    store_path: Path
    detection: DetectionConfig
    run: RunConfig


def load_config(config_path: Path, *, env: Mapping[str, str] | None = None) -> Config:
    """Load and validate a reid.toml file.

    The Immich API key may be omitted from the file and supplied via the
    IMMICH_API_KEY environment variable instead; the env var wins when both
    are set, so secrets need not live in config files.
    """
    env = os.environ if env is None else env
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"config file not readable: {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file is not valid TOML: {config_path}: {exc}") from exc

    immich_raw = _table(raw, "immich")
    base_url = _required_str(immich_raw, "base_url", "[immich]")
    page_size = _bounded_int(immich_raw, "page_size", 1, MAX_PAGE_SIZE, "[immich]", MAX_PAGE_SIZE)
    timeout_seconds = _positive_float(immich_raw, "timeout_seconds", "[immich]", 60.0)

    api_key = env.get(API_KEY_ENV, "").strip() or _required_str(immich_raw, "api_key", "[immich]")

    store_raw = _table(raw, "store")
    store_path = Path(_required_str(store_raw, "path", "[store]"))

    run_raw = _table(raw, "run")
    log_level = _log_level(run_raw.get("log_level", "info"))

    detection_raw = _table(raw, "detection")
    pixel_source = detection_raw.get("pixel_source", "preview")
    if not isinstance(pixel_source, str) or pixel_source not in PIXEL_SOURCES:
        raise ConfigError(
            f"[detection] pixel_source must be one of {', '.join(PIXEL_SOURCES)}"
        )
    confidence_floor = _confidence_floor(detection_raw)
    min_box_height = _min_box_height(detection_raw)

    return Config(
        immich=ImmichConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        ),
        store_path=store_path,
        detection=DetectionConfig(
            pixel_source=pixel_source,
            confidence_floor=confidence_floor,
            min_box_height=min_box_height,
        ),
        run=RunConfig(log_level=log_level),
    )


# -- parsing helpers --------------------------------------------------------


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _required_str(table: dict[str, Any], key: str, section: str) -> str:
    value = table.get(key)
    if value is None:
        raise ConfigError(f"{section} {key} is required")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section} {key} must be a non-empty string")
    return value.strip()


def _bounded_int(table: dict[str, Any], key: str, low: int, high: int, section: str, default: int) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ConfigError(f"{section} {key} must be an integer in [{low}, {high}]")
    return value


def _positive_float(table: dict[str, Any], key: str, section: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{section} {key} must be a positive number of seconds")
    return float(value)


def _log_level(value: Any) -> str:
    if not isinstance(value, str) or value.lower() not in LOG_LEVELS:
        raise ConfigError(f"[run] log_level must be one of {', '.join(LOG_LEVELS)}")
    return value.lower()


def _confidence_floor(detection_raw: dict[str, Any]) -> float:
    value = detection_raw.get("confidence_floor", 0.35)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ConfigError("[detection] confidence_floor must be a number in [0, 1]")
    return float(value)


def _min_box_height(detection_raw: dict[str, Any]) -> float:
    value = detection_raw.get("min_box_height", 100)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError("[detection] min_box_height must be a positive number of pixels")
    return float(value)
