"""Config file support: reid.toml, TOML-parsed, with env overrides for secrets.

UI tuning knobs arrive in later tickets; for now the config file holds the
Immich connection, the local store location, and run log settings.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_KEY_ENV = "IMMICH_API_KEY"
WRITE_API_KEY_ENV = "IMMICH_WRITE_API_KEY"
CORE_URL_ENV = "REID_CORE_URL"
CORE_KEY_ENV = "REID_CORE_API_KEY"

MAX_PAGE_SIZE = 1000

LOG_LEVELS = ("debug", "info", "warning", "error")

# Head detector weights vendored into the package (same dir batch.py uses for
# the detector and embedder weights).
_PACKAGED_HEAD_WEIGHTS = str(Path(__file__).parent / "weights" / "scrfd_500m.onnx")


class ConfigError(Exception):
    """The config file is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class ImmichConfig:
    base_url: str
    api_key: str
    write_api_key: str | None = None  # write-scoped key for confirm/promote
    page_size: int = MAX_PAGE_SIZE
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class CoreConfig:
    """The core API server (reid-core), which owns the store.

    The UI proxies /api/* to `url`. `api_key` gates the core when set:
    empty disables the check (fine for local dev), deployments should set
    it via REID_CORE_API_KEY or the config file.
    """

    url: str = "http://127.0.0.1:8090"
    api_key: str = ""
    # Bind address for reid-core itself. Loopback by default: the UI proxies
    # /api/*, and the nightly trigger reaches the core on the same machine.
    # Set a non-loopback address only together with an api_key.
    host: str = "127.0.0.1"
    port: int = 8090


@dataclass(frozen=True)
class UiConfig:
    """What the UI server needs: where the core is, and nothing else.

    The UI has no store and no Immich client, so it does not need the full
    config file. A laptop can run it with a [core]-only reid.toml (or no
    file at all, via the reid-ui CLI flags).
    """

    core: CoreConfig
    # Override for the Immich URL used by "open in Immich" links: the core
    # reports its own base_url, which may be a localhost address that means
    # nothing on another machine. When unset, load_ui_config falls back to
    # [immich] base_url so you only enter the URL twice when they actually
    # differ (UI on another machine).
    immich_url: str | None = None


@dataclass(frozen=True)
class DetectionConfig:
    """Quality gates and the pixel source: spec tuning knobs."""

    pixel_source: str = "preview"  # Immich preview thumbnail, or "original"
    confidence_floor: float = 0.35
    min_box_height: float = 100.0


@dataclass(frozen=True)
class HeadDetectionConfig:
    """SCRFD head detector configuration: spec tuning knobs."""

    weights_path: str
    input_size: int = 320
    confidence_threshold: float = 0.5
    top_expansion_ratio: float = 0.55
    width_expansion_ratio: float = 1.8
    nms_threshold: float = 0.4


PIXEL_SOURCES = ("preview", "original")


@dataclass(frozen=True)
class ClusteringConfig:
    """Clustering and anchoring thresholds: spec tuning knobs."""

    window_days: int = 1  # 1 = same calendar day
    distance_threshold: float = 0.5  # cosine distance; to calibrate
    min_cluster_size: int = 2  # ask-for-name queue minimum
    face_overlap: float = 0.5  # intersection / face-box-area minimum


@dataclass(frozen=True)
class RunConfig:
    log_level: str = "info"


@dataclass(frozen=True)
class Config:
    immich: ImmichConfig
    store_path: Path
    detection: DetectionConfig
    head_detection: HeadDetectionConfig
    clustering: ClusteringConfig
    run: RunConfig
    core: CoreConfig


def load_config(config_path: Path, *, env: Mapping[str, str] | None = None) -> Config:
    """Load and validate a reid.toml file.

    The Immich API keys may be omitted from the file and supplied via the
    IMMICH_API_KEY / IMMICH_WRITE_API_KEY environment variables instead; the
    env vars win when both are set, so secrets need not live in config files.
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
        raise ConfigError(
            f"config file is not valid TOML: {config_path}: {exc}"
        ) from exc

    immich_raw = _table(raw, "immich")
    base_url = _required_str(immich_raw, "base_url", "[immich]")
    page_size = _bounded_int(
        immich_raw, "page_size", 1, MAX_PAGE_SIZE, "[immich]", MAX_PAGE_SIZE
    )
    timeout_seconds = _positive_float(immich_raw, "timeout_seconds", "[immich]", 60.0)

    api_key = env.get(API_KEY_ENV, "").strip() or _required_str(
        immich_raw, "api_key", "[immich]"
    )
    write_api_key = env.get(WRITE_API_KEY_ENV, "").strip() or immich_raw.get(
        "write_api_key"
    )
    if write_api_key is not None and not isinstance(write_api_key, str):
        raise ConfigError("[immich] write_api_key must be a string")

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
    confidence_floor = _unit_float(
        detection_raw, "confidence_floor", "[detection]", 0.35
    )
    min_box_height = _min_box_height(detection_raw)

    clustering_raw = _table(raw, "clustering")
    window_days = _minimum_int(clustering_raw, "window_days", 1, "[clustering]", 1)
    distance_threshold = _float_in_range(
        clustering_raw,
        "distance_threshold",
        0.0,
        None,
        "[clustering]",
        0.5,
        "must be a non-negative number",
    )
    min_cluster_size = _minimum_int(
        clustering_raw, "min_cluster_size", 1, "[clustering]", 2
    )
    face_overlap = _unit_float(clustering_raw, "face_overlap", "[clustering]", 0.5)

    head_detection_raw = _table(raw, "head_detection")
    head_weights_path = _optional_str(
        head_detection_raw,
        "weights_path",
        "[head_detection]",
        _PACKAGED_HEAD_WEIGHTS,
    )
    head_input_size = _minimum_int(
        head_detection_raw, "input_size", 1, "[head_detection]", 320
    )
    head_confidence_threshold = _unit_float(
        head_detection_raw, "confidence_threshold", "[head_detection]", 0.5
    )
    head_top_expansion_ratio = _positive_float(
        head_detection_raw, "top_expansion_ratio", "[head_detection]", 0.55
    )
    head_width_expansion_ratio = _positive_float(
        head_detection_raw, "width_expansion_ratio", "[head_detection]", 1.8
    )
    head_nms_threshold = _unit_float(
        head_detection_raw, "nms_threshold", "[head_detection]", 0.4
    )

    core = _parse_core(raw, env)

    return Config(
        immich=ImmichConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            write_api_key=write_api_key,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        ),
        store_path=store_path,
        detection=DetectionConfig(
            pixel_source=pixel_source,
            confidence_floor=confidence_floor,
            min_box_height=min_box_height,
        ),
        head_detection=HeadDetectionConfig(
            weights_path=head_weights_path,
            input_size=head_input_size,
            confidence_threshold=head_confidence_threshold,
            top_expansion_ratio=head_top_expansion_ratio,
            width_expansion_ratio=head_width_expansion_ratio,
            nms_threshold=head_nms_threshold,
        ),
        clustering=ClusteringConfig(
            window_days=window_days,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            face_overlap=face_overlap,
        ),
        run=RunConfig(log_level=log_level),
        core=core,
    )


def load_ui_config(
    config_path: Path | None, *, env: Mapping[str, str] | None = None
) -> UiConfig:
    """Load only what `reid-ui` needs: the [core] section and [ui] overrides.

    A full reid.toml works unchanged; the Immich section is only consulted
    for [immich] base_url as a fallback when [ui] immich_url is unset, so
    viewer links work without entering the URL twice. With no path, defaults
    apply (localhost core, no token), and environment variables still win
    over file values.
    """
    env = os.environ if env is None else env
    raw: dict[str, Any] = {}
    if config_path is not None:
        try:
            with config_path.open("rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {config_path}") from exc
        except OSError as exc:
            raise ConfigError(
                f"config file not readable: {config_path}: {exc}"
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"config file is not valid TOML: {config_path}: {exc}"
            ) from exc

    ui_raw = raw.get("ui", {})
    if not isinstance(ui_raw, dict):
        raise ConfigError("[ui] must be a table")
    immich_url = ui_raw.get("immich_url")
    if immich_url is not None:
        if not isinstance(immich_url, str) or not immich_url:
            raise ConfigError("[ui] immich_url must be a non-empty string")
        immich_url = immich_url.rstrip("/")
    else:
        # Fall back to the core's Immich base_url so viewer links work
        # without entering the URL twice. A UI on another machine can still
        # override with [ui] immich_url when the core reports localhost.
        immich_raw = raw.get("immich", {})
        if isinstance(immich_raw, dict):
            base_url = immich_raw.get("base_url")
            if isinstance(base_url, str) and base_url:
                immich_url = base_url.rstrip("/")

    return UiConfig(core=_parse_core(raw, env), immich_url=immich_url)


def _parse_core(raw: Mapping[str, Any], env: Mapping[str, str]) -> CoreConfig:
    """Parse the [core] section, with env overrides for URL and token."""
    core_raw = raw.get("core", {})
    if not isinstance(core_raw, dict):
        raise ConfigError("[core] must be a table")
    core_url = env.get(CORE_URL_ENV, "").strip() or core_raw.get(
        "url", "http://127.0.0.1:8090"
    )
    if not isinstance(core_url, str) or not core_url:
        raise ConfigError("[core] url must be a non-empty string")
    core_api_key = env.get(CORE_KEY_ENV, "").strip() or core_raw.get("api_key", "")
    if not isinstance(core_api_key, str):
        raise ConfigError("[core] api_key must be a string")
    core_host = core_raw.get("host", "127.0.0.1")
    if not isinstance(core_host, str) or not core_host:
        raise ConfigError("[core] host must be a non-empty string")
    core_port = _bounded_int(core_raw, "port", 1, 65535, "[core]", 8090)
    return CoreConfig(
        url=core_url.rstrip("/"),
        api_key=core_api_key,
        host=core_host,
        port=core_port,
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


def _optional_str(table: dict[str, Any], key: str, section: str, default: str) -> str:
    """Read a string config value with a default; empty/invalid values raise."""
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section} {key} must be a non-empty string")
    return value.strip()


def _bounded_int(
    table: dict[str, Any], key: str, low: int, high: int, section: str, default: int
) -> int:
    value = table.get(key, default)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not low <= value <= high
    ):
        raise ConfigError(f"{section} {key} must be an integer in [{low}, {high}]")
    return value


def _positive_float(
    table: dict[str, Any], key: str, section: str, default: float
) -> float:
    value = table.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{section} {key} must be a positive number of seconds")
    return float(value)


def _float_in_range(
    table: dict[str, Any],
    key: str,
    low: float,
    high: float | None,
    section: str,
    default: float,
    message: str,
) -> float:
    value = table.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    in_range = (
        isinstance(value, (int, float))
        and value >= low
        and (high is None or value <= high)
    )
    if not in_range:
        raise ConfigError(f"{section} {key} {message}")
    return float(value)


def _unit_float(table: dict[str, Any], key: str, section: str, default: float) -> float:
    return _float_in_range(
        table, key, 0.0, 1.0, section, default, "must be a number in [0, 1]"
    )


def _minimum_int(
    table: dict[str, Any], key: str, low: int, section: str, default: int
) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise ConfigError(f"{section} {key} must be an integer of at least {low}")
    return value


def _log_level(value: Any) -> str:
    if not isinstance(value, str) or value.lower() not in LOG_LEVELS:
        raise ConfigError(f"[run] log_level must be one of {', '.join(LOG_LEVELS)}")
    return value.lower()


def _min_box_height(detection_raw: dict[str, Any]) -> float:
    value = detection_raw.get("min_box_height", 100)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(
            "[detection] min_box_height must be a positive number of pixels"
        )
    return float(value)
