"""Config tests: the clustering tuning knobs parse and validate."""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.config import (
    ClusteringConfig,
    ConfigError,
    UiConfig,
    load_config,
    load_ui_config,
)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "reid.toml"
    path.write_text(body)
    return path


MINIMAL = """
[immich]
base_url = "http://immich:2283"
api_key = "k"

[store]
path = "store.sqlite3"

[head_detection]
weights_path = "/home/loy/git/immich-person-reidentification/src/reid/weights/scrfd_500m.onnx"
"""


def clustering_defaults(tmp_path: Path) -> ClusteringConfig:
    return load_config(write_config(tmp_path, MINIMAL)).clustering


def test_clustering_defaults_match_the_spec(tmp_path: Path) -> None:
    clustering = clustering_defaults(tmp_path)
    assert clustering.window_days == 1  # same calendar day
    assert clustering.distance_threshold == 0.5  # to calibrate
    assert clustering.min_cluster_size == 2
    assert clustering.face_overlap == 0.5


def test_clustering_knobs_parse_from_the_config_file(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            MINIMAL
            + """
[clustering]
window_days = 3
distance_threshold = 0.35
min_cluster_size = 5
face_overlap = 0.6
""",
        )
    )
    assert config.clustering.window_days == 3
    assert config.clustering.distance_threshold == 0.35
    assert config.clustering.min_cluster_size == 5
    assert config.clustering.face_overlap == 0.6


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("window_days = 0\n", "window_days"),
        ('window_days = "soon"\n', "window_days"),
        ("distance_threshold = -0.1\n", "distance_threshold"),
        ('distance_threshold = "loose"\n', "distance_threshold"),
        ("min_cluster_size = 0\n", "min_cluster_size"),
        ("face_overlap = 1.5\n", "face_overlap"),
        ("face_overlap = -0.1\n", "face_overlap"),
    ],
)
def test_invalid_clustering_knobs_are_rejected(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            write_config(
                tmp_path,
                MINIMAL + f"\n[clustering]\n{body}",
            )
        )
    assert message in str(excinfo.value)


# --- [core] host/port -------------------------------------------------------


def test_core_host_port_default_loopback(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL))
    assert config.core.host == "127.0.0.1"
    assert config.core.port == 8090


def test_core_host_port_parse_from_file(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            MINIMAL
            + """
[core]
host = "0.0.0.0"
port = 8091
""",
        )
    )
    assert config.core.host == "0.0.0.0"
    assert config.core.port == 8091


def test_core_port_rejects_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            write_config(
                tmp_path,
                MINIMAL + "\n[core]\nport = 70000\n",
            )
        )
    assert "port" in str(excinfo.value)


# --- load_ui_config ----------------------------------------------------------


def CORE_ONLY_BODY() -> str:
    return """
[core]
url = "http://core.local:8090"
api_key = "abc123"
host = "0.0.0.0"
port = 8090
"""


def test_load_ui_config_reads_only_core_section(tmp_path: Path) -> None:
    cfg = load_ui_config(
        write_config(
            tmp_path,
            CORE_ONLY_BODY() + '[immich]\nbase_url = "http://x"\napi_key = "k"\n',
        )
    )
    assert isinstance(cfg, UiConfig)
    assert cfg.core.url == "http://core.local:8090"
    assert cfg.core.api_key == "abc123"
    assert cfg.core.host == "0.0.0.0"
    assert cfg.core.port == 8090
    # immich_url falls back to [immich] base_url when [ui] immich_url unset
    assert cfg.immich_url == "http://x"


def test_load_ui_config_immich_url_explicit_wins_over_base_url(
    tmp_path: Path,
) -> None:
    cfg = load_ui_config(
        write_config(
            tmp_path,
            CORE_ONLY_BODY()
            + """
[immich]
base_url = "http://x"

[ui]
immich_url = "http://immich.laptop:2283"
""",
        )
    )
    assert cfg.immich_url == "http://immich.laptop:2283"


def test_load_ui_config_immich_url_none_when_neither_set(tmp_path: Path) -> None:
    cfg = load_ui_config(write_config(tmp_path, CORE_ONLY_BODY()))
    assert cfg.immich_url is None


def test_load_ui_config_with_ui_immich_url(tmp_path: Path) -> None:
    cfg = load_ui_config(
        write_config(
            tmp_path,
            CORE_ONLY_BODY()
            + """
[ui]
immich_url = "http://immich.laptop:2283"
""",
        )
    )
    assert cfg.immich_url == "http://immich.laptop:2283"


def test_load_ui_config_no_path_defaults(tmp_path: Path) -> None:
    cfg = load_ui_config(None)
    assert cfg.core.url == "http://127.0.0.1:8090"
    assert cfg.core.api_key == ""
    assert cfg.core.host == "127.0.0.1"
    assert cfg.core.port == 8090
    assert cfg.immich_url is None


def test_load_ui_config_env_precedes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REID_CORE_URL", "http://env-core:9000")
    monkeypatch.setenv("REID_CORE_API_KEY", "env-key")
    cfg = load_ui_config(write_config(tmp_path, CORE_ONLY_BODY()))
    assert cfg.core.url == "http://env-core:9000"
    assert cfg.core.api_key == "env-key"


def test_load_ui_config_invalid_immich_url_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_ui_config(
            write_config(
                tmp_path,
                CORE_ONLY_BODY() + '\n[ui]\nimmich_url = ""\n',
            )
        )


def test_load_ui_config_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(ConfigError):
        load_ui_config(missing)


# --- Immich API key env overrides --------------------------------------------


def test_write_api_key_env_overrides_file(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        MINIMAL.replace(
            'api_key = "k"', 'api_key = "k"\nwrite_api_key = "file-write-key"'
        ),
    )
    cfg = load_config(path, env={"IMMICH_WRITE_API_KEY": "env-write-key"})
    assert cfg.immich.write_api_key == "env-write-key"
    cfg2 = load_config(path, env={})
    assert cfg2.immich.write_api_key == "file-write-key"


def test_api_key_env_still_wins(tmp_path: Path) -> None:
    cfg = load_config(
        write_config(tmp_path, MINIMAL), env={"IMMICH_API_KEY": "env-key"}
    )
    assert cfg.immich.api_key == "env-key"
    cfg2 = load_config(write_config(tmp_path, MINIMAL), env={})
    assert cfg2.immich.api_key == "k"
