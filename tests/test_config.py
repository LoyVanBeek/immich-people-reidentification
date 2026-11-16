"""Config tests: the clustering tuning knobs parse and validate."""

from __future__ import annotations

from pathlib import Path

import pytest
from reid.config import ClusteringConfig, ConfigError, load_config


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
