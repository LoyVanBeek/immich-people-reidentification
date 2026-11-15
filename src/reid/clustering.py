"""Pure clustering rules: cosine distance, average linkage, timespan windows.

A cluster is one person's appearance-group within one window — not a
persistent identity; clusters are rebuilt from scratch every run and never
merged across windows. Window membership uses the asset capture time
(EXIF) in the local timezone, falling back to the upload time when no
capture time is recorded.

Distance: cosine distance over the L2-normalized embeddings, clustered
with agglomerative hierarchical clustering (average linkage) cut at a
configurable threshold. Appearances that join no multi-appearance cluster
remain singletons.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo

import numpy as np


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances (1 - cosine similarity) in [0, 2].

    Rows are re-normalized defensively so unnormalized input cannot push
    distances out of range; identical rows still yield exactly 0.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.where(norms == 0.0, 1.0, norms)
    similarity = unit @ unit.T
    return 1.0 - similarity


def average_linkage_clusters(distance: np.ndarray, threshold: float) -> list[tuple[int, ...]]:
    """Agglomerative average-linkage clustering, cut at the threshold.

    Starts from single appearances and repeatedly merges the two clusters
    with the smallest mean pairwise distance while that distance is within
    the threshold. Ties merge the first pair in scan order, so the result
    is deterministic.
    """
    count = distance.shape[0]
    clusters: list[list[int]] = [[index] for index in range(count)]
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = 0.0
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                mean = float(distance[np.ix_(clusters[a], clusters[b])].mean())
                if best_pair is None or mean < best_distance:
                    best_pair, best_distance = (a, b), mean
        if best_pair is None or best_distance > threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return [tuple(sorted(members)) for members in clusters]


def capture_day(captured_at: str, tz: tzinfo | None = None) -> date | None:
    """The local calendar day an asset was captured, or None if unusable.

    A timestamp with a timezone is converted into the given timezone (the
    Pi's local timezone when omitted); a naive timestamp is treated as
    already-local wall time. Unparseable strings yield None.
    """
    try:
        moment = datetime.fromisoformat(captured_at)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.date()
    local = moment.astimezone(tz) if tz is not None else moment.astimezone()
    return local.date()


def partition_windows(days: Iterable[date], window_days: int) -> dict[date, str]:
    """Assign each capture day to a timespan window of at most window_days.

    Windows are blocks of consecutive calendar days, filled greedily from
    the earliest day onward; the window key is its first day (ISO). With
    the default window_days = 1 every day is its own window (the same
    calendar day).
    """
    remaining = sorted(set(days))
    assignment: dict[date, str] = {}
    index = 0
    while index < len(remaining):
        start = remaining[index]
        end = start + timedelta(days=window_days)
        while index < len(remaining) and remaining[index] < end:
            assignment[remaining[index]] = start.isoformat()
            index += 1
    return assignment
