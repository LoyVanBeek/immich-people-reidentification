"""The batch run: fetch new assets, embed every person appearance.

The checkpoint stores the newest asset createdAt seen in a fully completed
run; the next run fetches strictly newer assets. The checkpoint only advances
after the whole feed has been processed, so a crashed run resumes from the
last checkpoint instead of reprocessing the library. Appearances are stored
keyed by asset and appearance, so reprocessed assets replace their rows
idempotently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .detector import PersonDetector
from .embedder import PersonEmbedder
from .immich import Asset, ImmichClient
from .pipeline import GateConfig, GateCounts, process_asset
from .store import NEW_ASSET_FEED, Appearance, Store

_log = logging.getLogger("reid.batch")

WEIGHTS_DIR = Path(__file__).parent / "weights"

YOLO11N_WEIGHTS = WEIGHTS_DIR / "yolo11n.pt"
OSNET_AIN_WEIGHTS = WEIGHTS_DIR / "osnet_ain_x1_0_combineall.pth"


@dataclass(frozen=True)
class RunSummary:
    assets: tuple[Asset, ...]
    checkpoint_before: str | None
    checkpoint_after: str | None
    appearances: int
    gate_counts: GateCounts
    edge_clipped_kept: int
    assets_without_persons: int


def run_batch(cfg: Config) -> RunSummary:
    store = Store(cfg.store_path)
    client = ImmichClient(
        cfg.immich.base_url,
        cfg.immich.api_key,
        page_size=cfg.immich.page_size,
        timeout=cfg.immich.timeout_seconds,
    )

    checkpoint_before = store.get_checkpoint(NEW_ASSET_FEED)
    _log.info("batch run starting: checkpoint=%s", checkpoint_before)

    assets = client.fetch_new_assets(checkpoint_before)
    gate_config = GateConfig(
        confidence_floor=cfg.detection.confidence_floor,
        min_box_height=cfg.detection.min_box_height,
    )
    embedded, gate_counts, edge_clipped_kept, assets_without_persons = _process_assets(
        assets, client, cfg, store, gate_config
    )
    checkpoint_after = _advance_checkpoint(store, checkpoint_before, assets)
    _log.info(
        "batch run complete: fetched=%d appearances=%d checkpoint=%s -> %s",
        len(assets),
        embedded,
        checkpoint_before,
        checkpoint_after,
    )
    return RunSummary(
        assets=tuple(assets),
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
        appearances=embedded,
        gate_counts=gate_counts,
        edge_clipped_kept=edge_clipped_kept,
        assets_without_persons=assets_without_persons,
    )


def _process_assets(
    assets: list[Asset],
    client: ImmichClient,
    cfg: Config,
    store: Store,
    gate_config: GateConfig,
) -> tuple[int, GateCounts, int, int]:
    """Run detect, gate, and embed over every fetched asset.

    Returns the number of appearances embedded, the per-gate drop counts,
    the number of edge-clipped persons kept, and the number of assets with
    no kept appearance. Checkpoint advancement happens in run_batch.
    """
    detector: PersonDetector | None = None
    embedder: PersonEmbedder | None = None
    embedded = 0
    counts = GateCounts()
    edge_clipped_kept = 0
    assets_without_persons = 0
    for asset in assets:
        _log.info(
            "new asset id=%s file=%s created_at=%s",
            asset.id,
            asset.original_file_name,
            asset.created_at,
        )
        payload = client.fetch_pixels(asset.id, source=cfg.detection.pixel_source)
        if detector is None:
            detector = PersonDetector(str(YOLO11N_WEIGHTS))
            embedder = PersonEmbedder(str(OSNET_AIN_WEIGHTS))
        assert embedder is not None
        kept, asset_counts, clipped = process_asset(
            asset.id, payload, detector, embedder, gate_config
        )
        counts.low_confidence += asset_counts.low_confidence
        counts.below_min_height += asset_counts.below_min_height
        edge_clipped_kept += clipped
        embedded += len(kept)
        if not kept:
            assets_without_persons += 1
        store.add_appearances(
            asset.id,
            [
                Appearance(
                    asset_id=asset.id,
                    appearance=appearance.appearance,
                    embedding=appearance.embedding,
                    confidence=appearance.detection.confidence,
                    box_left=appearance.detection.left,
                    box_top=appearance.detection.top,
                    box_right=appearance.detection.right,
                    box_bottom=appearance.detection.bottom,
                    pixel_source=cfg.detection.pixel_source,
                )
                for appearance in kept
            ],
        )
    _log.info(
        "gate drops: low_confidence=%d below_min_height=%d",
        counts.low_confidence,
        counts.below_min_height,
    )
    return embedded, counts, edge_clipped_kept, assets_without_persons


def _advance_checkpoint(
    store: Store, checkpoint_before: str | None, assets: list[Asset]
) -> str | None:
    """Advance the checkpoint to the newest asset seen, or leave it if none.

    Ordering compares parsed timestamps (the checkpoint is a wall-clock
    instant), while the persisted value stays the raw ISO string Immich
    returned, sent back verbatim as the next run's createdAfter filter.
    """
    checkpoint_after = checkpoint_before
    if assets:
        newest_asset = max(assets, key=lambda asset: _parse_timestamp(asset.created_at))
        newest = newest_asset.created_at
        if checkpoint_before is None or _parse_timestamp(newest) > _parse_timestamp(
            checkpoint_before
        ):
            store.set_checkpoint(NEW_ASSET_FEED, newest)
            checkpoint_after = newest
    return checkpoint_after


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)
