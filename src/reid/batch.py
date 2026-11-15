"""The batch run: fetch new assets through the checkpointed feed.

No detection or embedding yet — this slice is the checkpointed asset feed.
The checkpoint stores the newest asset createdAt seen in a fully completed
run; the next run fetches strictly newer assets. The checkpoint only advances
after the whole feed has been read, so a crashed run resumes from the last
checkpoint instead of reprocessing the library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .config import Config
from .immich import Asset, ImmichClient
from .store import NEW_ASSET_FEED, Store

_log = logging.getLogger("reid.batch")


@dataclass(frozen=True)
class RunSummary:
    assets: tuple[Asset, ...]
    checkpoint_before: str | None
    checkpoint_after: str | None


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
    for asset in assets:
        _log.info(
            "new asset id=%s file=%s created_at=%s",
            asset.id,
            asset.original_file_name,
            asset.created_at,
        )

    checkpoint_after = _advance_checkpoint(store, checkpoint_before, assets)
    _log.info(
        "batch run complete: fetched=%d checkpoint=%s -> %s",
        len(assets),
        checkpoint_before,
        checkpoint_after,
    )
    return RunSummary(
        assets=tuple(assets),
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
    )


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
