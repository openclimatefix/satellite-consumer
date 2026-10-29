"""Optional disk cache for S3 listing results."""

import json
import logging
import os

log = logging.getLogger("sat_consumer")


def _cache_path(cache_dir: str, bucket: str, product_id: str) -> str:
    """Get the path to the cache file for a given bucket and product ID."""
    safe_product = product_id.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{bucket}_{safe_product}.json")


def load_cache(cache_dir: str, bucket: str, product_id: str) -> dict[str, list[str]]:
    """Load the S3 listing cache from disk, or return empty dict if missing."""
    path = _cache_path(cache_dir, bucket, product_id)
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
        log.debug("Loaded S3 listing cache with %d entries from %s", len(cache), path)
        return cache
    return {}


def save_cache(
    cache_dir: str,
    bucket: str,
    product_id: str,
    cache: dict[str, list[str]],
) -> None:
    """Save the S3 listing cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, bucket, product_id)
    with open(path, "w") as f:
        json.dump(cache, f)
    log.debug("Saved S3 listing cache with %d entries to %s", len(cache), path)
