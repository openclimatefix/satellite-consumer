"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import asyncio
import datetime as dt
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import eumdac
    import icechunk.repository

import pandas as pd

from satellite_consumer import models, storage
from satellite_consumer.download_eumetsat import (
    buffered_download_stream,
    get_products_iterator,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_raw

log = logging.getLogger(__name__)


async def consume_to_store(
    dt_range: tuple[dt.datetime, dt.datetime],
    cadence_mins: int,
    product_id: str,
    filter_regex: str,
    raw_zarr_paths: tuple[str, str],
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_lonlat: tuple[float, float, float, float] | None,
    eumetsat_credentials: tuple[str, str],
    dims_chunks_shards: tuple[list[str], list[int], list[int]],
    use_icechunk: bool = False,
    aws_credentials: tuple[
        str | None,
        str | None,
        str | None,
        str | None,
    ] = (None, None, None, None),
    gcs_credentials: str | None = None,
    concurrent_downloads: int = 5,
) -> None:
    """Consume satellite data into a zarr store."""
    product_iter = get_products_iterator(
        product_id=product_id,
        cadence_mins=cadence_mins,
        start=dt_range[0],
        end=dt_range[1],
        credentials=eumetsat_credentials,
    )

    dst: str | icechunk.repository.Repository = raw_zarr_paths[1]
    if use_icechunk:
        dst = storage.get_icechunk_repo(
            raw_zarr_paths[1],
            aws_access_key_id=aws_credentials[0],
            aws_secret_access_key=aws_credentials[1],
            aws_region_name=aws_credentials[2],
            aws_endpoint_url=aws_credentials[3],
            gcs_token=gcs_credentials,
        )

    existing_times = storage.get_existing_times(dst=dst, time_dim="time")

    # Defined a generator that yields only the products we need (skipping existing)
    def filtered_product_iter() -> "Iterator[eumdac.product.Product]":
        for i, p in enumerate(product_iter):
             log.info("processing product %d: %s", i + 1, p.sensing_end)
             rounded_time: dt.datetime = (
                pd.Timestamp(p.sensing_end)
                .round(f"{cadence_mins} min")
                .to_pydatetime()
                .astimezone(tz=dt.UTC)
            )
             if rounded_time in existing_times:
                log.debug(
                    "skipping product %d at %s, already in store",
                     i + 1,
                    rounded_time,
                )
                continue
             yield p

    num_skips: int = 0
    num_writes: int = 0

    # Use the buffered stream to download concurrently
    async for raw_filepaths in buffered_download_stream(
        products=filtered_product_iter(),
        folder=raw_zarr_paths[0],
        filter_regex=filter_regex,
        max_concurrent=concurrent_downloads,
    ):
         if not raw_filepaths:
             num_skips += 1
             continue

         log.info(". . . . [Processing] num_files=%d", len(raw_filepaths))

         try:
            # Process in thread to avoid blocking download loop
            ds = await asyncio.to_thread(
                process_raw,
                paths=raw_filepaths,
                channels=channels,
                resolution_meters=resolution_meters,
                crop_region_lonlat=crop_region_lonlat,
            )

            # Write to store (this part remains synchronous/blocking to ensure safety/ordering)
            storage.write_to_store(
                ds=ds,
                dst=dst,
                append_dim="time",
                dims=dims_chunks_shards[0],
                chunks=dims_chunks_shards[1],
                shards=dims_chunks_shards[2],
            )
            num_writes += 1
         except ValidationError as e:
            log.warning(
                "skipping invalid product: %s",
                str(e),
            )
            num_skips += 1
            continue

    log.info("path=%s, skips=%d, finished %d writes", raw_zarr_paths[1], num_skips, num_writes)
