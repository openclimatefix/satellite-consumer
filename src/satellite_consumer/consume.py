"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import logging
from typing import TYPE_CHECKING

import pandas as pd
import sentry_sdk

from satellite_consumer import models, storage
from satellite_consumer.download_eumetsat import (
    download_raw,
    get_products_iterator,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_raw

if TYPE_CHECKING:
    import icechunk.repository

log = logging.getLogger(__name__)


def consume_to_store(
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

    # Iterate through all products in search
    num_skips: int = 0
    num_duplicates: int = 0
    num_overwrites: int = 0

    for i, p in enumerate(product_iter):
        log.info("processing product %d: %s", i + 1, p.sensing_end)
        rounded_time: dt.datetime = (
            pd.Timestamp(p.sensing_end)
            .round(f"{cadence_mins} min")
            .to_pydatetime()
            .astimezone(tz=dt.UTC)
        )

        # Check if this timestamp already exists in the store
        if rounded_time in existing_times:
            num_duplicates += 1

            # We need to process the data first to get the satellite longitude
            # for the preference check. Download and process, then decide.
            try:
                raw_filepaths = download_raw(
                    product=p,
                    folder=raw_zarr_paths[0],
                    filter_regex=filter_regex,
                )
                ds = process_raw(
                    paths=raw_filepaths,
                    channels=channels,
                    resolution_meters=resolution_meters,
                    crop_region_lonlat=crop_region_lonlat,
                )

                # Get new satellite longitude for preference check
                new_longitude = storage._get_satellite_longitude(ds)

                # Check if we should overwrite based on satellite preference
                should_overwrite = storage.should_overwrite_existing(
                    dst=dst,
                    time_value=rounded_time,
                    new_satellite_longitude=new_longitude,
                )

                if not should_overwrite:
                    log.debug(
                        "skipping product %d at %s, already in store with preferred satellite",
                        i + 1,
                        rounded_time,
                    )
                    # Log to Sentry for monitoring
                    sentry_sdk.capture_message(
                        f"Duplicate timestamp detected: {rounded_time}",
                        level="info",
                    )
                    sentry_sdk.set_context("duplicate_info", {
                        "timestamp": str(rounded_time),
                        "new_satellite_longitude": new_longitude,
                        "action": "skipped",
                        "reason": "existing data from preferred satellite position",
                    })
                    num_skips += 1
                    continue

                # Should overwrite - log and proceed with write
                log.info(
                    "overwriting product at %s with data from preferred satellite position "
                    "(longitude: %s)",
                    rounded_time,
                    new_longitude,
                )
                sentry_sdk.capture_message(
                    f"Duplicate timestamp overwritten: {rounded_time}",
                    level="warning",
                )
                sentry_sdk.set_context("duplicate_info", {
                    "timestamp": str(rounded_time),
                    "new_satellite_longitude": new_longitude,
                    "action": "overwritten",
                    "reason": "new data from satellite closer to prime meridian",
                })
                num_overwrites += 1

                # Write the data (will overwrite existing)
                storage.write_to_store(
                    ds=ds,
                    dst=dst,
                    append_dim="time",
                    dims=dims_chunks_shards[0],
                    chunks=dims_chunks_shards[1],
                    shards=dims_chunks_shards[2],
                )

            except ValidationError as e:
                log.warning(
                    "skipping invalid product %d at %s: %s",
                    i + 1,
                    rounded_time,
                    str(e),
                )
                num_skips += 1
                continue
        else:
            # New timestamp - normal processing
            try:
                raw_filepaths = download_raw(
                    product=p,
                    folder=raw_zarr_paths[0],
                    filter_regex=filter_regex,
                )
                ds = process_raw(
                    paths=raw_filepaths,
                    channels=channels,
                    resolution_meters=resolution_meters,
                    crop_region_lonlat=crop_region_lonlat,
                )
                storage.write_to_store(
                    ds=ds,
                    dst=dst,
                    append_dim="time",
                    dims=dims_chunks_shards[0],
                    chunks=dims_chunks_shards[1],
                    shards=dims_chunks_shards[2],
                )
            except ValidationError as e:
                log.warning(
                    "skipping invalid product %d at %s: %s",
                    i + 1,
                    rounded_time,
                    str(e),
                )
                num_skips += 1
                continue

    # Log summary to Sentry if duplicates were encountered
    if num_duplicates > 0:
        sentry_sdk.capture_message(
            f"Run completed with {num_duplicates} duplicate timestamps "
            f"({num_overwrites} overwritten, {num_duplicates - num_overwrites} skipped)",
            level="info",
        )

    log.info(
        "path=%s, skips=%d, duplicates=%d (overwrites=%d), finished %d writes",
        raw_zarr_paths[1],
        num_skips,
        num_duplicates,
        num_overwrites,
        i + 1,
    )

