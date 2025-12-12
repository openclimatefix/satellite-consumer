"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import logging

from satellite_consumer import models, storage
from satellite_consumer.download_eumetsat import (
    download_raw,
    get_products_iterator,
)
from satellite_consumer.process import process_raw

log = logging.getLogger(__name__)


def consume_to_store(
        dt_range: tuple[dt.datetime, dt.datetime],
        cadence_mins: int,
        product_id: str,
        filter_regex: str,
        raw_zarr_paths: tuple[str, str],
        channels: list[models.SpectralChannel],
        resolution_meters: int,
        crop_region_geos: tuple[float, float, float, float] | None,
        eumetsat_credentials: tuple[str, str],
        icechunk: bool = False,
        aws_credentials: tuple[
            str | None, str | None, str | None, str | None,
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

    if icechunk:
        repo = storage.get_icechunk_repo(
            raw_zarr_paths[1],
            aws_access_key_id=aws_credentials[0],
            aws_secret_access_key=aws_credentials[1],
            aws_region_name=aws_credentials[2],
            aws_endpoint_url=aws_credentials[3],
            gcs_token=gcs_credentials,
        )

    num_skipped: int = 0
    # Iterate through all products in search
    for p in product_iter:
        raw_filepaths = download_raw(
            product=p,
            folder=raw_zarr_paths[0],
            filter_regex=filter_regex,
        )
        if len(raw_filepaths) == 0:
            num_skipped += 1
            continue
        ds = process_raw(
            paths=raw_filepaths,
            channels=channels,
            resolution_meters=resolution_meters,
            crop_region_geos=crop_region_geos,
        )
        if icechunk:
            storage.write_to_icechunk(ds=ds, repo=repo)
        else:
            storage.write_to_zarr(ds=ds, dst=raw_zarr_paths[1])

    log.debug("skips=%d, path=%s, finished writes", num_skipped, raw_zarr_paths[1])

