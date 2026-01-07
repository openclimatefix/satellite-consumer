"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import asyncio
import datetime as dt
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

import eumdac.product
import pandas as pd
import xarray as xr

from satellite_consumer import models, storage
from satellite_consumer.download_eumetsat import download_raw, get_products_iterator
from satellite_consumer.exceptions import DownloadError, ValidationError
from satellite_consumer.process import process_raw

if TYPE_CHECKING:
    import icechunk.repository

log = logging.getLogger(__name__)


T = TypeVar("T") # Type of the input item
R = TypeVar("R") # Type of the result
async def _buffered_apply(
    item_iter: Iterator[T],
    func: Callable[[T], R],
    buffer_size: int,
) -> AsyncIterator[R]:
    """Asynchronously applies a synchronous function to items using a sliding window buffer.

    The function `func` is applied simultaneously in threads up to a buffer size of `buffer_size`
    items ahead. It yields results as soon as they are available, strictly preserving the order of
    the input iterator.

    Args:
        item_iter: An iterator producing the input items.
        func: The function to apply to each item.
        buffer_size: The length of the buffer.

    Yields:
        The result of `func` applied to each item from `item_iter`, in the original order.
    """

    def create_task(item: T) -> asyncio.Task[R]:
        return asyncio.create_task(asyncio.to_thread(func, item))

    tasks: deque[asyncio.Task[R]] = deque()

    # Fill the buffer initially
    for item in islice(item_iter, buffer_size):
        tasks.append(create_task(item))

    # Loop through the remaining items: yield one, add one
    for item in item_iter:
        # Get next item and kick off the task before yielding
        result = await tasks.popleft()
        tasks.append(create_task(item))
        yield result

    # Drain the remaining tasks in the buffer
    while tasks:
        yield await tasks.popleft()


def _download_and_process(
    product: eumdac.product.Product,
    folder: str,
    filter_regex: str,
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_lonlat: tuple[float, float, float, float] | None,
) -> xr.Dataset | None:
    """Wrapper of the download and process functions."""
    try:
        log.info("downloading %s", product._id)
        raw_filepaths = download_raw(
            product=product,
            folder=folder,
            filter_regex=filter_regex,
        )

        log.info("processing %s", product._id)
        ds = process_raw(
            paths=raw_filepaths,
            channels=channels,
            resolution_meters=resolution_meters,
            crop_region_lonlat=crop_region_lonlat,
        )

        return ds

    except ValidationError as e:
        log.warning("skipping invalid product %s", str(e))
        return None

    except DownloadError as e:
        log.error("error downloading product %s", str(e))
        return None


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


    # Filter out the times which already exist in the store
    existing_times = storage.get_existing_times(dst=dst, time_dim="time")

    def _not_downloaded(product: eumdac.product.Product) -> bool:
        rounded_time: dt.datetime = (
            pd.Timestamp(product.sensing_end)
            .round(f"{cadence_mins} min")
            .to_pydatetime()
            .astimezone(tz=dt.UTC)
        )
        return rounded_time not in existing_times

    new_products_iter = filter(_not_downloaded, product_iter)

    # This function will be applied to all products
    bound_func = partial(
        _download_and_process,
        folder=raw_zarr_paths[0],
        filter_regex=filter_regex,
        channels=channels,
        resolution_meters=resolution_meters,
        crop_region_lonlat=crop_region_lonlat,
    )

    # Iterate through all products in search
    num_skips: int = 0
    total_num: int = 0
    async for ds in _buffered_apply(new_products_iter, bound_func, buffer_size=10):

        total_num += 1

        if ds is None:
            num_skips += 1

        else:
            log.info(f"saving image for timestamp {pd.Timestamp(ds.time.item())}")
            storage.write_to_store(
                ds=ds,
                dst=dst,
                append_dim="time",
                dims=dims_chunks_shards[0],
                chunks=dims_chunks_shards[1],
                shards=dims_chunks_shards[2],
            )

    log.info(
        "path=%s, skips=%d, finished %d writes",
        raw_zarr_paths[1],
        num_skips,
        total_num,
    )
