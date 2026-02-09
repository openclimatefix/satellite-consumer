"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import asyncio
import datetime as dt
import logging
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import eumdac.product
import numpy as np
import pandas as pd
import xarray as xr
from zarr.errors import UnstableSpecificationWarning

from satellite_consumer import models, storage
from satellite_consumer.download_eumetsat import download_raw, get_products_iterator
from satellite_consumer.exceptions import DownloadError, ValidationError
from satellite_consumer.process import process_raw
from satellite_consumer.request_patch import construct_patched_request_function

if TYPE_CHECKING:
    import icechunk.repository

warnings.simplefilter(action="ignore", category=UnstableSpecificationWarning)
log = logging.getLogger("sat_consumer")


def init_worker(timeout: int) -> None:
    """Patch the `eumdac.request._request()` function in all workers."""
    import eumdac.request

    eumdac.request._request = construct_patched_request_function(
        max_retries=3,
        backoff_factor=0.3,
        timeout=timeout,
    )


T = TypeVar("T")  # Type of the input
R = TypeVar("R")  # Type of the return


async def _buffered_apply(
    item_iter: Iterator[T],
    func: Callable[[T], R],
    buffer_size: int,
    max_workers: int,
    executor: Literal["threads", "processes"],
    initializer: Callable[[], None] | None = None,
) -> AsyncIterator[R]:
    """Asynchronously applies a synchronous function to items using a sliding window buffer.

    The function `func` is applied simultaneously in threads up to a buffer size of `buffer_size`
    items ahead. It yields results as soon as they are available, strictly preserving the order of
    the input iterator.

    Args:
        item_iter: An iterator producing the input items.
        func: The function to apply to each item.
        buffer_size: The length of the buffer.
        max_workers: The number of workers in the pool.
        executor: "threads" or "processes".
        initializer: Function that is called at the start of each worker process.

    Yields:
        The result of `func` applied to each item from `item_iter`, in the original order.
    """
    loop = asyncio.get_running_loop()

    ExecutorClass = ProcessPoolExecutor if executor == "processes" else ThreadPoolExecutor

    with ExecutorClass(max_workers=max_workers, initializer=initializer) as pool:
        tasks: deque[asyncio.Future[R]] = deque()

        # Fill the buffer initially
        for item in islice(item_iter, buffer_size):
            tasks.append(loop.run_in_executor(pool, func, item))

        # Loop through the remaining items: yield one, add one
        for item in item_iter:
            # Get next item and kick off the task before yielding
            result = await tasks.popleft()
            tasks.append(loop.run_in_executor(pool, func, item))
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
    keep_raw: bool,
    retries: int,
) -> xr.Dataset | Exception:
    """Wrapper of the download and process functions."""
    raw_filepaths: list[str] = []

    # Calling `product.qualityStatus` makes an http request which can be slow. This filter is  done
    # inside this function so that it can be run on a worker and so avoid stalling the main process
    if product.qualityStatus != "NOMINAL":
        return ValidationError(f"Product {product} qualityStatus is {product.qualityStatus}")

    try:
        t_start = time.time()
        raw_filepaths = download_raw(
            product=product,
            folder=folder,
            filter_regex=filter_regex,
            nest_by_date=keep_raw,
            retries=retries,
        )

        t_dl = time.time()
        ds = process_raw(
            paths=raw_filepaths,
            channels=channels,
            resolution_meters=resolution_meters,
            crop_region_lonlat=crop_region_lonlat,
        )

        t_end = time.time()
        log.debug(
            f"Downloaded ({t_dl - t_start:.2f}s) and processed ({t_end - t_dl:.2f}s)"
            f" for timestamp {np.datetime_as_string(ds.time.values[0], unit='s')}"
        )

        return ds

    except Exception as e:
        return e

    finally:
        # Cleanup files
        if not keep_raw and raw_filepaths:
            try:
                fs = storage.get_fs(folder)
                for path in raw_filepaths:
                    if fs.exists(path):
                        fs.delete(path)
            except Exception:
                # log this rather than returning it, since an error is already returned from the
                # except block
                log.warning(f"failed to clean up {raw_filepaths}")


class EMA:
    """Track Exponential moving average (EMA) of values."""

    def __init__(self, alpha: float = 0.3) -> None:
        """Track Exponential moving average (EMA) of values.

        Args:
            alpha: Smoothing factor in range [0, 1]. Increase to give more weight to recent values.
                Ranges from 0 (yields old value) to 1 (yields new value).
        """
        self.alpha: float = alpha
        self.last: float = 0
        self.calls: int = 0

    def __call__(self, x: float) -> float:
        """Add value to EMA and return new average.

        Args:
            x: New value to include in EMA.
        """
        beta = 1 - self.alpha
        self.last = self.alpha * x + beta * self.last
        self.calls += 1
        return self.last / (1 - beta**self.calls) if self.calls else self.last


def check_coords(ds: xr.Dataset, store_ds: xr.Dataset, skip_dims: list[str]) -> None:
    """Check the dimensions of the two datasets are identical."""
    if not ds[[d for d in ds.dims if d not in skip_dims]].equals(
        store_ds[[d for d in store_ds.dims if d not in skip_dims]],
    ):
        raise ValueError("Non-appending dimensions do not match existing store")


async def consume_to_store(
    dt_range: tuple[dt.datetime, dt.datetime],
    jump_to_latest: bool,
    cadence_mins: int,
    product_id: str,
    filter_regex: str,
    raw_zarr_paths: tuple[str, str],
    keep_raw: bool,
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_lonlat: tuple[float, float, float, float] | None,
    encoding: dict[str, Any],
    eumetsat_credentials: tuple[str, str],
    buffer_size: int,
    max_workers: int,
    accum_writes: int,
    executor: Literal["threads", "processes"],
    request_timeout: int,
    use_icechunk: bool = False,
    aws_credentials: tuple[
        str | None,
        str | None,
        str | None,
        str | None,
    ] = (None, None, None, None),

    gcs_credentials: str | None = None,
    retries: int = 6,
) -> None:
    """Consume satellite data into a zarr store."""
    # If the store already exists, open it and find its timestamps
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

    store_ds = storage.get_existing_dataset(dst)
    if store_ds is None:
        existing_times = []
    else:
        existing_times = (
            pd.to_datetime(store_ds.coords["time"].values, utc=True).to_pydatetime().tolist()
        )

    # Optionally set the start datetime to the last datetime in the store
    if jump_to_latest and store_ds is not None:
        start = existing_times[-1]
        log.info(f"skipping to end of store: {start}")
    else:
        start = dt_range[0]

    product_iter = get_products_iterator(
        product_id=product_id,
        cadence_mins=cadence_mins,
        start=start,
        end=dt_range[1],
        credentials=eumetsat_credentials,
    )

    # This function will be applied to all products
    bound_func = partial(
        _download_and_process,
        folder=raw_zarr_paths[0],
        filter_regex=filter_regex,
        channels=channels,
        resolution_meters=resolution_meters,
        crop_region_lonlat=crop_region_lonlat,
        keep_raw=keep_raw,
        retries=retries,
    )

    # This function is run in all worker processes
    bound_initializer = partial(init_worker, request_timeout)

    def _not_stored(product: eumdac.product.Product) -> bool:
        rounded_time: dt.datetime = (
            pd.Timestamp(product.sensing_end)
            .round(f"{cadence_mins} min")
            .to_pydatetime()
            .astimezone(tz=dt.UTC)
        )
        return rounded_time not in existing_times

    # Iterate through all products in search
    num_skips: int = 0
    total_num: int = 0
    num_errs: int = 0
    results: list[xr.Dataset] = []
    t_last: float = 0
    get_iter_time_ema = EMA()
    async for item in _buffered_apply(
        filter(_not_stored, product_iter),
        bound_func,
        buffer_size=buffer_size,
        max_workers=max_workers,
        executor=executor,
        initializer=bound_initializer,
    ):
        total_num += 1

        if isinstance(item, xr.Dataset):
            results.append(item)

            # If we've reached the write block size, concat the datasets and write out
            if len(results) == accum_writes:
                ds = xr.concat(results, dim="time") if accum_writes > 1 else results[0]

                # Check the non-append coords match the coords already in the store
                if store_ds is None:
                    write_new_store = True
                    store_ds = ds
                else:
                    write_new_store = False
                    check_coords(ds, store_ds, skip_dims=["time"])

                storage.write_to_store(
                    ds=ds,
                    dst=dst,
                    append_dim="time",
                    encoding=encoding,
                    write_new=write_new_store,
                )
                results = []

                # Log progress and timings
                latest_timestamp = pd.to_datetime(ds.time.values[-1]).tz_localize("UTC")
                progress_frac = (latest_timestamp - dt_range[0]) / (dt_range[1] - dt_range[0])
                num_images_remaining = (dt_range[1] - latest_timestamp).total_seconds() / (
                    60 * cadence_mins
                )

                t_now = time.time()
                # Skip the time taken by the first yielded item due to setup time
                if t_last == 0:
                    time_per_image = np.nan
                    eta = dt.timedelta(seconds=0)
                else:
                    time_per_image = get_iter_time_ema((t_now - t_last) / accum_writes)
                    eta = dt.timedelta(seconds=num_images_remaining * time_per_image)
                t_last = t_now

                log.info(
                    "%.2f%% progress through time range. %.2f seconds/image. ETA %s",
                    progress_frac * 100,
                    time_per_image,
                    eta,
                )

        elif isinstance(item, ValidationError):
            log.warning("skipping invalid product %s", str(item))
            num_skips += 1

        elif isinstance(item, DownloadError):
            log.error("error downloading product %s", str(item))
            num_errs += 1

        elif isinstance(item, Exception):
            raise item

        else:
            raise TypeError(f"Unexpected return type {type(item)}")

    # Write out any remaining values
    if len(results) > 0:
        ds = xr.concat(results, dim="time") if accum_writes > 1 else results[0]

        storage.write_to_store(
            ds=ds,
            dst=dst,
            append_dim="time",
            encoding=encoding,
            write_new=store_ds is None,
        )

    log.info(
        "path=%s, skips=%d, errs=%d, finished %d writes",
        raw_zarr_paths[1],
        num_skips,
        num_errs,
        total_num,
    )
