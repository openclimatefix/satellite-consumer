"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import asyncio
import datetime as dt
import logging
import shutil
import tempfile
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
from satellite_consumer.download_gk2a import download_raw_gk2a, get_products_iterator_gk2a
from satellite_consumer.download_gk2a import (
    get_timestamp_from_filename as gk2a_timestamp_from_filename,
)
from satellite_consumer.download_goes import download_raw_goes, get_products_iterator_goes
from satellite_consumer.download_goes import (
    get_timestamp_from_filename as goes_timestamp_from_filename,
)
from satellite_consumer.download_himawari import (
    download_raw_himawari,
    get_products_iterator_himawari,
)
from satellite_consumer.download_himawari import (
    get_timestamp_from_filename as himawari_timestamp_from_filename,
)
from satellite_consumer.exceptions import DownloadError, ValidationError
from satellite_consumer.process import process_raw
from satellite_consumer.request_patch import construct_patched_request_function

if TYPE_CHECKING:
    import icechunk.repository

warnings.simplefilter(action="ignore", category=UnstableSpecificationWarning)
log = logging.getLogger("sat_consumer")

# Each satellite in `application.conf` declares the source its data is served from,
# which determines how its products are searched for and downloaded.
DOWNLOADERS: dict[str, Callable[..., list[str]]] = {
    "eumetsat": download_raw,
    "goes": download_raw_goes,
    "himawari": download_raw_himawari,
    "gk2a": download_raw_gk2a,
}

TIMESTAMP_PARSERS: dict[str, Callable[[str], dt.datetime]] = {
    "goes": goes_timestamp_from_filename,
    "himawari": himawari_timestamp_from_filename,
    "gk2a": gk2a_timestamp_from_filename,
}


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


async def _buffered_apply[T, R](
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
    source: str = "eumetsat",
) -> xr.Dataset | Exception:
    """Wrapper of the download and process functions."""
    raw_filepaths: list[str] = []
    himawari_tmpdir: str | None = None

    # Choose the downloader based on the data source the satellite is served from
    downloader = DOWNLOADERS.get(source)
    if downloader is None:
        raise ValueError(f"Unknown source {source}. Expected one of {list(DOWNLOADERS)}")

    # Calling `product.qualityStatus` makes an http request which can be slow. This filter is  done
    # inside this function so that it can be run on a worker and so avoid stalling the main process
    #if product.qualityStatus != "NOMINAL":
    #    return ValidationError(f"Product {product} qualityStatus is {product.qualityStatus}")

    try:
        t_start = time.time()
        raw_filepaths = downloader(
            product=product,
            folder=folder,
            filter_regex=filter_regex,
            nest_by_date=keep_raw,
        )

        # For Himawari, satpy decompresses .bz2 files to a temp dir.
        # Override satpy's tmp_dir to use a local folder instead of /tmp.
        if source == "himawari":
            import satpy as _satpy
            himawari_tmpdir = tempfile.mkdtemp(dir=".")
            _satpy.config.set(tmp_dir=himawari_tmpdir)

        t_dl = time.time()
        ds = process_raw(
            paths=raw_filepaths,
            channels=channels,
            resolution_meters=resolution_meters,
            crop_region_lonlat=crop_region_lonlat,
            source=source,
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
        # Cleanup Himawari decompression temp dir
        if himawari_tmpdir is not None:
            try:
                shutil.rmtree(himawari_tmpdir)
            except Exception:
                log.warning(f"failed to clean up temp dir {himawari_tmpdir}")
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
        raise ValueError(f"Non-appending dimensions do not match existing store Found: {[d for d in ds.dims if d not in skip_dims]} Required: {[d for d in store_ds.dims if d not in skip_dims]}")

def force_coords(ds: xr.Dataset, store_ds: xr.Dataset, skip_dims: list[str]) -> xr.Dataset:
    """Force the dimensions of the two datasets to be identical."""
    if not ds[[d for d in ds.dims if d not in skip_dims]].equals(
        store_ds[[d for d in store_ds.dims if d not in skip_dims]],
    ):
        for dim in ds.dims:
            if dim not in store_ds.dims and dim not in skip_dims:
                ds[dim] = store_ds[dim]
    return ds

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
    satellite: str = "odegree",
    source: str = "eumetsat",
    s3_listing_cache_dir: str | None = None,
    low_memory: bool = False,
) -> None:
    """Consume satellite data into a zarr store."""
    if low_memory:
        buffer_size = 1
        max_workers = 1
        accum_writes = 1
        log.info("Low memory mode: buffer_size=1, max_workers=1, accum_writes=1")
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
        existing_times: set[dt.datetime] = set()
    else:
        existing_times = set(
            pd.Timestamp(t).floor(f"{cadence_mins}min").to_pydatetime().replace(tzinfo=dt.UTC)
            for t in pd.to_datetime(store_ds.coords["time"].values, utc=True)
        )

    # Optionally set the start datetime to the last datetime in the store
    if jump_to_latest and existing_times and (max(existing_times) > dt_range[0]):
        start = max(existing_times)
        log.info(f"skipping to end of store: {start}")
    else:
        start = dt_range[0]

    # The source the satellite is served from determines how its products are searched for.
    # EUMETSAT products come from the Data Store as `Product` objects, whereas the other
    # sources are searched for on S3 and so come as groups of object paths.
    timestamp_from_filename = TIMESTAMP_PARSERS.get(source)
    product_iter: Iterator[eumdac.product.Product] | Iterator[list[str]]
    match source:
        case "eumetsat":
            product_iter = get_products_iterator(
                product_id=product_id,
                cadence_mins=cadence_mins,
                start=start,
                end=dt_range[1],
                credentials=eumetsat_credentials,
            )
        case "goes":
            product_iter = get_products_iterator_goes(
                product_id=product_id,
                start=start,
                end=dt_range[1],
                channels=channels,
                satellite=satellite,
                cache_dir=s3_listing_cache_dir,
            )
        case "himawari":
            product_iter = get_products_iterator_himawari(
                product_id=product_id,
                start=start,
                end=dt_range[1],
                channels=channels,
                cache_dir=s3_listing_cache_dir,
            )
        case "gk2a":
            product_iter = get_products_iterator_gk2a(
                product_id=product_id,
                start=start,
                end=dt_range[1],
                channels=channels,
                cache_dir=s3_listing_cache_dir,
            )
        case _:
            raise ValueError(f"Unknown source {source}. Expected one of {list(DOWNLOADERS)}")


    # This function will be applied to all products
    bound_func = partial(
        _download_and_process,
        folder=raw_zarr_paths[0],
        filter_regex=filter_regex,
        channels=channels,
        resolution_meters=resolution_meters,
        crop_region_lonlat=crop_region_lonlat,
        keep_raw=keep_raw,
        source=source,
    )

    # This function is run in all worker processes
    bound_initializer = partial(init_worker, request_timeout)

    def _not_stored(product: eumdac.product.Product | list[str]) -> bool:
        if isinstance(product, eumdac.product.Product):
            rounded_time: dt.datetime = (
                pd.Timestamp(product.sensing_end)
                .floor(f"{cadence_mins}min")
                .to_pydatetime()
                .replace(tzinfo=dt.UTC)  # EUMETSAT files are UTC without an explicit timezone
            )
        elif isinstance(product, list) and timestamp_from_filename is not None:
            rounded_time = (
                pd.Timestamp(timestamp_from_filename(product[0]))
                .floor(f"{cadence_mins}min")
                .to_pydatetime()
                .replace(tzinfo=dt.UTC)
            )
        else:
            raise TypeError(f"Unexpected product type {type(product)} for source {source}")
        return rounded_time not in existing_times

    # Iterate through all products in search
    num_skips: int = 0
    total_num: int = 0
    num_errs: int = 0
    results: list[xr.Dataset] = []
    t_last: float = 0
    get_iter_time_ema = EMA()
    async for item in _buffered_apply(
        (p for p in product_iter if _not_stored(p)),
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
                ds = (
                    xr.concat(results, dim="time", join="exact") if accum_writes > 1 else results[0]
                )

                # Check the non-append coords match the coords already in the store
                if store_ds is None:
                    write_new_store = True
                    store_ds = ds
                else:
                    write_new_store = False
                    try:
                        force_coords(ds, store_ds, skip_dims=["time"])
                    except ValueError as e:
                        log.error("Non-append dimensions do not match existing store: %s", e)
                        continue

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
