"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""
import datetime as dt
import os
import itertools

import warnings
from collections.abc import Generator, Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

import eumdac.product
import icechunk
import numpy as np
from icechunk.xarray import to_icechunk
from joblib import Parallel, delayed
from loguru import logger as log

from satellite_consumer import storage
from satellite_consumer.config import (
    ConsumeCommandOptions,
    MergeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_raw,
    get_products_iterator,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_raw
from satellite_consumer.validate import validate

T = TypeVar("T")

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"



import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
    traces_sample_rate=1,
)
sentry_sdk.set_tag("app_name", "satellite_consumer")
sentry_sdk.set_tag("app_version", __version__)



def _consume_to_store(command_opts: ConsumeCommandOptions) -> None:
    """Logic for the consume command (and the archive command)."""
    window = command_opts.time_window
    product_iter = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    processed_filepaths: list[str] = []
    num_skipped: int = 0
    num_written: int = 0

    if command_opts.icechunk:
        repo, existing_times = storage.get_icechunk_repo(path=command_opts.zarr_path)

        def _batcher(it: Iterable[T], batch_size: int = 10) -> Generator[tuple[T, ...]]:
            """Yield successive n-sized chunks from an iterable (excepting the last batch)."""
            while True:
                batch = tuple(itertools.islice(it, batch_size))
                if not batch:
                    return
                yield batch

        for product_batch in _batcher(product_iter, command_opts.num_workers):
            log.debug("Processing batch of products", num_products=len(product_batch))

            # Download the files for each product in the batch in parallel
            raw_filegroups = Parallel(n_jobs=command_opts.num_workers, prefer="threads")(
                delayed(download_raw)(
                    product=p,
                    folder=f"{command_opts.workdir}/raw",
                    filter_regex=command_opts.satellite_metadata.file_filter_regex,
                    existing_times=existing_times,
                )
                for p in product_batch
            )

            log.debug("Downloaded raw files for product batch", num_filegroups=len(raw_filegroups))

            # Process each products set of files in series
            for i, raw_filepaths in enumerate(raw_filegroups):
                if len(raw_filepaths) == 0:
                    num_skipped += 1
                    continue
                da = process_raw(
                    paths=raw_filepaths,
                    channels=command_opts.satellite_metadata.channels,
                    resolution_meters=command_opts.resolution,
                    normalize=command_opts.rescale,
                    crop_region_geos=command_opts.crop_region_geos,
                )
                # Don't write invalid data to the store
                validate(src=da)

                # Commit the data to the icechunk store
                # * If the store was just created, write as a fresh repo
                log.debug(
                    "Writing data to icechunk store",
                    dst=command_opts.zarr_path,
                    time=str(np.datetime_as_string(da.coords["time"].values[0], unit="m")),
                )
                if i == 0 and len(existing_times) == 0:
                    session: icechunk.Session = repo.writable_session(branch="main")
                    # TODO: Remove warnings catch when Zarr makes up its mind about codecs
                    with warnings.catch_warnings(action="ignore"):
                        to_icechunk(
                            obj=da.to_dataset(name="data", promote_attrs=True),
                            session=session,
                            mode="w-",
                            encoding={
                                "time": {"units": "nanoseconds since 1970-01-01"},
                                "data": {"dtype": "f4"},
                            },
                        )
                    _ = session.commit(message="initial commit")
                # Otherwise, append the data to the existing store
                else:
                    session = repo.writable_session(branch="main")
                    # TODO: Remove warnings catch when Zarr makes up its mind about codecs
                    with warnings.catch_warnings(action="ignore"):
                        to_icechunk(
                            obj=da.to_dataset(name="data", promote_attrs=True),
                            session=session,
                            append_dim="time",
                            mode="a",
                            rebase_with=icechunk.ConflictDetector(),
                            rebase_tries=5,
                        )
                    _ = session.commit(
                        message=f"add {len(da.coords['time']) * len(da.coords['variable'])} images",
                    )
                num_written += 1
                processed_filepaths.extend(raw_filepaths)

        log.info(
            "Finished population of icechunk store",
            dst=command_opts.zarr_path,
            num_skipped=num_skipped,
            num_written=num_written,
        )

    else:
        fs = storage.get_fs(path=command_opts.zarr_path)
        # Use existing zarr store if it exists
        if fs.exists(command_opts.zarr_path.replace("s3://", "")):
            log.info("Using existing store", dst=command_opts.zarr_path)
        else:
            # Create new store
            log.debug("Creating new zarr store", dst=command_opts.zarr_path)
            _ = storage.create_empty_zarr(
                dst=command_opts.zarr_path,
                coords=command_opts.as_coordinates(),
            )

        def _etl(product: eumdac.product.Product) -> list[str]:
            """Download, process, and save a single NAT file."""
            raw_filepaths = download_raw(
                product,
                folder=f"{command_opts.workdir}/raw",
                filter_regex=command_opts.satellite_metadata.file_filter_regex,
            )
            if len(raw_filepaths) == 0:
                return []
            da = process_raw(
                paths=raw_filepaths,
                channels=command_opts.satellite_metadata.channels,
                resolution_meters=command_opts.resolution,
                normalize=command_opts.rescale,
                crop_region_geos=command_opts.crop_region_geos,
            )
            storage.write_to_zarr(da=da, dst=command_opts.zarr_path)
            return raw_filepaths

        # Iterate through all products in search
        for raw_filepaths in Parallel(
            n_jobs=command_opts.num_workers,
            return_as="generator",
            prefer="threads",
        )(delayed(_etl)(product) for product in product_iter):
            if len(raw_filepaths) == 0:
                num_skipped += 1
            else:
                processed_filepaths.extend(raw_filepaths)

        # Might not need this as skipped files are all NaN
        # and the validation step should catch it
        if num_skipped / (num_skipped + len(raw_filepaths)) > 0.05:
            raise ValidationError(
                f"Too many files had to be skipped "
                f"({num_skipped}/{num_skipped + len(raw_filepaths)}). "
                "Use dataset at your own risk!",
            )

        log.info(
            "Finished population of zarr store",
            dst=command_opts.zarr_path,
            num_skipped=num_skipped,
        )

        if command_opts.validate:
            validate(src=command_opts.zarr_path)


def _merge_command(command_opts: MergeCommandOptions) -> None:
    """Logic for the merge command."""
    zarr_paths = command_opts.zarr_paths
    log.info(
        f"Merging {len(zarr_paths)} stores",
        num=len(zarr_paths),
        consume_missing=command_opts.consume_missing,
    )
    fs = storage.get_fs(path=zarr_paths[0])

    for zarr_path in zarr_paths:
        if not fs.exists(zarr_path):
            if command_opts.consume_missing:
                _consume_to_store(
                    command_opts=ConsumeCommandOptions(
                        time=dt.datetime.strptime(
                            zarr_path.split("/")[-1].split("_")[0],
                            "%Y%m%dT%H%M",
                        ).replace(tzinfo=dt.UTC),
                        satellite=command_opts.satellite,
                        workdir=command_opts.workdir,
                        validate=True,
                        rescale=True,  # TODO: Make this an option
                        resolution=command_opts.resolution,
                    ),
                )
            else:
                raise FileNotFoundError(f"Zarr store not found at {zarr_path}")

    dst = storage.create_latest_zip(srcs=zarr_paths)
    log.info("Created latest.zip", dst=dst)


def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}'",
        version=__version__,
        start_time=str(prog_start),
        opts=config.command_options.__str__(),
    )

    if isinstance(config.command_options, ConsumeCommandOptions):
        _consume_to_store(command_opts=config.command_options)
    elif isinstance(config.command_options, MergeCommandOptions):
        _merge_command(command_opts=config.command_options)
    else:
        pass

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")
