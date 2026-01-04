"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import asyncio
import datetime as dt
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

import icechunk
import numpy as np
import sentry_sdk
from icechunk.xarray import to_icechunk
from loguru import logger as log

from satellite_consumer import storage
from satellite_consumer.config import (
    ConsumeCommandOptions,
    ExtractLatestCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    buffered_download_stream,
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


async def _consume_to_store(command_opts: ConsumeCommandOptions) -> None:
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

        async for raw_filepaths in buffered_download_stream(
            products=product_iter,
            folder=f"{command_opts.workdir}/raw",
            filter_regex=command_opts.satellite_metadata.file_filter_regex,
            max_concurrent=command_opts.num_workers,
        ):
            log.info(". . . . [Processing]", num_files=len(raw_filepaths))
            if len(raw_filepaths) == 0:
                num_skipped += 1
                continue

            # Process the raw files in a potentially non-blocking way
            # We use to_thread because process_raw is CPU bound and blocking
            da = await asyncio.to_thread(
                process_raw,
                paths=raw_filepaths,
                channels=command_opts.satellite_metadata.channels,
                resolution_meters=command_opts.resolution,
                normalize=command_opts.rescale,
                crop_region_geos=command_opts.crop_region_geos,
            )

            # Don't write invalid data to the store
            if command_opts.validate:
                validate(src=da)

            # Commit the data to the icechunk store
            # * If the store was just created, write as a fresh repo
            log.debug(
                "Writing data to icechunk store",
                dst=command_opts.zarr_path,
                time=str(np.datetime_as_string(da.coords["time"].values[0], unit="m")),
            )
            # Since Icechunk writes are not thread safe or async, we do this in the main thread
            # This effectively makes this the "sequential bottleneck" which is desired for ordering
            if len(existing_times) == 0 and num_written == 0:
                session: icechunk.Session = repo.writable_session(branch="main")
                # TODO: Remove warnings catch when Zarr makes up its mind about codecs
                with warnings.catch_warnings(action="ignore"):
                    to_icechunk(
                        obj=da.to_dataset(name="data", promote_attrs=True),
                        session=session,
                        mode="w-",
                        encoding={
                            "time": {
                                "units": "nanoseconds since 1970-01-01",
                                "calendar": "proleptic_gregorian",
                            },
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
                    )
                _ = session.commit(
                    message=f"add {len(da.coords['time']) * len(da.coords['variable'])} images",
                    rebase_with=icechunk.ConflictDetector(),
                    rebase_tries=5,
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

        # Iterate through all products in search
        async for raw_filepaths in buffered_download_stream(
            products=product_iter,
            folder=f"{command_opts.workdir}/raw",
            filter_regex=command_opts.satellite_metadata.file_filter_regex,
            max_concurrent=command_opts.num_workers,
        ):
            log.info(". . . . [Processing]", num_files=len(raw_filepaths))
            if len(raw_filepaths) == 0:
                num_skipped += 1
                continue

            da = await asyncio.to_thread(
                process_raw,
                paths=raw_filepaths,
                channels=command_opts.satellite_metadata.channels,
                resolution_meters=command_opts.resolution,
                normalize=command_opts.rescale,
                crop_region_geos=command_opts.crop_region_geos,
            )
            storage.write_to_zarr(da=da, dst=command_opts.zarr_path)
            processed_filepaths.extend(raw_filepaths)

        # Might not need this as skipped files are all NaN
        # and the validation step should catch it
        if (
            num_skipped > 0
            and len(processed_filepaths) > 0
            and num_skipped / (num_skipped + len(processed_filepaths)) > 0.05
        ):
            raise ValidationError(
                f"Too many files had to be skipped "
                f"({num_skipped}/{num_skipped + len(processed_filepaths)}). "
                "Use dataset at your own risk!",
            )

        log.info(
            "Finished population of zarr store",
            dst=command_opts.zarr_path,
            num_skipped=num_skipped,
        )

        if command_opts.validate:
            validate(src=command_opts.zarr_path)


def _extract_latest_command(command_opts: ExtractLatestCommandOptions) -> None:
    """Logic for the merge command."""
    log.info(f"Extracting latest data from {command_opts.zarr_path}")
    desired_image_num: int = (
        command_opts.window_mins // command_opts.satellite_metadata.cadence_mins
    )

    dst = storage.create_latest_zip(
        src=command_opts.zarr_path,
        time_slice=slice(-desired_image_num, None),
    )
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

    if os.getenv("SENTRY_DSN", "") != "":
        sentry_sdk.init(
            dsn=os.environ["SENTRY_DSN"],
            environment=os.getenv("ENVIRONMENT", "local"),
            traces_sample_rate=1,
        )
        sentry_sdk.set_tag("app_name", "satellite_consumer")
        sentry_sdk.set_tag("app_version", __version__)

    if isinstance(config.command_options, ConsumeCommandOptions):
        asyncio.run(_consume_to_store(command_opts=config.command_options))
    elif isinstance(config.command_options, ExtractLatestCommandOptions):
        _extract_latest_command(command_opts=config.command_options)
    else:
        pass

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")
