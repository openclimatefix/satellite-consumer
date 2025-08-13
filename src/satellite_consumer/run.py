"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import itertools
import os
import warnings
from collections.abc import Generator, Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

import eumdac.product
import icechunk
import numpy as np
import sentry_sdk
import zarr
from icechunk.xarray import to_icechunk
from joblib import Parallel, delayed
from loguru import logger as log

from satellite_consumer import storage
from satellite_consumer.config import (
    ConsumeCommandOptions,
    ExtractLatestCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_raw,
    get_products_iterator,
)
from satellite_consumer.download_gk2a import (
    download_raw_gk2a,
    get_products_iterator_gk2a,
)
from satellite_consumer.download_goes import (
    download_raw_goes,
    get_products_iterator_goes,
)
from satellite_consumer.download_himawari import (
    download_raw_himawari,
    get_products_iterator_himawari,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_raw
from satellite_consumer.validate import validate

T = TypeVar("T")

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"


def _remove_raw_filepaths(raw_filepaths: list[str]) -> None:
    for raw_filepath in raw_filepaths:
        try:
            os.remove(raw_filepath)
        except OSError as e:
            log.warning(
                "Failed to delete raw file",
                raw_filepath=raw_filepath,
                error=str(e),
            )


def _consume_to_store(command_opts: ConsumeCommandOptions) -> None:
    """Logic for the consume command (and the archive command)."""
    window = command_opts.time_window
    if "goes" in command_opts.satellite_metadata.region:
        get_iterator = get_products_iterator_goes
        load_raw = download_raw_goes
        satellite = "goes"
    elif "himawari" in command_opts.satellite_metadata.region:
        get_iterator = get_products_iterator_himawari
        load_raw = download_raw_himawari
        satellite = "himawari"
    elif "gk2a" in command_opts.satellite_metadata.region:
        get_iterator = get_products_iterator_gk2a
        load_raw = download_raw_gk2a
        satellite = "gk2a"
    else:
        get_iterator = get_products_iterator
        load_raw = download_raw
        satellite = "seviri"
    product_iter = get_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
        missing_product_threshold=0.99,
        resolution_meters=command_opts.resolution,
    )

    processed_filepaths: list[str] = []
    num_skipped: int = 0
    num_written: int = 0

    if command_opts.icechunk:
        repo, existing_times = storage.get_icechunk_repo(path=command_opts.zarr_path)

        def _batcher(it: Iterable[T], batch_size: int = 1) -> Generator[tuple[T, ...]]:
            """Yield successive n-sized chunks from an iterable (excepting the last batch)."""
            while True:
                batch = tuple(itertools.islice(it, batch_size))
                if not batch:
                    return
                yield batch

        for batch_num, product_batch in enumerate(_batcher(product_iter, command_opts.num_workers)):
            log.debug("Processing batch of products", num_products=len(product_batch))

            # Download the files for each product in the batch in parallel
            raw_filegroups = Parallel(n_jobs=command_opts.num_workers, prefer="threads")(
                delayed(load_raw)(
                    product=p,
                    folder=f"/scratch/{command_opts.workdir.split('/')[-1]}_{np.random.randint(1e6)}",
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
                try:
                    da = process_raw(
                        paths=raw_filepaths,
                        channels=command_opts.satellite_metadata.channels,
                        resolution_meters=command_opts.resolution,
                        normalize=command_opts.rescale,
                        crop_region_geos=command_opts.crop_region_geos,
                        satellite=satellite,
                    )
                except Exception as e:
                    log.error(
                        "Error processing raw files",
                        raw_filepaths=raw_filepaths,
                        error=str(e),
                    )
                    # Still remove the raw filepaths to save space
                    for raw_filepath in raw_filepaths:
                        try:
                            os.remove(raw_filepath)
                        except OSError as e:
                            log.warning(
                                "Failed to delete raw file",
                                raw_filepath=raw_filepath,
                                error=str(e),
                            )
                    num_skipped += 1
                    continue
                # Don't write invalid data to the store
                # validate(src=da)

                # Commit the data to the icechunk store
                # * If the store was just created, write as a fresh repo
                log.debug(
                    "Writing data to icechunk store",
                    dst=command_opts.zarr_path,
                    time=str(np.datetime_as_string(da.coords["time"].values[0], unit="m")),
                )
                if len(existing_times) == 0 and batch_num == 0 and i == 0:
                    session: icechunk.Session = repo.writable_session(branch="main")
                    # TODO: Remove warnings catch when Zarr makes up its mind about codecs
                    with warnings.catch_warnings(action="ignore"):
                        encoding = {
                            "time": {
                                "units": "nanoseconds since 1970-01-01",
                                "calendar": "proleptic_gregorian",
                            },
                        }
                        encoding.update(
                            {
                                v: {
                                    "dtype": "float16",
                                    "compressors": zarr.codecs.BloscCodec(
                                        cname="zstd",
                                        clevel=9,
                                        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
                                    ),
                                }
                                for v in da.data_vars
                                if v
                                not in [
                                    "start_time",
                                    "end_time",
                                    "x_geostationary_coordinates",
                                    "y_geostationary_coordinates",
                                    "orbital_parameters",
                                    "area",
                                    "platform_name",
                                ]
                            },
                        )
                        encoding.update(
                            {
                                "start_time": {
                                    "dtype": "datetime64[ns]",
                                    "units": "nanoseconds since 1970-01-01",
                                },
                                "end_time": {
                                    "dtype": "datetime64[ns]",
                                    "units": "nanoseconds since 1970-01-01",
                                },
                                "platform_name": {
                                    "dtype": "U12",
                                },
                                "area": {
                                    "dtype": "U512",
                                },
                            },
                        )
                        to_icechunk(obj=da.chunk({"time": 1, "x_geostationary": -1, "y_geostationary": -1}), session=session, mode="w-", encoding=encoding)
                    _ = session.commit(message="initial commit")
                # Otherwise, append the data to the existing store
                else:
                    # Check one last time that the time is not already in the store
                    if da.coords["time"].values[0] in existing_times:
                        log.warning(
                            "Skipping data with existing time",
                            time=str(np.datetime_as_string(da.coords["time"].values[0], unit="m")),
                        )
                        num_skipped += 1
                        _remove_raw_filepaths(raw_filepaths)
                        continue
                    session = repo.writable_session(branch="main")
                    # TODO: Remove warnings catch when Zarr makes up its mind about codecs
                    with warnings.catch_warnings(action="ignore"):
                        to_icechunk(
                            obj=da.chunk({"time": 1, "x_geostationary": -1, "y_geostationary": -1}),
                            session=session,
                            append_dim="time",
                            mode="a",
                        )
                    _ = session.commit(
                        message=f"add {len(da.coords['time'])} images",
                        rebase_with=icechunk.ConflictDetector(),
                        rebase_tries=10,
                    )
                num_written += 1
                processed_filepaths.extend(raw_filepaths)
                # Delete raw filepaths to save space
                _remove_raw_filepaths(raw_filepaths)

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
            raw_filepaths = download_raw_goes(
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
        _consume_to_store(command_opts=config.command_options)
    elif isinstance(config.command_options, ExtractLatestCommandOptions):
        _extract_latest_command(command_opts=config.command_options)
    else:
        pass

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")
