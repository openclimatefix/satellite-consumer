"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import math
from importlib.metadata import PackageNotFoundError, version

from loguru import logger as log

from satellite_consumer.config import (
    ArchiveCommandOptions,
    ConsumeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_nat,
    get_products_iterator,
)
from satellite_consumer.process import process_nat
from satellite_consumer.storage import create_empty_store, create_latest_zip, get_fs, write_to_zarr
from satellite_consumer.validate import validate

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

def _consume_command(command_opts: ArchiveCommandOptions | ConsumeCommandOptions) -> None:
    """Run the download and processing pipeline."""
    fs = get_fs(path=command_opts.zarr_path)

    window = command_opts.time_window

    product_iter, total = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    # Use existing zarr store if it exists
    if fs.exists(command_opts.zarr_path.replace("s3://", "")):
        log.info("Using existing zarr store", dst=command_opts.zarr_path)
    else:
        # Create new store
        log.info("Creating new zarr store", dst=command_opts.zarr_path)
        _ = create_empty_store(dst=command_opts.zarr_path, coords=command_opts.as_coordinates())

    # Iterate through all products in search
    nat_filepaths: list[str] = []
    for i, product in enumerate(product_iter): # Pretty sure enumerate is evaluated lazily
        product_time: dt.datetime = product.sensing_start.replace(second=0, microsecond=0)
        with log.contextualize(scan_time=str(product_time), scan_num=f"{i+1}/{total}"):

            nat_filepath = download_nat(
                product=product,
                folder=f"{command_opts.workdir}/raw",
            )

            da = process_nat(path=nat_filepath, hrv=command_opts.hrv)
            write_to_zarr(da=da, path=command_opts.zarr_path)
            nat_filepaths.append(nat_filepath)
            if i % math.ceil(total / 10) == 0:
                log.info(f"Processed {i+1} of {total} products.")

    log.info("Finished population of zarr store", dst=command_opts.zarr_path)

    if command_opts.validate:
        validate(dataset_path=command_opts.zarr_path)

    if isinstance(command_opts, ConsumeCommandOptions) and command_opts.latest_zip:
        zippath: str = create_latest_zip(zarr_path=command_opts.zarr_path)
        log.info(f"Created latest.zip at {zippath}", dst=zippath)

    if command_opts.delete_raw:
        if command_opts.workdir.startswith("s3://"):
            log.warning("delete-raw was specified, but deleting S3 files is not yet implemented")
        else:
            log.info(
                f"Deleting {len(nat_filepaths)} raw files in {command_opts.raw_folder}",
                num_files=len(nat_filepaths), dst=command_opts.raw_folder,
            )
            _ = [f.unlink() for f in nat_filepaths] # type:ignore


def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}'",
        version=__version__, start_time=str(prog_start),
    )

    if config.command == "archive" or config.command == "consume":
        _consume_command(command_opts=config.command_options)

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")

