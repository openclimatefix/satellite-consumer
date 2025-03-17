"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import os
from importlib.metadata import PackageNotFoundError, version

import sentry_sdk
import eumdac.product
from joblib import Parallel, delayed
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
from satellite_consumer.storage import create_empty_zarr, create_latest_zip, get_fs, write_to_zarr
from satellite_consumer.validate import validate

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

# Sentry initialization as per the suggestion
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"), 
    environment=os.getenv("ENVIRONMENT", "local"), 
    traces_sample_rate=1,
)
sentry_sdk.set_tag("app_name", "satellite_consumer")
sentry_sdk.set_tag("version", __version__)

def _consume_command(command_opts: ArchiveCommandOptions | ConsumeCommandOptions) -> None:
    """Run the download and processing pipeline."""
    fs = get_fs(path=command_opts.zarr_path)
    window = command_opts.time_window

    product_iter = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    # Ensure proper S3 path checking
    if command_opts.zarr_path.startswith("s3://"):
        log.info(f"Checking existence of S3 path: {command_opts.zarr_path}")
    elif fs.exists(command_opts.zarr_path):
        log.info(f"Using existing zarr store: {command_opts.zarr_path}")
    else:
        log.info(f"Creating new zarr store: {command_opts.zarr_path}")
        _ = create_empty_zarr(dst=command_opts.zarr_path, coords=command_opts.as_coordinates())

    def _etl(product: eumdac.product.Product) -> str | None:
        """Download, process, and save a single NAT file."""
        try:
            nat_filepath = download_nat(product, folder=f"{command_opts.workdir}/raw")
            da = process_nat(path=nat_filepath, hrv=command_opts.hrv)
            write_to_zarr(da=da, dst=command_opts.zarr_path)
            return nat_filepath
        except Exception as e:
            log.error(f"Failed to process product {product}: {e}")
            return None  # Ensure function always returns

    nat_filepaths: list[str] = []
    for nat_filepath in Parallel(
        n_jobs=command_opts.num_workers, return_as="generator",
    )(delayed(_etl)(product) for product in product_iter):
        if nat_filepath:  # Avoid adding None values
            nat_filepaths.append(nat_filepath)

    log.info(f"Finished population of zarr store: {command_opts.zarr_path}")

    if command_opts.validate:
        validate(dataset_path=command_opts.zarr_path)

    if isinstance(command_opts, ConsumeCommandOptions) and command_opts.latest_zip:
        zippath: str = create_latest_zip(dst=command_opts.zarr_path)
        log.info(f"Created latest.zip at {zippath}")

    if command_opts.delete_raw:
        if command_opts.workdir.startswith("s3://"):
            log.warning("delete-raw was specified, but deleting S3 files is not yet implemented")
        else:
            log.info(f"Deleting {len(nat_filepaths)} raw files in {command_opts.raw_folder}")
            for f in nat_filepaths:
                f.unlink()  # type: ignore

def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}', version: {__version__}, start_time: {prog_start}, opts: {config.command_options}"
    )

    if config.command in ["archive", "consume"]:
        _consume_command(command_opts=config.command_options)

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime}.")
