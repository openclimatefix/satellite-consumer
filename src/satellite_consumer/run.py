"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
from importlib.metadata import PackageNotFoundError, version

import eumdac.product
from joblib import Parallel, delayed
from loguru import logger as log

from satellite_consumer.config import (
    ConsumeCommandOptions,
    MergeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_nat,
    get_products_iterator,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_nat
from satellite_consumer.storage import create_empty_zarr, create_latest_zip, get_fs, write_to_zarr
from satellite_consumer.validate import validate

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


def _consume_to_store(command_opts: ConsumeCommandOptions) -> None:
    """Logic for the consume command (and the archive command)."""
    fs = get_fs(path=command_opts.zarr_path)

    window = command_opts.time_window

    product_iter = get_products_iterator(
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
        _ = create_empty_zarr(dst=command_opts.zarr_path, coords=command_opts.as_coordinates())

    def _etl(product: eumdac.product.Product) -> str | None:
        """Download, process, and save a single NAT file."""
        nat_filepath = download_nat(product, folder=f"{command_opts.workdir}/raw")
        if nat_filepath is None:
            return nat_filepath
        da = process_nat(path=nat_filepath, hrv=command_opts.hrv)
        write_to_zarr(da=da, dst=command_opts.zarr_path)
        return nat_filepath

    nat_filepaths: list[str] = []
    num_skipped: int = 0
    # Iterate through all products in search
    for nat_filepath in Parallel(
        n_jobs=command_opts.num_workers, return_as="generator",
        prefer="threads",
    )(delayed(_etl)(product) for product in product_iter):
        if nat_filepath is None:
            num_skipped += 1
        else:
            nat_filepaths.append(nat_filepath)

    # Might not need this as skipped files are all NaN
    # and the validation step should catch it
    if num_skipped / (num_skipped + len(nat_filepaths)) > 0.05:
        raise ValidationError(
            f"Too many files had to be skipped "
            f"({num_skipped}/{num_skipped + len(nat_filepaths)}). "
            "Use dataset at your own risk!",
        )

    log.info(
        "Finished population of zarr store",
        dst=command_opts.zarr_path, num_skipped=num_skipped,
    )

    if command_opts.validate:
        validate(src=command_opts.zarr_path)

    if command_opts.delete_raw:
        if command_opts.workdir.startswith("s3://"):
            log.warning("delete-raw was specified, but deleting S3 files is not yet implemented")
        else:
            log.info(
                f"Deleting {len(nat_filepaths)} raw files in {command_opts.raw_folder}",
                num_files=len(nat_filepaths), dst=command_opts.raw_folder,
            )
            _ = [f.unlink() for f in nat_filepaths] # type:ignore

def _merge_command(command_opts: MergeCommandOptions) -> None:
    """Logic for the merge command."""
    zarr_paths = command_opts.zarr_paths
    log.info(
        f"Merging {len(zarr_paths)} stores",
        num=len(zarr_paths), consume_missing=command_opts.consume_missing,
    )
    fs = get_fs(path=zarr_paths[0])

    for zarr_path in zarr_paths:
        if not fs.exists(zarr_path):
            if command_opts.consume_missing:
                _consume_to_store(command_opts=ConsumeCommandOptions(
                    time=dt.datetime.strptime(
                        zarr_path.split("/")[-1].split("_")[0], "%Y%m%dT%H%M",
                    ).replace(tzinfo=dt.UTC),
                    satellite=command_opts.satellite,
                    workdir=command_opts.workdir,
                    validate=True,
                    rescale=True, # TODO: Make this an option
                    hrv=command_opts.hrv,
                ))
            else:
                raise FileNotFoundError(f"Zarr store not found at {zarr_path}")

    dst = create_latest_zip(srcs=zarr_paths)
    log.info("Created latest.zip", dst=dst)


def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}'",
        version=__version__, start_time=str(prog_start), opts=config.command_options.__str__(),
    )

    if isinstance(config.command_options, ConsumeCommandOptions):
        _consume_to_store(command_opts=config.command_options)
    elif isinstance(config.command_options, MergeCommandOptions):
        _merge_command(command_opts=config.command_options)
    else:
        pass

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")

