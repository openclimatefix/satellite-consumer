"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
import math
import pathlib
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import xarray as xr
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
from satellite_consumer.storage import write_to_zarr, create_latest_zip
from satellite_consumer.validate import validate

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

def _consume_command(command_opts: ArchiveCommandOptions | ConsumeCommandOptions) -> None:
    """Run the download and processing pipeline."""
    window = command_opts.get_time_window()

    product_iter, total = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    # Use existing zarr store if it exists
    store_da: xr.DataArray | None = None
    zarr_path: str = command_opts.get_zarr_path()
    if pathlib.Path(zarr_path).exists():
        log.info(f"Using existing zarr store at '{zarr_path}'")
        store_da = xr.open_dataarray(zarr_path, engine="zarr", consolidated=True)

    # Iterate through all products in search
    nat_filepaths: list[pathlib.Path] = []
    for i, product in enumerate(product_iter): # Pretty sure enumerate is lazy
        product_time: dt.datetime = product.sensing_start.replace(second=0, microsecond=0)
        with log.contextualize(scan_time=str(product_time)):

            # Skip products already present in store
            if store_da is not None \
                and np.datetime64(product_time, "ns") in store_da.coords["time"].values:
                log.debug("Skipping already present entry in store")
                continue

            # For non-existing products, download and process
            nat_filepath = download_nat(
                product=product,
                folder=pathlib.Path(command_opts.workdir) / "raw",
            )

            da = process_nat(path=nat_filepath, hrv=command_opts.hrv)
            write_to_zarr(da=da, zarr_path=pathlib.Path(zarr_path))
            nat_filepaths.append(nat_filepath)
            if i % math.ceil(total / 10) == 0:
                log.info(f"Processed {i+1} of {total} products.")

    if command_opts.validate:
        validate(dataset_path=zarr_path)

    if isinstance(command_opts, ConsumeCommandOptions) and command_opts.latest_zip:
        zippath: str = create_latest_zip(zarr_path=zarr_path)
        log.info(f"Created latest.zip at {zippath}")

    if command_opts.delete_raw:
        log.info(
            f"Deleting {len(nat_filepaths)} raw files in "
            "{command_opts.get_raw_folder()}.",
        )
        _ = [f.unlink() for f in nat_filepaths] # type:ignore


def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command {config.command}",
        version=__version__, start_time=str(prog_start),
    )

    if config.command == "archive" or config.command == "consume":
        _consume_command(command_opts=config.command_options)

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")

