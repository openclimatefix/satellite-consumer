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

from satellite_consumer.config import SatelliteConsumerConfig
from satellite_consumer.download_eumetsat import (
    download_nat,
    get_products_iterator,
)
from satellite_consumer.process import process_nat
from satellite_consumer.storage import write_to_zarr

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command {config.command}",
        version=__version__, start_time=prog_start,
    )

    if config.command == "archive" or config.command == "consume":
        window = config.command_options.get_time_window()

        product_iter, total = get_products_iterator(
            sat_metadata=config.command_options.satellite_metadata,
            start=window[0],
            end=window[1],
        )

        # Use existing zarr store if it exists
        ds: xr.Dataset | None = None
        zarr_path: str = config.command_options.get_zarr_path()
        if pathlib.Path(zarr_path).exists():
            log.info(f"Using existing zarr store at '{zarr_path}'")
            ds = xr.open_zarr(zarr_path, consolidated=True)

        # Iterate through all products in search
        nat_filepaths: list[pathlib.Path] = []
        for i, product in enumerate(product_iter): # Pretty sure enumerate is lazy
            product_time: dt.datetime = product.sensing_start.replace(second=0, microsecond=0)
            with log.contextualize(product=product, scan_time=product_time):

                # Skip products already present in store
                if ds is not None and np.datetime64(product_time, "ns") in ds.coords["time"].values:
                    log.debug("Skipping already present entry in store")
                    continue

                # For non-existing products, download and process
                nat_filepath = download_nat(
                    product=product,
                    folder=pathlib.Path(config.command_options.workdir) / "raw",
                )
                if nat_filepath is None:
                    raise OSError(f"Failed to download product '{product}'")

                da = process_nat(nat_filepath, dstype)
                write_to_zarr(da=da, zarr_path=pathlib.Path(zarr_path))
                nat_filepaths.append(nat_filepath)
                if i % math.ceil(total / 10) == 0:
                    log.info(f"Processed {i} of {total} products.")

        runtime = dt.datetime.now(tz=dt.UTC) - prog_start
        log.info(f"Completed archive for args: {args} in {runtime!s}.")

        if args.validate:
            ds = xr.open_zarr(zarr_path, consolidated=True)
            check_data_quality(ds)

        # Delete raw files, if desired
        if args.delete_raw:
            log.info(f"Deleting {len(nat_filepaths)} raw files in {folder.as_posix()}.")
            for f in nat_filepaths:
                f.unlink()


if __name__ == "__main__":
    # Parse running args
    args = parser.parse_args()
    run(args)

