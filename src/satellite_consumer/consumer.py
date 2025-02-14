"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import argparse
import datetime as dt
import logging
import pathlib
import traceback
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyresample
import xarray as xr
import yaml
from tqdm import tqdm

# Reduce verbosity of dependacies
for logger in [
    "cfgrib",
    "charset_normalizer",
    "eumdac", # If you want to know about throttling, set this to WARNING
    "native_msg",
    "pyorbital",
    "pyresample",
    "requests",
    "satpy",
    "urllib3",
]:
    logging.getLogger(logger).setLevel(logging.ERROR)
np.seterr(divide="ignore")

log = logging.getLogger("sat-etl")

def write_to_zarr(
    da: xr.DataArray,
    zarr_path: pathlib.Path,
) -> None:
    """Write the given data array to the given zarr store.

    If a Zarr store already exists at the given path, the DataArray will be appended to it.

    Any attributes on the dataarray object are serialized to json-compatible strings.
    """
    mode = "a" if zarr_path.exists() else "w"
    extra_kwargs: dict[str, Any] = {
        "append_dim": "time",
    } if mode == "a" else {
        "encoding": {
            "time": {"units": "nanoseconds since 1970-01-01"},
        },
    }
    # Convert attributes to be json	serializable
    for key, value in da.attrs.items():
        if isinstance(value, dict):
            # Convert np.float32 to Python floats (otherwise yaml.dump complains)
            for inner_key in value:
                inner_value = value[inner_key]
                if isinstance(inner_value, np.floating):
                    value[inner_key] = float(inner_value)
            da.attrs[key] = yaml.dump(value)
        if isinstance(value, bool | np.bool_):
            da.attrs[key] = str(value)
        if isinstance(value, pyresample.geometry.AreaDefinition):
            da.attrs[key] = value.dump()
        # Convert datetimes
        if isinstance(value, dt.datetime):
            da.attrs[key] = value.isoformat()

    try:
        _ = da.chunk({
            "time": 1,
            "x_geostationary": -1,
            "y_geostationary": -1,
            "variable": 1,
        }).to_dataset(
            name="data",
            promote_attrs=True,
        ).to_zarr(
            store=zarr_path,
            compute=True,
            consolidated=True,
            mode=mode,
            **extra_kwargs,
        )
    except Exception as e:
        log.error(f"Error writing dataset to zarr store {zarr_path} with mode {mode}: {e}")
        traceback.print_tb(e.__traceback__)

    return None

def _fname_to_scantime(fname: str) -> dt.datetime:
    """Converts a filename to a datetime.

    Files are of the form:
    `MSGX-SEVI-MSG15-0100-NA-20230910221240.874000000Z-NA.nat`
    So determine the time from the first element split by '.'.
    """
    return dt.datetime.strptime(fname.split(".")[0][-14:], "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)

def run(args: argparse.Namespace) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)
    log.info(f"{prog_start!s}: Running with args: {args}")

    # Get values from args
    folder: pathlib.Path = args.path
    sat_config = CONFIGS[args.sat]
    start: dt.datetime = dt.datetime.strptime(args.month, "%Y-%m").replace(tzinfo=dt.UTC)
    end: dt.datetime = (start + pd.DateOffset(months=1, minutes=-1)).to_pydatetime()
    dstype: Literal["hrv", "nonhrv"] = "hrv" if args.hrv else "nonhrv"

    product_iter, total = get_products_iterator(
        sat_config=sat_config,
        start=start,
        end=end,
        token=_gen_token(),
    )

    # Use existing zarr store if it exists
    ds: xr.Dataset | None = None
    zarr_path = folder / start.strftime(sat_config.zarr_fmtstr[dstype])
    if zarr_path.exists():
        log.info(f"Using existing zarr store at '{zarr_path}'")
        ds = xr.open_zarr(zarr_path, consolidated=True)

    # Iterate through all products in search
    nat_filepaths: list[pathlib.Path] = []
    for product in tqdm(product_iter, total=total, miniters=50):

        # Skip products already present in store
        if ds is not None:
            product_time: dt.datetime = product.sensing_start.replace(second=0, microsecond=0)
            if np.datetime64(product_time, "ns") in ds.coords["time"].values:
                log.debug(
                    f"Skipping entry '{product!s}' as '{product_time}' already in store",
                )
                continue

        # For non-existing products, download and process
        nat_filepath = download_nat(
            product=product,
            folder=folder / args.sat,
        )
        if nat_filepath is None:
            raise OSError(f"Failed to download product '{product}'")
        da = process_nat(nat_filepath, dstype)
        write_to_zarr(da=da, zarr_path=zarr_path)
        nat_filepaths.append(nat_filepath)

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

