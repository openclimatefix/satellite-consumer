"""Storage module for reading and writing data to disk."""


import datetime as dt
import pathlib
from typing import cast

import numpy as np
import pyresample
import xarray as xr
import yaml
import zarr
from zarr.storage._zip import ZipStoreAccessModeLiteral


def write_to_zarr(
    da: xr.DataArray,
    zarr_path: pathlib.Path,
) -> None:
    """Write the given data array to the given zarr store.

    If a Zarr store already exists at the given path, the DataArray will be appended to it.

    Any attributes on the dataarray object are serialized to json-compatible strings.
    """
    mode: str = "a" if zarr_path.exists() else "w"
    extra_kwargs: dict[str, object] = {
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
            da.attrs[key] = value.dump() # type:ignore
        # Convert datetimes
        if isinstance(value, dt.datetime):
            da.attrs[key] = value.isoformat()

    store: zarr.storage.StoreLike = zarr.storage.LocalStore(zarr_path.as_posix())

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
            store=store,
            compute=True,
            mode=mode,
            consolidated=True,
            **extra_kwargs,
        )
    except Exception as e:
        raise OSError(f"Error writing dataset to zarr store {zarr_path}: {e}") from e

    return None

def create_latest_zip(zarr_path: str) -> str:
    """Convert a zarr store at the given path to a zip store."""
    ds: xr.Dataset = xr.open_zarr(zarr_path)
    zipstore: zarr.storage.ZipStore = zarr.storage.ZipStore(
        zarr_path.rsplit("/", 1)[0] + "/latest.zarr.zip",
        mode="w",
    )
    try:
        ds.to_zarr(store=zipstore, mode="a")
    except Exception as e:
        raise OSError(f"Error writing dataset to zip store {zipstore}: {e}") from e
    return zipstore.path.as_posix()


def _fname_to_scantime(fname: str) -> dt.datetime:
    """Converts a filename to a datetime.

    Files are of the form:
    `MSGX-SEVI-MSG15-0100-NA-20230910221240.874000000Z-NA.nat`
    So determine the time from the first element split by '.'.
    """
    return dt.datetime.strptime(fname.split(".")[0][-14:], "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)

