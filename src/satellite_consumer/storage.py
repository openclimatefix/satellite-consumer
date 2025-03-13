"""Storage module for reading and writing data to disk."""


import datetime as dt
import os
import tempfile

import fsspec
import numpy as np
import pyresample
import s3fs
import xarray as xr
import yaml
import zarr
from fsspec.implementations.local import LocalFileSystem
from loguru import logger as log

from satellite_consumer.config import Coordinates


def write_to_zarr(
    da: xr.DataArray,
    dst: str,
    ) -> None:
    """Write the given data array to the given zarr store.

    If a Zarr store already exists at the given path, the DataArray will be appended to it.

    Any attributes on the dataarray object are serialized to json-compatible strings.

    Args:
        da: The data array to write as a Zarr store.
        dst: The path to the Zarr store to write to. Can be a local filepath or S3 URL.
    """
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

    try:
        store_da: xr.DataArray = xr.open_dataarray(dst, engine="zarr", consolidated=False)
        time_idx: int = list(store_da.coords["time"].values).index(da.coords["time"].values[0])
        log.debug("Writing dataarray to zarr store", dst=dst, time_idx=time_idx)
        _ = da.to_dataset(name="data", promote_attrs=True).to_zarr(
            store=dst, compute=True, mode="a", consolidated=False,
            region={
                "time": slice(time_idx, time_idx + 1),
                "y_geostationary": slice(0, len(store_da.coords["y_geostationary"])),
                "x_geostationary": slice(0, len(store_da.coords["x_geostationary"])),
                "variable": "auto",
            },
        )
    except Exception as e:
        raise OSError(f"Error writing dataset to zarr store {dst}: {e}") from e

    return None

def create_latest_zip(dsts: list[str]) -> str:
    """Convert zarr store(s) at the given path to a zip store."""
    fs = get_fs(path=dsts[0])

    # Open the zarr store and write it to a zip store
    ds: xr.Dataset = xr.open_mfdataset(
        dsts, consolidated=False, concat_dim="time", combine="nested",
    )

    zippath: str = dsts[0].rsplit("/", 1)[0] + "/latest.zarr.zip"
    with tempfile.NamedTemporaryFile(suffix=".zip") as fsrc,\
        zarr.storage.ZipStore(path=fsrc.name, mode="w") as store:
        try:
            _ = ds.to_zarr(store=store) # type: ignore
            fs.put(lpath=fsrc.name, rpath=zippath, overwrite=True)
        except Exception as e:
            raise OSError(f"Error writing dataset to zip store '{zippath}': {e}") from e
    return zippath


def create_empty_zarr(dst: str, coords: Coordinates) -> xr.DataArray:
    """Create an empty zarr store at the given path.

    Coordinate values are written to the zarr store as arrays.
    The array is initialized with NaN values.
    """
    group: zarr.Group = zarr.create_group(dst, overwrite=True)

    time_zarray: zarr.Array = group.create_array(
        name="time", dimension_names=["time"],
        shape=(len(coords.time),), dtype="int", attributes={
            "units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian",
        },
    )
    time_zarray[:] = coords.time

    y_geo_zarray = group.create_array(
        name="y_geostationary", dimension_names=["y_geostationary"],
        shape=(len(coords.y_geostationary),), dtype="float", attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    y_geo_zarray[:] = coords.y_geostationary

    x_geo_zarray = group.create_array(
        name="x_geostationary", dimension_names=["x_geostationary"],
        shape=(len(coords.x_geostationary),), dtype="float", attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    x_geo_zarray[:] = coords.x_geostationary

    var_zarray = group.create_array(
        name="variable", dimension_names=["variable"], shape=(len(coords.variable),), dtype="str",
    )
    var_zarray[:] = coords.variable

    _ = group.create_array(
        name="data", dimension_names=coords.dims(), dtype="float",
        shape=coords.shape(), chunks=coords.chunks(),
        fill_value=np.nan, config={"write_empty_chunks": False},
    )

    da = xr.open_dataarray(dst, engine="zarr", consolidated=False)
    return da


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    """Get relevant filesystem for the given path.

    Args:
        path: The path to get the filesystem for. Use a protocol compatible with fsspec
            e.g. `s3://bucket-name/path/to/file` for remote access.
    """
    fs: fsspec.AbstractFileSystem = LocalFileSystem(auto_mkdir=True)
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(
            anon=False,
            key=os.getenv("AWS_ACCESS_KEY_ID", None),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY", None),
            client_kwargs={
                "region_name": os.getenv("AWS_REGION", "eu-west-1"),
                "endpoint_url": os.getenv("AWS_ENDPOINT", None),
            },
        )
    return fs

