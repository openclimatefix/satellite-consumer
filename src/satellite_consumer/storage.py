"""Storage module for reading and writing data to disk."""


import datetime as dt
import os
import shutil
import tempfile
import warnings

import numpy as np
import pyresample
import xarray as xr
import yaml
import zarr
from loguru import logger as log
from obstore.fsspec import FsspecStore

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
        # TODO: Remove warnings catch when Zarr makes up its mind on time objects
        with warnings.catch_warnings(action="ignore"):
            store_da: xr.DataArray = xr.open_dataarray(dst, engine="zarr", consolidated=False)

        time_idx: int = list(store_da.coords["time"].values).index(da.coords["time"].values[0])
        log.debug("Writing dataarray to zarr store", dst=dst, time_idx=time_idx)

        with warnings.catch_warnings(action="ignore"):
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

def create_latest_zip(srcs: list[str]) -> str:
    """Convert zarr store(s) at the given path to a zip store."""
    # Open the zarr store and write it to a zip store
    fs = get_fs(srcs[0])
    zippath: str = srcs[0].rsplit("/", 1)[0] + "/latest.zarr.zip"

    dst: str = zippath.split("s3://")[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, src in enumerate(srcs):
            try:
                store: str = tmpdir + "/merged.zarr"
                with warnings.catch_warnings(action="ignore"):
                    ds = xr.open_zarr(src, consolidated=False)
                    _ = ds.to_zarr(
                        store=store,
                        consolidated=False,
                        mode="w" if i==0 else "a-",
                        append_dim=None if i==0 else "time",
                    )
                log.debug(
                    "Written dataset to merged store",
                    src=src, num=f"{i+1}/{len(srcs)}",
                )
            except Exception as e:
                raise OSError(f"Error writing dataset to merged store '{store}': {e}") from e
        shutil.make_archive(
            base_name=tmpdir+"/merged_zipped",
            format="zip",
            root_dir=tmpdir,
            base_dir="merged.zarr",
        )
        log.debug("Zipped merged store")
        fs.put(lpath=tmpdir+"/merged_zipped.zip", rpath=dst)

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

    # TODO: Remove this when Zarr makes up its mind on string codecs
    with warnings.catch_warnings(action="ignore"):
        var_zarray = group.create_array(
            name="variable", dimension_names=["variable"],
            shape=(len(coords.variable),), dtype="str",
        )
        var_zarray[:] = coords.variable

    _ = group.create_array(
        name="data", dimension_names=coords.dims(), dtype="float",
        shape=coords.shape(), chunks=coords.chunks(), shards=coords.shards(),
        fill_value=np.nan, config={"write_empty_chunks": False},
    )

    with warnings.catch_warnings(action="ignore"):
        da = xr.open_dataarray(dst, engine="zarr", consolidated=False)
    return da


def get_fs(path: str) -> FsspecStore:
    """Get relevant filesystem for the given path.

    Args:
        path: The path to get the filesystem for. Use a protocol compatible with obstore
            e.g. `s3://bucket-name/path/to/file` for remote access.
    """
    if path.startswith("s3://"):
        # Build S3 filesystem with obstore
        region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "eu-west-1"))
        endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT")
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        return FsspecStore(  # type: ignore[call-overload,no-any-return]
            "s3",
            region=region,
            endpoint=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )
    else:
        return FsspecStore("file")

