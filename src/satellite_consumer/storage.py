"""Storage module for reading and writing data to disk."""

import datetime as dt
import os
import re
import tempfile
import warnings

import fsspec
import gcsfs
import icechunk
import numpy as np
import s3fs
import xarray as xr
import zarr
from fsspec.implementations.local import LocalFileSystem
from loguru import logger as log

from satellite_consumer.config import Coordinates


def write_to_zarr(
    ds: xr.Dataset,
    dst: str,
) -> None:
    """Write the given data array to the given zarr store.

    If a Zarr store already exists at the given path, the DataArray will be appended to it.

    Any attributes on the dataarray object are serialized to json-compatible strings.

    Args:
        da: The data array to write as a Zarr store.
        dst: The path to the Zarr store to write to. Can be a local filepath or S3 URL.
    """
    log.debug("Writing to store", coords=ds.coords.sizes, dst=dst)
    try:
        # TODO: Remove warnings catch when Zarr makes up its mind on time objects
        with warnings.catch_warnings(action="ignore"):
            store_ds: xr.Dataset = xr.open_zarr(dst, consolidated=False)

        time_idx: int = list(store_ds.coords["time"].values).index(ds.coords["time"].values[0])
        log.debug("Writing dataset to zarr store", dst=dst, time_idx=time_idx)

        with warnings.catch_warnings(action="ignore"):
            _ = ds.to_zarr(
                store=dst,
                compute=True,
                mode="a",
                consolidated=False,
                region={
                    "time": slice(time_idx, time_idx + 1),
                    "y_geostationary": slice(0, len(store_ds.coords["y_geostationary"])),
                    "x_geostationary": slice(0, len(store_ds.coords["x_geostationary"])),
                    "channel": "auto",
                },
            )
    except Exception as e:
        raise OSError(f"Error writing dataset to zarr store {dst}: {e}") from e

    return None


def create_latest_zip(src: str, time_slice: slice) -> str:
    """Extract the latest windowed data from the store and write it to a zipped zarr."""
    repo, _ = get_icechunk_repo(path=src)
    session: icechunk.Session = repo.readonly_session(branch="main")
    store_ds: xr.Dataset = xr.open_zarr(session.store, consolidated=False)
    fs = get_fs(src)
    dst = src.rsplit("/", 1)[0] + "/latest.zarr.zip"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".zarr.zip") as tmpzip:
        with zarr.storage.ZipStore(mode="w", path=tmpzip.name) as store:
            store_ds.isel({"time": time_slice}).to_zarr( #type: ignore
                store=store,
                consolidated=False,
                mode="w",
            )
        fs.put(tmpzip.name, dst)

    return dst


def create_empty_zarr(dst: str, coords: Coordinates) -> xr.DataArray:
    """Create an empty zarr store at the given path.

    Coordinate values are written to the zarr store as arrays.
    The array is initialized with NaN values.
    """
    group: zarr.Group = zarr.create_group(dst, overwrite=True)

    time_zarray: zarr.Array = group.create_array(
        name="time",
        dimension_names=["time"],
        shape=(len(coords.time),),
        dtype="int",
        attributes={
            "units": "nanoseconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
            "description": "The nominal end time of the image scan"
        },
    )
    time_zarray[:] = coords.time

    y_geo_zarray = group.create_array(
        name="y_geostationary",
        dimension_names=["y_geostationary"],
        shape=(len(coords.y_geostationary),),
        dtype="float32",
        attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    y_geo_zarray[:] = coords.y_geostationary

    x_geo_zarray = group.create_array(
        name="x_geostationary",
        dimension_names=["x_geostationary"],
        shape=(len(coords.x_geostationary),),
        dtype="float32",
        attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    x_geo_zarray[:] = coords.x_geostationary

    # TODO: Remove this when Zarr makes up its mind on string codecs
    with warnings.catch_warnings(action="ignore"):
        channel_zarray = group.create_array(
            name="channel",
            dimension_names=["channel"],
            shape=(len(coords.channel),),
            dtype="str",
        )
        channel_zarray[:] = coords.channel

        _ = group.create_array(
            name="instrument",
            dimension_names=["time"],
            shape=(len(coords.time),),
            dtype="str",
            attributes={"description": "Name of the satellite used"},
        )

    for coord in ["satellite_actual_longitude", "satellite_actual_latitude", "satellite_actual_altitude"]:
        _ = group.create_array(
            name=coord,
            dimension_names=["time"],
            shape=(len(coords.time),),
            dtype="float32",
            attributes={"description": "Actual as opposed to nominal postion of the satellite"},
        )

    for coord in ["cal_slope", "cal_offset"]:
        _ = group.create_array(
            name=coord,
            dimension_names=["time", "channel"],
            shape=(len(coords.time),len(coords.channel)),
            dtype="float32",
            attributes={"description": "Calibration paramaters for conversion from counts to radiance"},
        )

    _ = group.create_array(
        name="data",
        dimension_names=coords.dims(),
        dtype="float32",
        shape=coords.shape(),
        chunks=coords.chunks(),
        shards=coords.shards(),
        fill_value=np.nan,
        config={"write_empty_chunks": False},
    )

    with warnings.catch_warnings(action="ignore"):
        ds: xr.Dataset = xr.open_dataset(dst, engine="zarr", consolidated=False)

    log.debug("Created empty zarr store", dst=dst, coords=ds.coords.sizes)
    return ds.data_vars["data"]


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
    elif path.startswith("gcs://"):
        fs = gcsfs.GCSFileSystem(
            token=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None),
        )
    return fs


def get_icechunk_repo(path: str) -> tuple[icechunk.Repository, list[dt.datetime]]:
    """Get an icechunk repository for the given path.

    Args:
        path: The path to the icechunk repository. Use a protocol compatible with fsspec
            e.g. `s3://bucket-name/path/to/file` for remote access.

    Returns:
        A tuple containing the icechunk repository, and a list of times that exist in it already.
    """
    result = re.match(
        r"^(?P<protocol>[\w]{2,6}):\/\/(?P<bucket>[\w-]+)\/(?P<prefix>[\w.\/-]+)$",
        path,
    )
    storage_config: icechunk.Storage
    repo: icechunk.Repository

    # Make Icechunk storage config according to the given path
    if result:
        match (result.group("protocol"), result.group("bucket"), result.group("prefix")):
            case ("s3", bucket, prefix):
                log.debug("Initializing S3 backend", bucket=bucket, prefix=prefix)
                storage_config = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    access_key_id=os.getenv("AWS_ACCESS_KEY_ID", None),
                    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", None),
                    region=os.getenv("AWS_REGION", "eu-west-1"),
                    endpoint_url=os.getenv("AWS_ENDPOINT", None),
                )
            case ("gcs", bucket, prefix):
                log.debug("Initializing GCS backend", bucket=bucket, prefix=prefix)
                storage_config = icechunk.gcs_storage(
                    bucket=bucket,
                    prefix=prefix,
                    application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None),
                )
            case _:
                raise OSError(f"Unsupported protocol in path: {path}")
    else:
        # Try to do a local store
        log.debug("Initializing local filesystem backend", path=path)
        storage_config = icechunk.local_filesystem_storage(path=path)

    if icechunk.Repository.exists(storage=storage_config):
        # Return existing store and the times in it
        log.debug("Using existing icechunk store", path=path)
        repo = icechunk.Repository.open(storage=storage_config)
        ro_session = repo.readonly_session(branch="main")
        ds: xr.Dataset = xr.open_zarr(ro_session.store, consolidated=False)
        times: list[dt.datetime] = [
            dt.datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=dt.UTC)
            for t in np.datetime_as_string(ds.coords["time"].values, unit="m").tolist()
        ]
        return repo, times

    repo = icechunk.Repository.create(storage=storage_config)
    log.debug("Created new icechunk store", path=path)
    return repo, []
