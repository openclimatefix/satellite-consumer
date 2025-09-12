"""Storage module for reading and writing data to disk."""

import datetime as dt
import os
import re
import tempfile
import warnings

import gcsfs
import icechunk
import numpy as np
import xarray as xr
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
    log.debug("Writing to store", coords=da.coords.sizes, dst=dst)
    try:
        # TODO: Remove warnings catch when Zarr makes up its mind on time objects
        with warnings.catch_warnings(action="ignore"):
            store_da: xr.DataArray = xr.open_dataarray(dst, engine="zarr", consolidated=False)

        time_idx: int = list(store_da.coords["time"].values).index(da.coords["time"].values[0])
        log.debug("Writing dataarray to zarr store", dst=dst, time_idx=time_idx)

        with warnings.catch_warnings(action="ignore"):
            _ = da.to_dataset(name="data", promote_attrs=True).to_zarr(
                store=dst,
                compute=True,
                mode="a",
                consolidated=False,
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
        },
    )
    time_zarray[:] = coords.time

    y_geo_zarray = group.create_array(
        name="y_geostationary",
        dimension_names=["y_geostationary"],
        shape=(len(coords.y_geostationary),),
        dtype="float",
        attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    y_geo_zarray[:] = coords.y_geostationary

    x_geo_zarray = group.create_array(
        name="x_geostationary",
        dimension_names=["x_geostationary"],
        shape=(len(coords.x_geostationary),),
        dtype="float",
        attributes={
            "coordinate_reference_system": "geostationary",
        },
    )
    x_geo_zarray[:] = coords.x_geostationary

    # TODO: Remove this when Zarr makes up its mind on string codecs
    with warnings.catch_warnings(action="ignore"):
        var_zarray = group.create_array(
            name="variable",
            dimension_names=["variable"],
            shape=(len(coords.variable),),
            dtype="str",
        )
        var_zarray[:] = coords.variable

    _ = group.create_array(
        name="data",
        dimension_names=coords.dims(),
        dtype="float",
        shape=coords.shape(),
        chunks=coords.chunks(),
        shards=coords.shards(),
        fill_value=np.nan,
        config={"write_empty_chunks": False},
    )

    with warnings.catch_warnings(action="ignore"):
        da = xr.open_dataarray(dst, engine="zarr", consolidated=False)

    log.debug("Created empty zarr store", dst=dst, coords=da.coords.sizes)
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
    elif path.startswith("gcs://"):
        # Use gcsfs for GCS (obstore GCS support via FsspecStore not fully compatible yet)
        return gcsfs.GCSFileSystem(
            token=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None),
        )
    else:
        return FsspecStore("file")


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
