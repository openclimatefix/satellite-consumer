"""Storage module for reading and writing data to disk."""

import logging
import re
from typing import Any

import fsspec
import gcsfs
import icechunk
import numpy as np
import s3fs
import xarray as xr
import zarr
import zarr.codecs
import zarr.storage
from fsspec.implementations.local import LocalFileSystem
from icechunk.xarray import to_icechunk

log = logging.getLogger(__name__)

def encoding(ds: xr.Dataset) -> dict[str, Any]:
    """Get the encoding dictionary for writing the dataset to Zarr."""
    return {
        "data": {
            "compressors": zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=3,
                shuffle=zarr.codecs.BloscShuffle.bitshuffle,
            ),
            "fill_value": np.float32(np.nan),
            "chunks": (1, 400, 400, len(ds.coords["channel"].values)),
        },
        "instrument": {"dtype": "str"},
        "satellite_actual_longitude": {"dtype": "float32"},
        "satellite_actual_latitude": {"dtype": "float32"},
        "satellite_actual_altitude": {"dtype": "float32"},
        # Coordinates
        "channel": {"dtype": "str"},
        "time": {
            "dtype": "int",
            "units": "nanoseconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
        },
    }


def write_to_zarr(
    ds: xr.Dataset,
    dst: str,
    append_dim: str = "time",
) -> None:
    """Write the given dataset to the destination.

    If a Zarr store already exists at the given path, the dataset will be appended to it.
    """
    try:
        store_ds = xr.open_zarr(dst, consolidated=False)
        # If the time to be added is already in the store, don't do anything
        if np.isin(ds.coords[append_dim].values, store_ds.coords[append_dim].values).all():
            return None

    except FileNotFoundError:
        # Write new store, specifying encodings
        _ = ds.to_zarr(
            dst,
            mode="w-",
            consolidated=False,
            write_empty_chunks=False,
            zarr_format=3,
            compute=True,
            encoding=encoding(ds),
        )
        return None

    # Write the data to the existing store safely, failing if the non-appending dimensions differ
    # * Check the non-appending dimensions match via a non-computed write
    if not ds[[d for d in ds.dims if d != append_dim]].equals(
        store_ds[[d for d in store_ds.dims if d != append_dim]],
    ):
        raise ValueError("Non-appending dimensions do not match existing store")
    # * Append the new time dimension coordinate
    _ = ds.to_zarr(
        store=dst,
        mode="a",
        append_dim=append_dim,
        compute=True,
        write_empty_chunks=False,
        zarr_format=3,
        consolidated=False,
    )

    return None

def write_to_icechunk(
    ds: xr.Dataset,
    repo: icechunk.repository.Repository,
    branch: str = "main",
    append_dim: str = "time",
) -> None:
    """Write the given dataset to the icechunk repository.

    If a Zarr store already exists at the given path, the dataset will be appended to it.
    """
    session = repo.writable_session(branch=branch)
    if repo.exists(repo.storage):
        store_ds: xr.Dataset = xr.open_zarr(session.store, consolidated=False)
        # If the time to be added is already in the store, don't do anything
        if np.isin(ds.coords[append_dim].values, store_ds.coords[append_dim].values).all():
            return None
        to_icechunk(
            obj=ds,
            session=session,
            append_dim=append_dim,
            mode="a",
        )
        _ = session.commit(
            message=f"add data for {ds.coords[append_dim].values}",
            rebase_with=icechunk.ConflictDetector(),
            rebase_tries=5,
        )
    else:
        to_icechunk(
            obj=ds,
            session=session,
            mode="w-",
            encoding=encoding(ds),
        )
        _ = session.commit(message="initial commit")

def get_fs(
    path: str,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region_name: str | None = None,
    aws_endpoint_url: str | None = None,
    gcs_token: str | None = None,
) -> fsspec.AbstractFileSystem:
    """Get relevant filesystem for the given path.

    Args:
        path: The path to get the filesystem for. Use a protocol compatible with fsspec
            e.g. `s3://bucket-name/path/to/file` for remote access.
        aws_access_key_id: AWS access key ID for S3 access.
        aws_secret_access_key: AWS secret access key for S3 access.
        aws_region_name: AWS region name for S3 access.
        aws_endpoint_url: AWS endpoint URL for S3 access.
        gcs_token: GCS token for GCS access.
    """
    fs: fsspec.AbstractFileSystem = LocalFileSystem(auto_mkdir=True)
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(
            anon=False,
            key=aws_access_key_id,
            secret=aws_secret_access_key,
            client_kwargs={
                "region_name": aws_region_name,
                "endpoint_url": aws_endpoint_url,
            },
        )
    elif path.startswith("gcs://"):
        fs = gcsfs.GCSFileSystem(
            token=gcs_token,
        )
    return fs


def get_icechunk_repo(
    path: str,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region_name: str | None = None,
    aws_endpoint_url: str | None = None,
    gcs_token: str | None = None,
) -> icechunk.Repository:
    """Get an icechunk repository for the given path.

    Args:
        path: The path to the icechunk repository. Use a protocol compatible with fsspec
            e.g. `s3://bucket-name/path/to/file` for remote access.
        aws_access_key_id: AWS access key ID for S3 access.
        aws_secret_access_key: AWS secret access key for S3 access.
        aws_region_name: AWS region name for S3 access.
        aws_endpoint_url: AWS endpoint URL for S3 access.
        gcs_token: GCS token for GCS access.
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
                log.debug("bucket=%s, prefix=%s, initializing S3 backend", bucket, prefix)
                storage_config = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key,
                    region=aws_region_name,
                    endpoint_url=aws_endpoint_url,
                )
            case ("gcs", bucket, prefix):
                log.debug("bucket=%s, prefix=%s, initializing GCS backend", bucket, prefix)
                storage_config = icechunk.gcs_storage(
                    bucket=bucket,
                    prefix=prefix,
                    application_credentials=gcs_token,
                )
            case _:
                raise OSError(f"Unsupported protocol in path: {path}")
    else:
        # Try to do a local store
        log.debug("path=%s, initializing local filesystem backend", path)
        storage_config = icechunk.local_filesystem_storage(path=path)

    if icechunk.Repository.exists(storage=storage_config):
        # Return existing store and the times in it
        log.debug("path=%s, using existing icechunk store", path)
        repo = icechunk.Repository.open(storage=storage_config)
        return repo

    repo = icechunk.Repository.create(storage=storage_config)
    log.debug("path=%s, created new icechunk store", path)
    return repo
