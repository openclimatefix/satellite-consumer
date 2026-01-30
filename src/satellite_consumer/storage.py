"""Storage module for reading and writing data to disk."""

import logging
import re
from typing import Any, TypeVar, overload

import fsspec
import gcsfs
import icechunk
import s3fs
import xarray as xr
import zarr
import zarr.codecs
import zarr.errors
import zarr.storage
from fsspec.implementations.local import LocalFileSystem
from icechunk.xarray import to_icechunk

log = logging.getLogger("sat_consumer")


T = TypeVar("T")


@overload
def _sanitize_encoding(ds: xr.Dataset, dims: list[str], data: dict[Any, Any]) -> dict[str, Any]: ...


@overload
def _sanitize_encoding[T](
    ds: xr.Dataset,
    dims: list[str],
    data: T,
) -> T: ...


def _sanitize_encoding(
    ds: xr.Dataset,
    dims: list[str],
    data: Any,
) -> Any:
    """Get the encoding dictionary for writing the dataset to Zarr."""
    if isinstance(data, dict):
        sanitized_data: dict[str, Any] = {}
        for key, value in data.items():
            if key in ["chunks", "shards"] and isinstance(value, list):
                # Replace all -1's with the correspoinding dimension length
                # Might not be chunked along all dims, hence strict=False
                sanitized_data[key] = [
                    cd[0] if cd[0] > 0 else len(ds.coords[cd[1]].values)
                    for cd in zip(value, dims, strict=False)
                ]
            elif key == "compressors":
                # Replace any mention of compressors with a standard Blosc Zstd compressor
                sanitized_data[key] = zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=3,
                    shuffle=zarr.codecs.BloscShuffle.bitshuffle,
                )
            elif key == "_ARRAY_DIMENSIONS":
                # Remove _ARRAY_DIMENSIONS key as it is not a valid Zarr encoding key, if exists
                continue
            else:
                sanitized_data[key] = _sanitize_encoding(ds=ds, dims=dims, data=value)

        return sanitized_data
    return data


def write_to_store(
    ds: xr.Dataset,
    dst: str | icechunk.repository.Repository,
    append_dim: str,
    encoding: dict[str, Any],
    write_new: bool,
) -> None:
    """Write the given dataset to the destination.

    If a store already exists at the given path, the dataset will be appended to it.
    """
    dims = encoding["_ARRAY_DIMENSIONS"]
    if dims != list(ds.dims):
        raise ValueError(
            "Provided dimensions do not match dataset dimensions."
            f" Provided: {dims}, Dataset: {list(ds.dims)}. "
            "Ensure process step outputs dimensions in the order specified here.",
        )

    if isinstance(dst, icechunk.repository.Repository):
        session = dst.writable_session(branch="main")
        if write_new:
            to_icechunk(
                obj=ds,
                session=session,
                mode="w-",
                encoding=_sanitize_encoding(ds=ds, dims=dims, data=encoding),
            )
            commit_message = "initial commit"
        else:
            to_icechunk(
                obj=ds,
                session=session,
                append_dim=append_dim,
                mode="a",
            )
            commit_message = f"add data for {ds.coords[append_dim].values}"

        session.commit(
            message=commit_message,
            rebase_with=icechunk.ConflictDetector(),
            rebase_tries=5,
        )

    elif isinstance(dst, str):
        if write_new:
            ds.to_zarr(
                store=dst,
                mode="w-",
                zarr_format=3,
                consolidated=False,
                write_empty_chunks=False,
                compute=True,
                encoding=_sanitize_encoding(ds=ds, dims=dims, data=encoding),
            )
        else:
            ds.to_zarr(
                store=dst,
                mode="a",
                append_dim=append_dim,
                zarr_format=3,
                consolidated=False,
                write_empty_chunks=False,
                compute=True,
            )


def get_existing_dataset(dst: str | icechunk.repository.Repository) -> xr.Dataset | None:
    """Get the dataset in the store if it exists."""
    try:
        if isinstance(dst, str):
            store_ds: xr.Dataset = xr.open_zarr(dst, consolidated=False)
        elif isinstance(dst, icechunk.repository.Repository):
            session = dst.readonly_session(branch="main")
            store_ds = xr.open_zarr(session.store, consolidated=False)
        return store_ds
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return None


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
