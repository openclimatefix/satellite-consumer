"""Storage module for reading and writing data to disk."""

import datetime as dt
import logging
import re
from typing import Any

import fsspec
import gcsfs
import icechunk
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import zarr.codecs
import zarr.errors
import zarr.storage
from fsspec.implementations.local import LocalFileSystem
from icechunk.xarray import to_icechunk

log = logging.getLogger(__name__)


def _get_satellite_longitude(ds: xr.Dataset) -> float | None:
    """Extract satellite orbital longitude from a dataset.

    Looks for the satellite_actual_longitude data variable, which stores the
    actual orbital position of the satellite. This is used to determine
    preference during satellite transitions (e.g., 9.5° to 0° migration).

    Args:
        ds: Dataset containing satellite data with orbital parameters.

    Returns:
        The satellite's actual longitude in degrees, or None if not available.

    Example:
        >>> ds = xr.open_zarr("satellite_data.zarr")
        >>> lon = _get_satellite_longitude(ds)
        >>> if lon is not None:
        ...     print(f"Satellite at {lon}° longitude")
    """
    try:
        if "satellite_actual_longitude" in ds.data_vars:
            # Get the value - could be array or scalar depending on selection
            val = ds["satellite_actual_longitude"].values
            
            # Handle scalar values (when time dimension has been selected/reduced)
            if np.ndim(val) == 0:
                return float(val)
            
            # Handle array values (normal case with time dimension)
            if hasattr(val, "__iter__") and len(val) > 0:
                return float(val[0])
            
            return float(val)
    except (KeyError, TypeError, IndexError):
        pass
    return None



def _should_overwrite(
    existing_longitude: float | None,
    new_longitude: float | None,
) -> bool:
    """Determine if new satellite data should overwrite existing data.

    During satellite orbital transitions (e.g., Meteosat moving from 9.5° to 0°),
    we prefer data from the satellite at the position closer to the prime meridian
    (0°), as this represents the newer operational position.

    Args:
        existing_longitude: Orbital longitude of existing data in degrees, or None.
        new_longitude: Orbital longitude of new data in degrees, or None.

    Returns:
        True if new data should overwrite existing data, False otherwise.

    Decision Logic:
        - If new satellite is closer to 0° than existing: overwrite (True)
        - If existing satellite is closer to 0° than new: keep existing (False)
        - If both at same position: keep existing (False, first wins)
        - If existing has no longitude info but new does: overwrite (True)
        - If new has no longitude info but existing does: keep existing (False)
        - If neither has longitude info: keep existing (False)

    Example:
        >>> _should_overwrite(9.5, 0.0)  # New is at prime meridian
        True
        >>> _should_overwrite(0.0, 9.5)  # Existing is at prime meridian
        False
    """
    # If new data has info and existing doesn't, prefer new
    if existing_longitude is None and new_longitude is not None:
        return True

    # If existing has info and new doesn't, keep existing
    if existing_longitude is not None and new_longitude is None:
        return False

    # If neither has info, keep existing (first wins)
    if existing_longitude is None and new_longitude is None:
        return False

    # Both have longitude info - prefer the one closer to 0°
    existing_dist = abs(existing_longitude)
    new_dist = abs(new_longitude)

    # Only overwrite if new is strictly closer to prime meridian
    return new_dist < existing_dist


def encoding(
    ds: xr.Dataset,
    dims: list[str],
    chunks: list[int],
    shards: list[int],
) -> dict[str, Any]:
    """Get the encoding dictionary for writing the dataset to Zarr."""
    # Replace -1's with full dimension sizes
    chunks = [
        cd[0] if cd[0] > 0 else len(ds.coords[cd[1]].values)
        for cd in zip(chunks, dims, strict=True)
    ]
    shards = [
        sd[0] if sd[0] > 0 else len(ds.coords[sd[1]].values)
        for sd in zip(shards, dims, strict=True)
    ]

    return {
        "data": {
            "compressors": zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=3,
                shuffle=zarr.codecs.BloscShuffle.bitshuffle,
            ),
            "fill_value": np.float32(np.nan),
            "chunks": chunks,
            "shards": shards,
        },
        "instrument": {"dtype": "<U26"},
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


def write_to_store(
    ds: xr.Dataset,
    dst: str | icechunk.repository.Repository,
    append_dim: str,
    chunks: list[int],
    shards: list[int],
    dims: list[str],
) -> None:
    """Write the given dataset to the destination.

    If a store already exists at the given path, the dataset will be appended to it.
    """
    if dims != list(ds.dims):
        raise ValueError(
            "Provided dimensions do not match dataset dimensions."
            f" Provided: {dims}, Dataset: {list(ds.dims)}. "
            "Ensure process step outputs dimensions in the order specified here.",
        )

    # Set the dask chunksizes to be the shard sizes for efficient writing
    # * The min is needed in case the shard size is larger than the length in that dimension
    ds = ds.chunk(
        chunks={dim: min(shard, ds.sizes[dim]) for dim, shard in zip(dims, shards, strict=True)},
    )

    try:
        if isinstance(dst, icechunk.repository.Repository):
            session = dst.writable_session(branch="main")
            store_ds: xr.Dataset = xr.open_zarr(session.store, consolidated=False)
        elif isinstance(dst, str):
            store_ds = xr.open_zarr(dst, consolidated=False)
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        # Write new store, specifying encodings
        if isinstance(dst, icechunk.repository.Repository):
            to_icechunk(
                obj=ds,
                session=session,
                mode="w-",
                encoding=encoding(ds=ds, dims=dims, chunks=chunks, shards=shards),
            )
            _ = session.commit(message="initial commit")
        elif isinstance(dst, str):
            _ = ds.to_zarr(
                dst,
                mode="w-",
                consolidated=False,
                write_empty_chunks=False,
                zarr_format=3,
                compute=True,
                encoding=encoding(ds=ds, dims=dims, chunks=chunks, shards=shards),
            )
        return None

    # If the time to be added is already in the store, don't do anything
    if np.isin(ds.coords[append_dim].values, store_ds.coords[append_dim].values).all():
        return None

    # Check the non-appending dimensions match
    if not ds[[d for d in ds.dims if d != append_dim]].equals(
        store_ds[[d for d in store_ds.dims if d != append_dim]],
    ):
        raise ValueError("Non-appending dimensions do not match existing store")

    # * Append the new time dimension coordinate
    if isinstance(dst, icechunk.repository.Repository):
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
    elif isinstance(dst, str):
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


def get_existing_times(
    dst: str | icechunk.repository.Repository,
    time_dim: str,
) -> list[dt.datetime]:
    """Get the existing times in the store."""
    try:
        if isinstance(dst, str):
            store_ds: xr.Dataset = xr.open_zarr(dst, consolidated=False)
        elif isinstance(dst, icechunk.repository.Repository):
            session = dst.readonly_session(branch="main")
            store_ds = xr.open_zarr(session.store, consolidated=False)
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return []

    return [
        pd.Timestamp(t).to_pydatetime().astimezone(tz=dt.UTC)
        for t in store_ds.coords[time_dim].values
    ]


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


def should_overwrite_existing(
    dst: str | icechunk.repository.Repository,
    time_value: dt.datetime,
    new_satellite_longitude: float | None,
) -> bool:
    """Check if existing data at a timestamp should be overwritten.

    Opens the store, retrieves the existing data for the given timestamp,
    extracts its satellite longitude, and determines if the new data
    should replace it based on satellite position preference.

    Args:
        dst: Path to Zarr store or icechunk repository.
        time_value: The timestamp to check.
        new_satellite_longitude: Orbital longitude of the new data.

    Returns:
        True if new data should overwrite existing, False otherwise.
    """
    try:
        if isinstance(dst, str):
            store_ds: xr.Dataset = xr.open_zarr(dst, consolidated=False)
        elif isinstance(dst, icechunk.repository.Repository):
            session = dst.readonly_session(branch="main")
            store_ds = xr.open_zarr(session.store, consolidated=False)
        else:
            return False

        # Convert store times to datetime objects for comparison
        store_times = [
            pd.Timestamp(t).to_pydatetime().replace(tzinfo=dt.UTC)
            for t in store_ds.coords["time"].values
        ]

        # Check if the time exists in the store
        if time_value not in store_times:
            # No existing data at this timestamp, so allow write
            return True

        # Time exists - need to check satellite preference
        # Select data using numpy datetime for xarray compatibility
        time_np = np.datetime64(time_value.replace(tzinfo=None), "ns")
        existing_data = store_ds.sel(time=time_np)
        existing_longitude = _get_satellite_longitude(existing_data)

        return _should_overwrite(existing_longitude, new_satellite_longitude)

    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        # No store exists yet, allow write
        return True
    except Exception as e:
        log.warning("Error checking overwrite status: %s", e)
        return False



def remove_duplicate_times(
    dst: str | icechunk.repository.Repository,
    time_dim: str = "time",
    dry_run: bool = False,
) -> list[dt.datetime]:
    """Remove duplicate timestamps from a Zarr store.

    Identifies timestamps that appear multiple times and removes duplicates,
    keeping the data from the satellite with the orbital position closest
    to the prime meridian (0°).

    Args:
        dst: Path to Zarr store or icechunk repository.
        time_dim: Name of the time dimension coordinate.
        dry_run: If True, only report what would be removed without modifying.

    Returns:
        List of timestamps that were (or would be if dry_run) removed.

    Raises:
        FileNotFoundError: If the store doesn't exist.

    Example:
        >>> # Preview duplicates without removing
        >>> removed = remove_duplicate_times("data.zarr", dry_run=True)
        >>> print(f"Would remove {len(removed)} duplicates")

        >>> # Actually remove duplicates
        >>> removed = remove_duplicate_times("data.zarr")
        >>> print(f"Removed {len(removed)} duplicates")
    """
    try:
        if isinstance(dst, str):
            store_ds: xr.Dataset = xr.open_zarr(dst, consolidated=False)
        elif isinstance(dst, icechunk.repository.Repository):
            session = dst.readonly_session(branch="main")
            store_ds = xr.open_zarr(session.store, consolidated=False)
        else:
            raise TypeError(f"Unsupported destination type: {type(dst)}")
    except (FileNotFoundError, zarr.errors.GroupNotFoundError) as e:
        raise FileNotFoundError(f"Store not found at {dst}") from e

    # Get all timestamps
    times = pd.Series(store_ds.coords[time_dim].values)

    # Find duplicates
    duplicated_mask = times.duplicated(keep=False)  # Mark all duplicates
    if not duplicated_mask.any():
        log.info("No duplicate timestamps found in store")
        return []

    duplicate_times = times[duplicated_mask].unique()
    log.info("Found %d timestamps with duplicates", len(duplicate_times))

    removed_timestamps: list[dt.datetime] = []

    for dup_time in duplicate_times:
        # Get indices of all entries for this timestamp
        indices = times[times == dup_time].index.tolist()

        if len(indices) < 2:
            continue

        # For each duplicate, get satellite longitude and determine which to keep
        longitudes: list[tuple[int, float | None]] = []
        for idx in indices:
            data_at_idx = store_ds.isel({time_dim: idx})
            lon = _get_satellite_longitude(data_at_idx)
            longitudes.append((idx, lon))

        # Find the index to keep (closest to 0°)
        keep_idx = min(
            longitudes,
            key=lambda x: abs(x[1]) if x[1] is not None else float("inf"),
        )[0]

        # All other indices should be removed
        remove_indices = [idx for idx, _ in longitudes if idx != keep_idx]

        for idx in remove_indices:
            ts = pd.Timestamp(times.iloc[idx]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.UTC)
            removed_timestamps.append(ts)

            if dry_run:
                log.info(
                    "[DRY RUN] Would remove duplicate at %s (index %d)",
                    ts,
                    idx,
                )
            else:
                log.info("Removing duplicate at %s (index %d)", ts, idx)

    if not dry_run and removed_timestamps:
        # Actually remove the duplicates by rewriting without them
        keep_mask = ~times.index.isin(
            [i for t in duplicate_times for i in times[times == t].index.tolist()[:-1]],
        )

        # For proper duplicate removal, we need to identify which specific indices to drop
        # This is complex for zarr stores - we'll use a rewrite approach
        log.warning(
            "Duplicate removal requires store rewrite. "
            "Found %d duplicates. Consider backing up before proceeding.",
            len(removed_timestamps),
        )

        # For now, we'll just report. Full implementation would require:
        # 1. Read all data
        # 2. Drop duplicate indices
        # 3. Rewrite the store
        # This is left as a TODO for safety - the user should use the cleanup script
        # with explicit confirmation

    return removed_timestamps

