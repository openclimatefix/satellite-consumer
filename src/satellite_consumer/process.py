"""Functions for processing satellite data."""

import contextlib
import ctypes
import ctypes.util
import datetime as dt
import gc
import importlib.metadata
import logging
import os
import warnings
from collections.abc import Mapping
from typing import Any

import dask
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import yaml
from pyresample.geometry import AreaDefinition
from satpy.readers.seviri_l1b_native import NativeMSGFileHandler
from satpy.scene import Scene

from satellite_consumer import models
from satellite_consumer.exceptions import DownloadError, ValidationError

log = logging.getLogger("sat_consumer")


def _dummy_add_scanline_acq_time(
    self: NativeMSGFileHandler,
    *args: object,
    **kwargs: object,
) -> None:
    """Dummy function to patch satpy for speed.

    The private `NativeMSGFileHandler._add_scanline_acq_time()` method takes about 1 second to run
    per image, and is run each time the `scene.load()` method is called. This method adds the
    `"acq_time"` variable to the dataset which we drop within `_map_scene_to_dataset()` anyway.
    Also, nothing else in the processing pipeline depends on `"acq_time"` (as of satpy 0.59.0), so
    we can safely remove this to gain a speed-up.
    """
    pass


# Patch the method to nullify it
NativeMSGFileHandler._add_scanline_acq_time = (  # type: ignore[method-assign]
    _dummy_add_scanline_acq_time
)

#: Threads one image is computed with. Each thread holds satpy's intermediates for the
#: chunk it is working on (several arrays of up to dask's ``array.chunk-size`` each), so
#: dask's default of one thread per core made the transient peak of a full-disc image scale
#: with the host's core count. 0 keeps dask's default.
DASK_THREADS = int(os.environ.get("SATCONS_DASK_THREADS", "8"))


def store_dtypes(
    encoding: Mapping[str, Any],
    channels: list[models.SpectralChannel],
) -> dict[str, np.dtype]:
    """The dtype each channel is stored as, where the encoding narrows it below float32.

    The image is cast to it while it is computed, chunk by chunk, so a full-disc image is
    never resident at float32. The cast is the one xarray applies when it encodes the
    float32 array for the store, so the stored values are unchanged.
    """
    out: dict[str, np.dtype] = {}
    for channel in channels:
        spec = encoding.get(channel.name, None)
        dtype = spec.get("dtype", None) if spec is not None else None
        if dtype is None:
            continue
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.floating) and dtype.itemsize < np.dtype(np.float32).itemsize:
            out[channel.name] = dtype
    return out


def _compute_threads() -> contextlib.AbstractContextManager[Any]:
    """Bound the threads a ``.load()`` uses to :data:`DASK_THREADS`."""
    if DASK_THREADS <= 0:
        return contextlib.nullcontext()
    return dask.config.set(scheduler="threads", num_workers=DASK_THREADS)


def process_raw(
    paths: list[str],
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_lonlat: tuple[float, float, float, float] | None = None,
    source: str = "eumetsat",
    store_dtypes: Mapping[str, np.dtype] | None = None,
    collect_garbage: bool = False,
) -> xr.Dataset:
    """Process a set of raw files into an xarray DataArray.

    Args:
        paths: List of file paths to raw satellite data files.
        channels: List of channels to extract.
        resolution_meters: Desired spatial resolution in meters.
        crop_region_lonlat: Optional tuple defining the lon-lat coordinate
            region to crop to, in the form (lon_min, lat_min, lon_max, lat_max).
        source: The source the data is served from, as declared by the satellite's
            `source` key in `application.conf`. One of 'eumetsat', 'goes',
            'himawari', or 'gk2a'.
        store_dtypes: Channels to compute straight into a narrower float dtype, the one
            they are stored as; see :func:`store_dtypes`.
        collect_garbage: Free the Scene as soon as the image is built (see below). Only
            safe when no other image is being read at the same time: collecting closes the
            Scene's netCDF files, and netCDF-C is not thread-safe, so a close racing another
            thread's read fails that read ("NetCDF: Not a valid ID") or silently drops a
            segment from its image.
    """
    try:
        # Meteosat 3rd gen don't output .nat files, and so requires a different loader
        reader_kwargs: dict[str, Any] = {}
        loader: str
        match source:
            case "eumetsat":
                loader = "fci_l1c_nc"
                if paths[0].endswith(".nat"):
                    loader = "seviri_l1b_native"
                    # Nominal calibration represents raw integer counts to radiance via slope and
                    # intercept. Include flag surfaces calibration values
                    reader_kwargs = {
                        "calib_mode": "nominal",
                        "include_raw_metadata": True,
                        "fill_disk": True
                    }
            case "goes":
                loader = "abi_l1b"
            case "himawari":
                loader = "ahi_hsd"
            case "gk2a":
                loader = "ami_l1b"
            case _:
                raise ValueError(
                    f"Unsupported source: {source}. Supported sources are:"
                    f" 'eumetsat', 'goes', 'himawari', 'gk2a'.",
                )
        scene: Scene = Scene(
            filenames={loader: paths},  # type:ignore
            reader_kwargs=reader_kwargs,
        )
        scene.load(  # type: ignore
            wishlist=[c.name for c in channels],
            # The resolution arg has to be "*" for 3000m, for some reason
            resolution=resolution_meters if resolution_meters < 3000 and "HRV" not in [c.name for c in channels] else "*",
            calibration=[c.representation for c in channels],
        )

    except ValueError as e:
        # There seem to be corrupted (short) files in the EUMETSAT Data Store archive
        # Catch this error for the shorter than expected files
        if "buffer is smaller than requested size" in str(e):
            raise ValidationError(f"Truncated satellite file(s) {paths} detected: {e}") from e
        raise DownloadError(f"Error reading paths as satpy Scene: {e}") from e

    except Exception as e:
        raise DownloadError(f"Error reading paths as satpy Scene: {e}") from e

    ds: xr.Dataset = _map_scene_to_dataset(
        scene=scene,
        channels=channels,
        crop_region_lonlat=crop_region_lonlat,
        store_dtypes=store_dtypes,
    )

    # The Scene and its file handlers sit in reference cycles that keep the decoded
    # segments alive (~2.8 GB for a 500 m FCI disc) until Python's next full collection,
    # by when several images' worth have piled up. Free them now, then hand the freed
    # pages back to the OS.
    del scene
    if collect_garbage:
        gc.collect()
        release_freed_memory()

    return ds


_LIBC = ctypes.CDLL(ctypes.util.find_library("c")) if ctypes.util.find_library("c") else None


def release_freed_memory() -> None:
    """Return freed heap pages to the OS (glibc ``malloc_trim``); a no-op elsewhere.

    glibc keeps freed memory in its per-thread arenas, so a process that decodes and
    drops a multi-GB image every few seconds otherwise holds its peak indefinitely.
    """
    trim = getattr(_LIBC, "malloc_trim", None)
    if trim is not None:
        trim(0)


def _map_scene_to_dataset(
    scene: Scene,
    channels: list[models.SpectralChannel],
    crop_region_lonlat: tuple[float, float, float, float] | None = None,
    store_dtypes: Mapping[str, np.dtype] | None = None,
) -> xr.Dataset:
    """Converts a Scene with satellite data into a data array.

    Note!!!:
        This function fudges the timestamps somewhat!
        The time coordinate that gets applied to the ouptut DataArray
        is the `end_time` attribute of the Scene. This is the time
        the scan finished, not the time it started. I don't dare change
        it in order to stay consistent with all the historical data,
        but it is important to be aware of it.
    """
    # Convert the Scene to a DataArray, filtering to the desired data variables. Everything
    # up to the load stays lazy: the cast to the stored dtype and the flip of a descending
    # axis then run chunk by chunk, so the image is computed once, contiguous and already
    # in the dtype it is stored as, rather than at float32 and copied again on write.
    with warnings.catch_warnings(action="ignore"):
        ds: xr.Dataset = (
            scene.to_xarray_dataset()  # type: ignore
            .drop_vars(["acq_time", "crs"], errors="ignore")
            .astype(np.float32)
        )
        for name, dtype in (store_dtypes or {}).items():
            if name in ds.data_vars:
                ds[name] = ds[name].astype(dtype)
        ds = _sort_xy_coords(ds, dims=("x", "y"))
        with _compute_threads():
            ds = ds.load()

    # Skip timestamps where any requested channel is missing
    missing = [c.name for c in channels if c.name not in ds.data_vars]
    if missing:
        raise ValidationError(
            f"Missing channels {missing} from dataset. "
            f"Available variables: {list(ds.data_vars)}",
        )

    if "time_parameters" in ds.attrs:
        time = pd.Timestamp(ds.attrs["time_parameters"]["nominal_end_time"]).as_unit("ns")
        obs_start_time = pd.Timestamp(ds.attrs["time_parameters"]["observation_start_time"]).as_unit("ns")
        obs_end_time = pd.Timestamp(ds.attrs["time_parameters"]["observation_end_time"]).as_unit("ns")
    else:
        time = pd.Timestamp(ds.attrs["end_time"]).as_unit("ns")
        obs_start_time = pd.Timestamp(ds.attrs["start_time"]).as_unit("ns")
        obs_end_time = pd.Timestamp(ds.attrs["end_time"]).as_unit("ns")
    platform_name: str = ds.attrs["platform_name"]
    area_def: AreaDefinition = ds.attrs["area"]
    #cal_slope, cal_offset = _get_calib_coefficients(ds, channels)
    orbital_params = _get_orbital_params(ds)

    # Stack channels into a new dimension and compile the metadata
    ds = _stack_channels_to_dim(ds, channels)

    ds = ds.expand_dims({"time": [time]})

    ds = ds.assign(
        instrument=("time", np.array([platform_name]).astype("<U26")),
        #cal_slope=(["time",], [cal_slope]),
        #cal_offset=(["time",], [cal_offset]),
        observation_start_time=("time", [obs_start_time]),
        observation_end_time=("time", [obs_end_time]),
        x_geostationary_coordinates=(["time","x"], [ds["x"].values]),
        y_geostationary_coordinates=(["time","y"], [ds["y"].values]),
        **{k: ("time", [v]) for k, v in orbital_params.items()},  # type: ignore
    )

    # Increase clarity of coordinates, including coordinate dimension names and attributes
    ds = ds.rename({"x": "x_geostationary", "y": "y_geostationary"})
    for coord in ["x_geostationary", "y_geostationary"]:
        ds.coords[coord].attrs["coordinate_reference_system"] = "geostationary"

    # Make sure dimensions and coordinates are in expected order
    ds = ds.transpose("time", "y_geostationary", "x_geostationary")
    ds = _sort_xy_coords(ds)

    # Serialize attributes to be JSON-compatible, do for each subunit
    ds.attrs = _serialize_dict(ds.attrs)

    if crop_region_lonlat is not None:
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(proj="latlong", datum="WGS84"),
            pyproj.Proj(area_def.crs),
            always_xy=True,
        )

        left, bottom, right, top = transformer.transform_bounds(
            left=crop_region_lonlat[0],
            bottom=crop_region_lonlat[1],
            right=crop_region_lonlat[2],
            top=crop_region_lonlat[3],
        )

        ds = ds.sel(
            x_geostationary=slice(left, right),
            y_geostationary=slice(bottom, top),
        )

    return ds


def _serialize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursive helper function to serialize nested dicts."""
    sd: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dt.datetime):
            sd[key] = value.isoformat()
        elif isinstance(value, bool | np.bool_):
            sd[key] = str(value)
        elif isinstance(value, AreaDefinition):
            sd[key] = yaml.load(value.dump(), Loader=yaml.SafeLoader)  # type: ignore
        elif isinstance(value, dict):
            sd[key] = _serialize_dict(value)
        else:
            sd[key] = str(value)
    return sd


def _stack_channels_to_dim(ds: xr.Dataset, channels: list[models.SpectralChannel]) -> xr.Dataset:
    """Stack the channels into a new dimension and filter and compile metadata."""
    top_level_attrs = ["reader", "area", "georef_offset_corrected", "sensor"]
    attrs = {k: v for k, v in ds.attrs.items() if k in top_level_attrs}
    attrs["satellite_consumer_version"] = importlib.metadata.version("satellite_consumer")

    # For each channel, add their attributes to the top-level dataset attributes
    channel_attrs = ["units", "wavelength", "standard_name", "calibration", "resolution"]
    attrs["channels"] = {}
    for channel in channels:
        attrs["channels"][channel.name] = {
            k: v for k, v in ds[channel.name].attrs.items() if k in channel_attrs
        }

    #ds = (
    #    ds.to_dataarray(name="data", dim="channel")
    #    .to_dataset(promote_attrs=True)
    #    .sel(channel=[c.name for c in channels])
    #)

    # Replace the attrs with the compiled version
    #ds.data_vars["data"].attrs.clear()
    ds.attrs = attrs
    # Clear attrs from each of the data variables, since they are now in the top-level attrs
    for channel in channels:
        ds[channel.name].attrs.clear()

    return ds


def _get_calib_coefficients(
    ds: xr.Dataset,
    channels: list[models.SpectralChannel],
) -> tuple[list[float], list[float]]:
    """Pull the calibration slope and offset for each channel.

    These values effectively act as a y=mx+c to convert from raw counts to radiance.
    """
    calib_attrs = ds.attrs["raw_metadata"]["15_DATA_HEADER"]["RadiometricProcessing"][
        "Level15ImageCalibration"
    ]

    cal_slope = [calib_attrs["CalSlope"][c.satpy_index] for c in channels]
    cal_offset = [calib_attrs["CalOffset"][c.satpy_index] for c in channels]

    return cal_slope, cal_offset


def _get_orbital_params(ds: xr.Dataset) -> dict[str, float]:
    """Extract orbital parameters from the dataset attributes."""
    # These might be in the channel attributes, but we want them at the top level for easier access
    if "orbital_parameters" not in ds.attrs:
        # Go through data vars to get the orbital parameters from the first channel that has them
        for var in ds.data_vars:
            if "orbital_parameters" in ds[var].attrs:
                ds.attrs["orbital_parameters"] = ds[var].attrs["orbital_parameters"]
                break
    keys = [
        "satellite_actual_longitude",
        "satellite_actual_latitude",
        "satellite_actual_altitude",
        "projection_longitude",
        "projection_latitude",
        "projection_altitude",
    ]
    if "satellite_actual_longitude" not in ds.attrs["orbital_parameters"]:
        # Some satellites (i.e. GOES) only have nominal values, so we use those instead.
        keys = [
            "satellite_nominal_longitude",
            "satellite_nominal_latitude",
            "satellite_nominal_altitude",
            "projection_longitude",
            "projection_latitude",
            "projection_altitude",
        ]
    return {k.replace("nominal", "actual"): float(ds.attrs["orbital_parameters"][k]) for k in keys}


def _get_earthdisk_nan_frac(
    ds: xr.Dataset,
    area_def: AreaDefinition,
    chunksize: int = 500,
) -> float:
    """Calculate the fraction of NaN values on the earth-disk in the dataset."""
    # Use chunking to speed up the lon-lat generation
    chunks = [
        [
            min(chunksize, ds.sizes[dim] - i * chunksize)
            for i in range(int(np.ceil(ds.sizes[dim] / chunksize)))
        ]
        for dim in ["y", "x"]
    ]

    # This returns a lon-lat ndarray that is infinite off-earth-disk
    # Use this as a mask to check how many NaNs there are on-earth-disk
    lons, _ = area_def.get_lonlats(chunks=chunks)  # type: ignore
    on_earth_mask = np.isfinite(lons).compute()

    # Calculate the mean NaN fraction on-earth-disk for each channel
    # We do this in a loop to avoid slow xarray operations
    ds_nan = ds.isnull()
    channel_nan_fracs = [
        ds_nan.data_vars[var].values[on_earth_mask].mean() for var in ds_nan.data_vars
    ]
    return float(np.mean(channel_nan_fracs))


def _sort_xy_coords(
    ds: xr.Dataset,
    dims: tuple[str, str] = ("x_geostationary", "y_geostationary"),
) -> xr.Dataset:
    """Sort the Dataset spatial coordinates in ascending order."""
    for dim in dims:
        dim_diffs = np.diff(ds[dim].values)
        # If coord is in monotonic descending order, reverse it
        if (dim_diffs < 0).all():
            ds = ds.isel({dim: slice(None, None, -1)})

        # Else check that it is monotonic ascending
        else:
            if not (dim_diffs > 0).all():
                raise ValueError(f"{dim} coordinate is not monotonic.")

    return ds
