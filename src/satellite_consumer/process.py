"""Functions for processing satellite data."""

import datetime as dt
import importlib.metadata
import logging
import warnings
from typing import Any

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
from satellite_consumer.exceptions import ValidationError

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


def process_raw(
    paths: list[str],
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_lonlat: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Process a set of raw files into an xarray DataArray.

    Args:
        paths: List of file paths to raw satellite data files.
        channels: List of channels to extract.
        resolution_meters: Desired spatial resolution in meters.
        crop_region_lonlat: Optional tuple defining the lon-lat coordinate
            region to crop to, in the form (lon_min, lat_min, lon_max, lat_max).
    """
    try:
        # Meteosat 3rd gen don't output .nat files, and so requires a different loader
        loader: str = "fci_l1c_nc"
        reader_kwargs: dict[str, Any] = {}
        if paths[0].endswith(".nat"):
            loader = "seviri_l1b_native"
            # Nominal calibration represents raw integer counts to radiance via slope and intercept
            # Include flag surfaces calibration values
            reader_kwargs = {
                "calib_mode": "nominal",
                "include_raw_metadata": True,
            }

        scene: Scene = Scene(
            filenames={loader: paths},  # type:ignore
            reader_kwargs=reader_kwargs,
        )
        scene.load(  # type: ignore
            wishlist=[c.name for c in channels],
            # The resolution arg has to be "*" for 3000m, for some reason
            resolution=resolution_meters if resolution_meters < 3000 else "*",
            calibration=[c.representation for c in channels],
        )

    except Exception as e:
        raise OSError(f"Error reading paths as satpy Scene: {e}") from e

    ds: xr.Dataset = _map_scene_to_dataset(
        scene=scene,
        channels=channels,
        crop_region_lonlat=crop_region_lonlat,
    )

    return ds


def _map_scene_to_dataset(
    scene: Scene,
    channels: list[models.SpectralChannel],
    crop_region_lonlat: tuple[float, float, float, float] | None = None,
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
    # Convert the Scene to a DataArray, filtering to the desired data variables
    with warnings.catch_warnings(action="ignore"):
        ds: xr.Dataset = (
            scene.to_xarray_dataset()  # type: ignore
            .drop_vars(["acq_time", "crs"], errors="ignore")
            .astype(np.float32)
            .load()
        )

    # Extract values from attributes before we overwrite them
    time = pd.Timestamp(ds.attrs["time_parameters"]["nominal_end_time"]).as_unit("ns")
    platform_name: str = ds.attrs["platform_name"]
    area_def: AreaDefinition = ds.attrs["area"]
    cal_slope, cal_offset = _get_calib_coefficients(ds, channels)
    orbital_params = _get_orbital_params(ds)

    # RSS has 12.5% on-disk NaNs for their L1.5 data, so we allow up to 13.5%
    nan_frac = _get_earthdisk_nan_frac(ds, area_def)
    if nan_frac > 0.2:
        raise ValidationError(f"Too many NaN values on earth-disk in the data array: {nan_frac}")

    # Stack channels into a new dimension and compile the metadata
    ds = _stack_channels_to_dim(ds, channels)

    ds = ds.expand_dims({"time": [time]})

    ds = ds.assign(
        instrument=("time", np.array([platform_name]).astype("<U26")),
        cal_slope=(["time", "channel"], [cal_slope]),
        cal_offset=(["time", "channel"], [cal_offset]),
        **{k: ("time", [v]) for k, v in orbital_params.items()},  # type: ignore
    )

    # Increase clarity of coordinates, including coordinate dimension names and attributes
    ds = ds.rename({"x": "x_geostationary", "y": "y_geostationary"})
    for coord in ["x_geostationary", "y_geostationary"]:
        ds.coords[coord].attrs["coordinate_reference_system"] = "geostationary"

    # Make sure dimensions and coordinates are in expected order
    ds = ds.transpose("time", "y_geostationary", "x_geostationary", "channel")
    ds = _sort_xy_coords(ds)

    # Serialize attributes to be JSON-compatible
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

        log.info(str(ds.data_vars["data"].attrs))

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

    ds = (
        ds.to_dataarray(name="data", dim="channel")
        .to_dataset(promote_attrs=True)
        .sel(channel=[c.name for c in channels])
    )

    # Replace the attrs with the compiled version
    ds.data_vars["data"].attrs.clear()
    ds.attrs = attrs

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
    keys = [
        "satellite_actual_longitude",
        "satellite_actual_latitude",
        "satellite_actual_altitude",
        "projection_longitude",
        "projection_latitude",
        "projection_altitude",
    ]
    return {k: float(ds.attrs["orbital_parameters"][k]) for k in keys}


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


def _sort_xy_coords(ds: xr.Dataset) -> xr.Dataset:
    """Sort the Dataset spatial coordinates in ascending order."""
    for dim in ["x_geostationary", "y_geostationary"]:
        dim_diffs = np.diff(ds[dim].values)
        # If coord is in monotonic descending order, reverse it
        if (dim_diffs < 0).all():
            ds = ds.isel({dim: slice(None, None, -1)})

        # Else check that it is monotonic ascending
        else:
            if not (dim_diffs > 0).all():
                raise ValueError(f"{dim} coordinate is not monotonic.")

    return ds
