"""Functions for processing satellite data."""

import datetime as dt
import warnings
from typing import Any

import hdf5plugin  # noqa: F401
import importlib_metadata
import numpy as np
import pandas as pd
import pyresample.geometry
import xarray as xr
import yaml
from satpy.scene import Scene

from satellite_consumer import models


def process_raw(
    paths: list[str],
    channels: list[models.SpectralChannel],
    resolution_meters: int,
    crop_region_geos: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Process a set of raw files into an xarray DataArray.

    Args:
        paths: List of file paths to raw satellite data files.
        channels: List of channels to extract.
        resolution_meters: Desired spatial resolution in meters.
        crop_region_geos: Optional tuple defining the geostationary coordinate
            region to crop to, in the form (x_min, y_min, x_max, y_max).
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
        scene.load( # type: ignore
            wishlist=[c.name for c in channels],
            # The resolution arg has to be "*" for 3000m, for some reason
            resolution=resolution_meters if resolution_meters < 3000 else "*",
            calibration=[c.representation for c in channels],
        )

    except Exception as e:
        raise OSError(f"Error reading paths as satpy Scene: {e}") from e

    try:
        ds: xr.Dataset = _map_scene_to_dataset(
            scene=scene,
            channels=channels,
            crop_region_geos=crop_region_geos,
        )
    except Exception as e:
        raise ValueError(f"Error converting paths to Dataset: {e}") from e

    return ds


def _map_scene_to_dataset(
    scene: Scene,
    channels: list[models.SpectralChannel],
    crop_region_geos: tuple[float, float, float, float] | None = None,
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
    for channel in scene.wishlist:
        scene[channel] = scene[channel].drop_vars("acq_time", errors="ignore")

    with warnings.catch_warnings(action="ignore"):
        ds: xr.Dataset = (
            scene.to_xarray_dataset() # type: ignore
            .drop_vars("acq_time", errors="ignore")[[c.name for c in channels]]
            .to_dataarray(name="data", dim="channel")
            .to_dataset(promote_attrs=False)
        )

    # Satpy returns a latlon ndarray that is infinite off-earth-disk
    # Use this as a mask to check there are not too many NaNs on-earth-disk
    # * RSS has 12.5% on-disk NaNs for their L1.5 data, so we allow up to 13%
    lons, _ = ds.data_vars["data"].attrs["area"].get_lonlats()
    nan_frac = ds.data_vars["data"].isnull().where(np.isfinite(lons)).mean().values
    if nan_frac > 0.13:
        raise ValueError(f"Too many NaN values on-disk in the data array: {nan_frac}")

    # Ensure DataArray has a time dimension
    rounded_time = pd.Timestamp(ds.data_vars["data"].attrs["time_parameters"]["nominal_end_time"])
    if "time" not in ds.dims:
        time = pd.to_datetime(rounded_time)
        ds = ds.expand_dims({"time": [time]})

    # Add the satellite coordinates, and satellite name as non-dimensional coords
    # These are pulled from the first channel, as they are identical across the board
    cal_slope, cal_offset = _get_calib_coefficients(scene, channels)
    first_satpy_channel = scene[next(iter(scene.wishlist))]
    ds = ds.assign(
        instrument=("time", [first_satpy_channel.attrs["platform_name"]]),
        cal_slope=(["time", "channel"], [cal_slope]),
        cal_offset=(["time", "channel"], [cal_offset]),
        **{ # type: ignore
            k: ("time", [float(first_satpy_channel.attrs["orbital_parameters"][k])])
            for k in [
                "satellite_actual_longitude",
                "satellite_actual_latitude",
                "satellite_actual_altitude",
            ]
        },
    )

    # Increase clarity of coordinates, including coordinate dimension names and attributes
    ds = ds.rename({"x": "x_geostationary", "y": "y_geostationary"})
    for name in ["x_geostationary", "y_geostationary"]:
        ds.coords[name].attrs["coordinate_reference_system"] = "geostationary"

    ds = (
        ds.transpose("time", "y_geostationary", "x_geostationary", "channel")
        .sortby(["y_geostationary", "x_geostationary"])
        .sel(channel=[c.name for c in channels])
    )
    ds["data"] = ds["data"].astype(np.float32)

    if crop_region_geos is not None:
        ds = ds.sel(
            x_geostationary=slice(crop_region_geos[0], crop_region_geos[2]),
            y_geostationary=slice(crop_region_geos[1], crop_region_geos[3]),
        )

    del ds["crs"]
    ds = ds.load()

    ds.data_vars["data"].attrs = _filter_attrs(scene=scene)

    return ds


def _filter_attrs(scene: Scene) -> dict[str, Any]:
    """Filter the attributes returned by satpy down to a subset of interest."""
    attrs: dict[str, Any] = {
        k: v
        for k, v in scene[next(iter(scene.wishlist))].attrs.items()
        if k in ["reader", "area", "georef_offset_corrected", "sensor"]
    }
    attrs["satellite_consumer_version"] = importlib_metadata.version("satellite_consumer")

    # For each channel, add their attributes to the top-level dataset attributes
    attrs["channels"] = {}
    for channel in scene.wishlist:
        attrs["channels"][channel["name"]] = {}
        for attr in [
            ca
            for ca in scene[channel].attrs
            if ca in ["units", "wavelength", "standard_name", "calibration", "resolution"]
        ]:
            attrs["channels"][channel["name"]][attr] = scene[channel].attrs[attr]

    def _serialize(d: dict[str, Any]) -> dict[str, Any]:
        """Recursive helper function to serialize nested dicts."""
        sd: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, dt.datetime):
                sd[key] = value.isoformat()
            elif isinstance(value, bool | np.bool_):
                sd[key] = str(value)
            elif isinstance(value, pyresample.geometry.AreaDefinition):
                sd[key] = yaml.load(value.dump(), Loader=yaml.SafeLoader) # type: ignore
            elif isinstance(value, dict):
                sd[key] = _serialize(value)
            else:
                sd[key] = str(value)
        return sd

    return _serialize(attrs)


def _get_calib_coefficients(
    scene: Scene,
    channels: list[models.SpectralChannel],
) -> tuple[list[float], list[float]]:
    """Pull the calibration slope and offset for each channel.

    These values effectively act as a y=mx+c to convert from raw counts to radiance.
    """
    calib_attrs = scene[channels[0].name].attrs["raw_metadata"]["15_DATA_HEADER"][
        "RadiometricProcessing"
    ]["Level15ImageCalibration"]

    cal_slope = [calib_attrs["CalSlope"][c.satpy_index] for c in channels]
    cal_offset = [calib_attrs["CalOffset"][c.satpy_index] for c in channels]

    return cal_slope, cal_offset

