"""Functions for processing satellite data."""

import datetime as dt
from typing import Any

import hdf5plugin  # type: ignore # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import pyresample.geometry
import satpy
import xarray as xr
import yaml
from loguru import logger as log

from satellite_consumer.config import SpectralChannelMetadata

OSGB, WGS84 = (27700, 4326)
transformer: pyproj.Transformer = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)
"""Transformer for converting between WGS84 and OSGB coordinates."""


def process_raw(
    paths: list[str],
    channels: list[SpectralChannelMetadata],
    resolution_meters: int,
    normalize: bool = True,
    crop_region_geos: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Process a set of raw files into an xarray DataArray.

    Args:
        paths: List of paths to the raw files to open. Can be a local paths or
            S3 urls e.g. `s3://bucket-name/path/to/file`.
        channels: List of channel names to load from the raw files.
        resolution_meters: The resolution in meters to load the data at.
        normalize: Whether to normalize the data to the unit interval [0, 1].
        crop_region_geos: Optional bounds to crop the data to, in the format
            (west, south, east, north) in geostationary coordinates.
            If None, no cropping is applied.
    """
    log.debug(
        "Reading raw files as a satpy Scene",
        resolution=resolution_meters,
        num_files=len(paths),
    )
    try:
        loader: str = "seviri_l1b_native" if paths[0].endswith(".nat") else "fci_l1c_nc"
        scene: satpy.Scene = satpy.Scene(filenames={loader: paths})  # type:ignore
        cnames: list[str] = [c.name for c in channels if resolution_meters in c.resolution_meters]
        scene.load(
            cnames,
            resolution=resolution_meters if resolution_meters < 3000 else "*",
        )
    except Exception as e:
        raise OSError(f"Error reading paths as satpy Scene: {e}") from e

    try:
        log.debug("Converting Scene to dataarray", normalize=normalize)
        da: xr.DataArray = _map_scene_to_dataarray(
            scene=scene,
            calculate_osgb=False,
            crop_region_geos=crop_region_geos,
        )
    except Exception as e:
        raise ValueError(f"Error converting paths to DataArray: {e}") from e

    if normalize:
        # Rescale the data, save as dataarray
        try:
            da = _normalize(
                da=da,
                channels=[c for c in channels if resolution_meters in c.resolution_meters],
            )
        except Exception as e:
            raise ValueError(f"Error rescaling dataarray: {e}") from e

    # Reorder the coordinates, and set the data type
    del da["crs"]
    da = da.transpose("time", "y_geostationary", "x_geostationary", "variable")
    da = da.astype(np.float32)

    return da


def _map_scene_to_dataarray(
    scene: satpy.Scene,  # type:ignore # Don't know why it dislikes this
    calculate_osgb: bool = True,
    crop_region_geos: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Converts a Scene with satellite data into a data array.

    Note!!!:
        This function fudges the timestamps somewhat!
        The time coordinate that gets applied to the ouptut DataArray
        is the `end_time` attribute of the Scene. This is the time
        the scan finished, not the time it started. I don't dare change
        it in order to stay consistent with all the historical data,
        but it is important to be aware of it.

    Args:
        scene: The satpy.Scene containing the satellite data.
        calculate_osgb: Whether to calculate OSGB coordinates.
        crop_region_geos: Optional bounds to crop the data to, in the format
    """
    for channel in scene.wishlist:
        # Drop unwanted variables
        scene[channel] = scene[channel].drop_vars("acq_time", errors="ignore")

    # Convert the Scene to a DataArray
    da: xr.DataArray = scene.to_xarray_dataset().to_array(name="data")
    if crop_region_geos is not None:
        da = (
            da.where(da.coords["x"] >= crop_region_geos[0], drop=True)
            .where(da.coords["x"] <= crop_region_geos[2], drop=True)
            .where(da.coords["y"] >= crop_region_geos[1], drop=True)
            .where(da.coords["y"] <= crop_region_geos[3], drop=True)
        )

    # Handle attrs, converting to serializable format
    da.attrs["variables"] = {}
    for channel in scene.wishlist:
        # Add channel metadata dataarray variables attributes
        da.attrs["variables"][channel["name"]] = {}
        for attr in [
            ca
            for ca in scene[channel].attrs
            if ca not in ["area", "_satpy_id", "orbital_parameters", "time_parameters"]
        ]:
            da.attrs["variables"][channel["name"]][attr] = scene[channel].attrs[attr]

    def _serialize(d: dict[str, Any]) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, dt.datetime):
                sd[key] = value.isoformat()
            elif isinstance(value, bool | np.bool_):
                sd[key] = str(value)
            elif isinstance(value, pyresample.geometry.AreaDefinition):
                sd[key] = yaml.load(value.dump(), Loader=yaml.SafeLoader)  # type:ignore
            elif isinstance(value, dict):
                sd[key] = _serialize(value)
            else:
                sd[key] = str(value)
        return sd

    da.attrs = _serialize(da.attrs)

    # Ensure DataArray has a time dimension
    rounded_time = pd.Timestamp(da.attrs["time_parameters"]["nominal_end_time"]).round("5 min")
    da.attrs["end_time"] = rounded_time.__str__()
    if "time" not in da.dims:
        time = pd.to_datetime(rounded_time)
        da = da.assign_coords({"time": time}).expand_dims("time")

    # Increase clarity of coordinates, including coordinate dimension names and attributes
    da = da.rename({"x": "x_geostationary", "y": "y_geostationary"})
    for name in ["x_geostationary", "y_geostationary"]:
        da.coords[name].attrs["coordinate_reference_system"] = "geostationary"

    if calculate_osgb:
        log.debug("Calculating OSGB coordinates")
        lon, lat = scene[scene.wishlist[0]].attrs["area"].get_lonlats()
        osgb_x, osgb_y = transformer.transform(lat, lon)
        da = da.assign_coords(
            coords={
                "x_osgb": (("y_geostationary", "x_geostationary"), np.float32(osgb_x)),
                "y_osgb": (("y_geostationary", "x_geostationary"), np.float32(osgb_y)),
            },
        )
        da.coords["x_osgb"].attrs = {
            "units": "meter",
            "coordinate_reference_system": "OSGB",
            "name": "Easting",
        }
        da.coords["y_osgb"].attrs = {
            "units": "meter",
            "coordinate_reference_system": "OSGB",
            "name": "Northing",
        }

    da = (
        da.transpose("time", "y_geostationary", "x_geostationary", "variable")
        .chunk(
            chunks={"time": 1, "y_geostationary": -1, "x_geostationary": -1, "variable": 1},
        )
        .sortby(["variable", "y_geostationary"])
        .sortby("x_geostationary", ascending=False)
    )

    return da


def _normalize(da: xr.DataArray, channels: list[SpectralChannelMetadata]) -> xr.DataArray:
    """Normalize DataArray values into the unit interval [0, 1].

    Normalization is carried out based on an approximation of the minimum and maximum
    values of each spectral channel. These values were calculated from a subset of
    image data and are not exact.

    NaNs in the original DataArray are preserved in the normalized DataArray.
    """
    known_variables = {c.name for c in channels}
    incoming_variables = set(da.coords["variable"].values.tolist())
    if not incoming_variables.issubset(known_variables):
        raise ValueError(
            "Cannot rescale DataArray as some variables present are not recognized: "
            f"'{incoming_variables.difference(known_variables)}'",
        )

    # For each channel, subtract the minimum and divide by the range
    for variable in da.coords["variable"]:
        channel_metadata = next(filter(lambda c: c.name == variable, channels))
        da.loc[{"variable": variable}] -= channel_metadata.minimum
        da.loc[{"variable": variable}] /= channel_metadata.range
    # da -= [c.minimum for c in channels]
    # da /= [c.maximum - c.minimum for c in channels]

    # Since the mins and maxes are approximations, clip the values to [0, 1]
    da = da.clip(min=0, max=1).astype(np.float32)

    return da
