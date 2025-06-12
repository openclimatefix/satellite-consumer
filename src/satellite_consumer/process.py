"""Functions for processing satellite data."""

import datetime as dt
from typing import Any

import hdf5plugin  # type: ignore # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import satpy
import xarray as xr
import yaml
from loguru import logger as log

from satellite_consumer.config import SpectralChannelMetadata

OSGB, WGS84 = (27700, 4326)
transformer: pyproj.Transformer = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB)
"""Transformer for converting between WGS84 and OSGB coordinates."""

REGION_MAP: dict[str, tuple[int, int, int, int]] = {
    "UK": (-17, 44, 11, 73), "India": (60, 6, 97, 37),
}
"""Geographic bounds for various regions of interest that can be cropped to.

The bounds are in order of min_lon, min_lat, max_lon, max_lat.
See (see https://satpy.readthedocs.io/en/stable/_modules/satpy/scene.html)
"""


def process_raw(
    paths: list[str],
    channels: list[SpectralChannelMetadata],
    resolution_meters: int,
    normalize: bool = True,
) -> xr.DataArray:
    """Process a set of raw files into an xarray dataset.

    Args:
        paths: List of paths to the raw files to open. Can be a local paths or
            S3 urls e.g. `s3://bucket-name/path/to/file`.
        channels: List of channel names to load from the raw files.
        resolution_meters: The resolution in meters to load the data at.
        normalize: Whether to normalize the data to the unit interval [0, 1].
    """
    log.debug(
        "Reading raw files as a satpy Scene",
        resolution=resolution_meters, num_files=len(paths),
    )
    try:
        loader: str = "seviri_l1b_native" if paths[0].endswith(".nat") else "fci_l1c_nc"
        scene: satpy.Scene = satpy.Scene(filenames={loader: paths}) # type:ignore
        scene.load(
            [c.name for c in channels if resolution_meters in c.resolution_meters],
            resolution=resolution_meters if resolution_meters < 3000 else "*",
        )
    except Exception as e:
        raise OSError(f"Error reading paths as satpy Scene: {e}") from e

    try:
        log.debug("Converting Scene to dataarray", normalize=normalize)
        da: xr.DataArray = _map_scene_to_dataarray(
            scene=scene,
            crop_region=None,
            calculate_osgb=False,
        )
    except Exception as e:
        raise ValueError(f"Error converting paths to DataArray: {e}") from e

    if normalize:
        # Rescale the data, save as dataarray
        try:
            da = _normalize(da=da)
        except Exception as e:
            raise ValueError(f"Error rescaling dataarray: {e}") from e

    # Reorder the coordinates, and set the data type
    del da["crs"]
    da = da.transpose("time", "y_geostationary", "x_geostationary", "variable")
    da = da.astype(np.float32)

    return da


def _map_scene_to_dataarray(
    scene: satpy.Scene, # type:ignore # Don't know why it dislikes this
    crop_region: str | None,
    calculate_osgb: bool = True,
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
        crop_region: The region to crop the data to.
        calculate_osgb: Whether to calculate OSGB coordinates.
    """
    if crop_region is not None:
        if crop_region not in REGION_MAP:
            raise ValueError(
                f"Unknown crop_region '{crop_region}'. Expected one of of '{REGION_MAP.keys()}'.",
            )
        crop_bounds = REGION_MAP[crop_region]
        log.debug(f"Cropping scene to region '{crop_region}'", crop_bounds=crop_bounds)
        try:
            scene = scene.crop(ll_bbox=crop_bounds)
        except NotImplementedError:
            # 15 minutely data by default doesn't work for some reason, have to resample it
            # * The resampling is different for the HRV band
            # TODO: Test this works
            scene = scene.resample(
                destination="msg_seviri_rss_1km" \
                    if scene.wishlist == set("HRV") \
                    else "msg_seviri_rss_3km",
            ).crop(ll_bbox=crop_bounds)

    for channel in scene.wishlist:
        scene[channel] = scene[channel].drop_vars("acq_time", errors="ignore")
        # Write individual channel attributes to the top level scene attributes
        # * This prevents information loss when converting to a dataarray
        # * Ignores attributes on the channel that are not useful or not serializeable
        for attr in [ca for ca in scene[channel].attrs if ca not in ["area", "_satpy_id"]]:
            scene.attrs[f"{channel['name']}_{attr}"] = scene[channel].attrs[attr].__repr__()

    da: xr.DataArray = scene.to_xarray_dataset().to_array().rename("data")
    da.attrs = da.attrs | scene.attrs

    # Ensure DataArray has a time dimension
    da.attrs["end_time"] = pd.Timestamp(da.attrs["end_time"]).round("5 min").__str__()
    if "time" not in da.dims:
        time = pd.to_datetime(pd.Timestamp(da.attrs["end_time"]).round("5 min"))
        da = da.assign_coords({"time": time}).expand_dims("time")

    # Increase clarity of coordinates, including coordinate dimension names and attributes
    da = da.rename({"x": "x_geostationary", "y": "y_geostationary"})
    for name in ["x_geostationary", "y_geostationary"]:
        da.coords[name].attrs["coordinate_reference_system"] = "geostationary"

    if calculate_osgb:
        log.debug("Calculating OSGB coordinates")
        lon, lat = scene[scene.wishlist[0]].attrs["area"].get_lonlats()
        osgb_x, osgb_y = transformer.transform(lat, lon)
        da = da.assign_coords(coords={
            "x_osgb": (("y_geostationary", "x_geostationary"), np.float32(osgb_x)),
            "y_osgb": (("y_geostationary", "x_geostationary"), np.float32(osgb_y)),
        })
        da.coords["x_osgb"].attrs = {
            "units": "meter", "coordinate_reference_system": "OSGB", "name": "Easting",
        }
        da.coords["y_osgb"].attrs = {
            "units": "meter", "coordinate_reference_system": "OSGB", "name": "Northing",
        }

    da = da.transpose("time", "y_geostationary", "x_geostationary", "variable").chunk(
        chunks={"time": 1, "y_geostationary": -1, "x_geostationary": -1, "variable": 1},
    ).sortby(["variable", "y_geostationary"]).sortby("x_geostationary", ascending=False)

    da.attrs = _serialize_attrs(da.attrs)
    log.debug("Attributes", **da.attrs)

    return da


def _normalize(da: xr.DataArray) -> xr.DataArray:
    """Normalize DataArray values into the unit interval [0, 1].

    Normalization is carried out based on an approximation of the minimum and maximum
    values of each spectral channel. These values were calculated from a subset of
    image data and are not exact.

    NaNs in the original DataArray are preserved in the normalized DataArray.
    """
    known_variables = {c.name for c in SEVIRI_CHANNELS}
    incoming_variables = set(da.coords["variable"].values.tolist())
    if not incoming_variables.issubset(known_variables):
        raise ValueError(
                "Cannot rescale DataArray as some variables present are not recognized: "
                f"'{incoming_variables.difference(known_variables)}'",
        )

    # For each channel, subtract the minimum and divide by the range
    for variable in da.coords["variable"]:
        channel_metadata = next(filter(lambda c: c.name == variable, SEVIRI_CHANNELS))
        da.loc[{"variable": variable}] -= channel_metadata.minimum
        da.loc[{"variable": variable}] /= channel_metadata.range
    # da -= [c.minimum for c in channels]
    # da /= [c.maximum - c.minimum for c in channels]

    # Since the mins and maxes are approximations, clip the values to [0, 1]
    da = da.clip(min=0, max=1).astype(np.float32)

    return da

def _serialize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Ensure each value of dict can be serialized.

    This is required before saving to Zarr because Zarr represents attrs values in a
    JSON file (.zmetadata).

    The `area` field (which is a `pyresample.geometry.AreaDefinition` object gets turned
    into a YAML string, which can be loaded again using
    `area_definition = pyresample.area_config.load_area_from_string(data_array.attrs['area'])`

    Returns attrs dict where every value has been made serializable.
    """
    for key, value in attrs.items():
        # Convert Dicts
        if isinstance(value, dict):
            # Convert np.float32 to Python floats (otherwise yaml.dump complains)
            for inner_key in value:
                inner_value = value[inner_key]
                if isinstance(inner_value, np.floating):
                    value[inner_key] = float(inner_value)
            attrs[key] = yaml.dump(value)

        # Convert Numpy bools
        if isinstance(value, bool | np.bool_):
            attrs[key] = str(value)
        elif isinstance(value, dt.datetime):
            attrs[key] = value.isoformat()
        # Convert other dumpable things
        else:
            try:
                attrs[key] = value.dump() # type: ignore
            except AttributeError:
                # If the value is not dumpable, just convert it to a string
                attrs[key] = str(value)

    return attrs

