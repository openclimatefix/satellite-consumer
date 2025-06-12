"""Functions for validating the quality of the data in a dataset."""

import warnings

import numpy as np
import xarray as xr
from loguru import logger as log

from satellite_consumer.exceptions import ValidationError


def validate(
        src: str | xr.DataArray,
        nans_in_check_region_threshold: float = 0.05,
        images_failing_nan_check_threshold: float = 0,
        check_region_xy_slices: tuple[slice, slice] = (
            slice(-480_064.6, -996_133.85), slice(4_512_606.3, 5_058_679.8),
        ),
) -> tuple[int, int]:
    """Check the quality of the data in the given dataset.

    Looks for the number of NaNs in the data over important regions.

    Args:
        src: Path to the dataset to validate.
        nans_in_check_region_threshold: Percentage of NaNs in the check region
            above which to consider an image as having too many NaNs.
        images_failing_nan_check_threshold: Percentage of images with too many NaNs
            above which to consider the dataset as failing validation.
        check_region_xy_slices: Slices to use for the check region.

    Returns:
        A tuple containing the number of images that failed the NaN check and
        the total number of images checked.
    """

    def _calc_null_percentage(data: np.typing.NDArray[np.float32]) -> float:
        nulls = np.isnan(data)
        if 0 in data.shape:
            log.warning(
                "Validation region has 0 area, check input slices correspond"
                "to coordinate values in the dataset",
            )
            return 1.0
        return float(nulls.sum() / len(nulls))

    # TODO: Remove warnings catch when Zarr makes up its mind about codecs
    with warnings.catch_warnings(action="ignore"):
        da: xr.DataArray = src if isinstance(src, xr.DataArray) else xr.open_dataarray(
            src, engine="zarr", consolidated=False,
        )
    if not {"x_geostationary", "y_geostationary"}.issubset(set(da.dims)):
        raise ValidationError(
            "Cannot validate dataset at path {src}. "
            "Expected dimensions ['x_geostationary', 'y_geostationary'] not present. "
            "Got: {list(ds.data_vars['data'].dims)}",
        )

    result = xr.apply_ufunc(
        _calc_null_percentage,
        da.sel(
            y_geostationary=check_region_xy_slices[1],
            x_geostationary=check_region_xy_slices[0],
        ),
        input_core_dims=[["y_geostationary", "x_geostationary"]],
        vectorize=True,
        dask="parallelized",
    )

    failed_image_count: int = (result > nans_in_check_region_threshold).sum().values
    total_image_count: int = result.size
    failed_image_percentage: float = failed_image_count / total_image_count
    if failed_image_percentage > images_failing_nan_check_threshold:
        raise ValidationError(
            f"Dataset at path {src} failed validation. "
            f"{failed_image_percentage:.2%} of images have greater than 5% null values"
            f"({failed_image_count}/{total_image_count})",
        )
    log.info(
        f"{failed_image_count}/{total_image_count} "
        f"({failed_image_percentage:.2%}) of images have greater than 5% null values",
    )
    return failed_image_count, total_image_count

