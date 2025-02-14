"""Functions for validating the quality of the data in a dataset."""

import numpy as np
import xarray as xr


def is_valid(
        ds: xr.Dataset,
        nans_in_check_region_threshold: float = 0.05,
        images_failing_nan_check_threshold: float = 0,
        xy_slices: tuple[slice, slice] = (
            slice(-480_064.6, -996_133.85), slice(4_512_606.3, 5_058_679.8),
        ),
) -> bool:
    """Check the quality of the data in the given dataset.

    Looks for the number of NaNs in the data over important regions.

    Args:
        ds: Dataset to validate.
        nans_in_check_region_threshold: Percentage of NaNs in the check region
            above which to consider an image as having too many NaNs.
        images_failing_nan_check_threshold: Percentage of images with too many NaNs
            above which to consider the dataset as failing validation.
        xy_slices: Slices to use for the check region.
    """

    def _calc_null_percentage(data: np.ndarray) -> float:
        nulls = np.isnan(data)
        return float(nulls.sum() / len(nulls))

    result = xr.apply_ufunc(
        _calc_null_percentage,
        ds.data_vars["data"].sel(
            x_geostationary=xy_slices[0],
            y_geostationary=xy_slices[1],
        ),
        input_core_dims=[["x_geostationary", "y_geostationary"]],
        vectorize=True,
        dask="parallelized",
    )

    failed_image_count: int = (result > nans_in_check_region_threshold).sum().values
    total_image_count: int = result.size
    failed_image_percentage: float = failed_image_count / total_image_count
    log.info(
        f"{failed_image_count}/{total_image_count} "
        f"({failed_image_percentage:.2%}) of images have greater than 5% null values",
    )
    return failed_image_percentage <= images_failing_nan_check_threshold

