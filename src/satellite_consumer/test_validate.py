import tempfile
import unittest
from typing import TypedDict

import numpy as np

from satellite_consumer.config import Coordinates
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.storage import create_empty_zarr
from satellite_consumer.test_mocks import mocks3
from satellite_consumer.validate import validate


class TestValidate(unittest.TestCase):
    """Test the validate functions."""

    def test_validate_fails_on_empty_zarrs(self) -> None:
        """Test that the function creates an empty zarr store."""

        class TestContainer(TypedDict):
            name: str
            dst: str

        with mocks3() as s3dir, tempfile.TemporaryDirectory(suffix="zarr") as tmpdir:
            coords = Coordinates(
                time=[np.datetime64(f"2021-01-01T0{h}:00", "ns") for h in range(0, 3)],
                y_geostationary=list(range(1392)),
                x_geostationary=list(range(3712)),
                variable=["VIS006", "IR_016"],
            )

            tests: list[TestContainer] = [
                {"name": "test_local", "dst": tmpdir + "/test.zarr"},
                {"name": "test_s3", "dst": s3dir + "test.zarr"},
            ]

            for test in tests:
                with self.subTest(name=test["name"]):
                    store_da = create_empty_zarr(dst=test["dst"], coords=coords)
                    self.assertTrue(
                        np.isnan(store_da.values).all(),
                        msg="All values in empty store should be NaN",
                    )

                    with self.assertRaises(
                        ValidationError,
                        msg="Empty store should fail validation",
                    ):
                        validate(
                            src=test["dst"],
                            check_region_xy_slices=(slice(None, None), slice(None, None)),
                        )
