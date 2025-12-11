import tempfile
import unittest
from typing import TypedDict

import numpy as np
import xarray as xr

from satellite_consumer.storage import get_fs, write_to_zarr
from satellite_consumer.test_mocks import mocks3


class TestStorage(unittest.TestCase):
    """Test the storage functions."""

    def test_get_s3fs(self) -> None:
        """Test that the function returns a filesystem."""
        with mocks3() as dst:
            fs = get_fs(path=dst)
            self.assertIsNotNone(fs)
            self.assertTrue(fs.isdir(dst))

    def test_write_to_zarr(self) -> None:
        """Test that the function writes to an S3 bucket."""

        class TestCase(TypedDict):
            name: str
            dst: str

        with mocks3() as s3dir, tempfile.TemporaryDirectory(suffix="zarr") as tmpdir:
            tests: list[TestCase] = [
                {"name": "test_local", "dst": tmpdir + "/test.zarr"},
                {"name": "test_s3", "dst": s3dir + "test.zarr"},
            ]

            da: xr.DataArray = xr.DataArray(
                name="data",
                coords={
                    "time": [np.datetime64("2021-01-01T00:00", "ns")],
                    "y_geostationary": np.linspace(-6980250.0, 6980250.0, 1392),
                    "x_geostationary": np.linspace(-18500000.0, 18500000.0, 3712),
                    "channel": ["VIS", "IR"],
                },
                data=np.ones(shape=(1, 1392, 3712, 2)),
            )
            ds: xr.Dataset = da.to_dataset()

            for test in tests:
                with self.subTest(name=test["name"]):
                    write_to_zarr(ds=ds, dst=test["dst"])
                    store_da = xr.open_dataarray(test["dst"], engine="zarr", consolidated=False)
                    self.assertTrue((store_da.isel(time=0).values == 1.0).all())
