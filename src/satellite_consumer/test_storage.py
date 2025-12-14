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

            ds: xr.Dataset = xr.Dataset(
                coords={
                    "time": [np.datetime64("2021-01-01T00:00", "ns")],
                    "y_geostationary": np.linspace(-6980250.0, 6980250.0, 1392),
                    "x_geostationary": np.linspace(-18500000.0, 18500000.0, 3712),
                    "channel": ["VIS", "IR"],
                },
                data_vars={
                    "data": (
                        ["time", "y_geostationary", "x_geostationary", "channel"],
                        np.ones(shape=(1, 1392, 3712, 2)),
                    ),
                    "instrument": (["time"], ["FAKE"]),
                    "satellite_actual_longitude": (["time"], [0.0]),
                    "satellite_actual_latitude": (["time"], [0.0]),
                    "satellite_actual_altitude": (["time"], [35786023.0]),
                },
            )

            for test in tests:
                with self.subTest(name=test["name"]):
                    write_to_zarr(
                        ds=ds,
                        dst=test["dst"],
                        append_dim="time",
                        dims=["time", "y_geostationary", "x_geostationary", "channel"],
                        chunks=[1, 696, 1856, 2],
                        shards=[1, 1392, 3712, 2],
                    )
                    store_ds = xr.open_zarr(test["dst"], consolidated=False)
                    self.assertTrue((store_ds.data_vars["data"].isel(time=0).values == 1.0).all())
