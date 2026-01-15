import tempfile
import unittest
from typing import Any, TypedDict

import numpy as np
import xarray as xr
import zarr.codecs

from satellite_consumer.storage import _sanitize_encoding, get_fs, write_to_store
from satellite_consumer.test_mocks import mocks3

encoding: dict[str, Any] = {
    "_ARRAY_DIMENSIONS": ["time", "y_geostationary", "x_geostationary", "channel"],
    "data": {
        "chunks": [1, 696, 1856, 2],
        "shards": [1, 1392, 3712, -1],
        "compressors": "yes",
    },
}

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
        "cal_slope": (["time", "channel"], [[1.0, 1.0]]),
        "cal_offset": (["time", "channel"], [[1.0, 1.0]]),
        "satellite_actual_longitude": (["time"], [0.0]),
        "satellite_actual_latitude": (["time"], [0.0]),
        "satellite_actual_altitude": (["time"], [35786023.0]),
        "projection_longitude": (["time"], [0.0]),
        "projection_latitude": (["time"], [0.0]),
        "projection_altitude": (["time"], [35786023.0]),
    },
)


class TestStorage(unittest.TestCase):
    """Test the storage functions."""

    def test_sanitize_encoding(self) -> None:
        """Test that the function sanitizes the encoding dictionary."""
        sanitized_encoding = _sanitize_encoding(
            ds=ds,
            dims=encoding["_ARRAY_DIMENSIONS"],
            data=encoding,
        )
        self.assertEqual(sanitized_encoding["data"]["chunks"], [1, 696, 1856, 2])
        self.assertEqual(sanitized_encoding["data"]["shards"], [1, 1392, 3712, 2])
        self.assertIsInstance(sanitized_encoding["data"]["compressors"], zarr.codecs.BloscCodec)
        self.assertIsNone(sanitized_encoding.get("_ARRAY_DIMENSIONS"))

    def test_get_s3fs(self) -> None:
        """Test that the function returns a filesystem."""
        with mocks3() as dst:
            fs = get_fs(path=dst)
            self.assertIsNotNone(fs)
            self.assertTrue(fs.isdir(dst))

    def test_write_to_store(self) -> None:
        """Test that the function writes to a zarr store."""

        class TestCase(TypedDict):
            name: str
            dst: str

        with mocks3() as s3dir, tempfile.TemporaryDirectory(suffix="zarr") as tmpdir:
            tests: list[TestCase] = [
                {"name": "test_local", "dst": tmpdir + "/test.zarr"},
                {"name": "test_s3", "dst": s3dir + "test.zarr"},
            ]

            for test in tests:
                with self.subTest(name=test["name"]):
                    write_to_store(
                        ds=ds,
                        dst=test["dst"],
                        append_dim="time",
                        encoding=encoding,
                    )
                    store_ds = xr.open_zarr(test["dst"], consolidated=False)
                    self.assertTrue((store_ds.data_vars["data"].isel(time=0).values == 1.0).all())
