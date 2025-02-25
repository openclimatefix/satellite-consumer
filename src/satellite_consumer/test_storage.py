import contextlib
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from botocore.client import BaseClient as BotocoreClient
from botocore.session import Session
from moto.server import ThreadedMotoServer

from satellite_consumer.storage import create_empty_store, get_fs, write_to_zarr


@contextlib.contextmanager
def mocks3() -> BotocoreClient:
    server = ThreadedMotoServer()
    server.start()
    with patch.dict("os.environ", {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_SECURITY_TOKEN": "test",
        "AWS_SESSION_TOKEN": "test",
        "AWS_ENDPOINT_URL": "http://localhost:5000",
        "AWS_DEFAULT_REGION": "us-east-1",
    }, clear=True):
        s3_client: BotocoreClient = Session().create_client(
            service_name="s3", region_name="us-east-1",
        )
        s3_client.create_bucket(Bucket="test-bucket")
        try:
            yield s3_client
        finally:
            s3_client.close()
            server.stop()

class TestGetS3FS(unittest.TestCase):
    """Test function to get an S3 filesystem."""

    def test_get_s3fs(self) -> None:
        """Test that the function returns a filesystem."""
        with mocks3():
            fs = get_fs("s3://test-bucket/")
            self.assertIsNotNone(fs)
            self.assertTrue(fs.isdir("s3://test-bucket/"))

class TestWriteToZarr(unittest.TestCase):
    """Test writing to a Zarr store."""

    def test_writes_to_s3(self) -> None:
        """Test that the function writes to an S3 bucket."""

        da: xr.DataArray = xr.DataArray(
           coords={
               "time": [np.datetime64("2021-01-01", "ns")],
               "x_geostationary": [1, 2, 3, 4],
               "y_geostationary": [1, 2, 3, 4],
               "variable": ["VIS006", "IR_016"],
            },
           data=np.random.rand(1, 4, 4, 2),
        )

        with mocks3():
            _ = create_empty_store(
                dst="s3://test-bucket/test.zarr",
                coords={
                    "time": [np.datetime64(f"2021-01-01T0{h}", "ns") for h in range(0, 3)],
                    "x_geostationary": [1, 2, 3, 4],
                    "y_geostationary": [1, 2, 3, 4],
                    "variable": ["VIS006", "IR_016"],
                },
            )

            write_to_zarr(da, "s3://test-bucket/test.zarr")
            fs = get_fs("s3://test-bucket/")
            self.assertTrue(fs.isdir("s3://test-bucket/test.zarr/"))

