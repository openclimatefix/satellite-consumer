import os
import unittest
from types import TracebackType
from unittest.mock import patch

import numpy as np
import xarray as xr
from botocore.client import BaseClient as BotocoreClient
from botocore.session import Session
from moto.server import ThreadedMotoServer

from satellite_consumer.storage import get_s3_fs, write_to_zarr


class MockS3Bucket:

    client: BotocoreClient
    server: ThreadedMotoServer
    bucket: str = "test-bucket"
    region: str = "us-east-1"
    endpoint: str = "http://localhost:5000"

    def __enter__(self) -> None:
        """Create a mock S3 server and bucket."""
        self.server = ThreadedMotoServer()
        self.server.start()

        session = Session()
        self.client = session.create_client(
            service_name="s3",
            region_name=self.region,
            endpoint_url=self.endpoint,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

        self.client.create_bucket(
            Bucket=self.bucket,
        )

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
        )
        """Delete all objects in the bucket and stop the server."""
        if "Contents" in response:
            for obj in response["Contents"]:
                self.client.delete_object(
                    Bucket=self.bucket,
                    Key=obj["Key"],
                )
        self.server.stop()

    @staticmethod
    def patch_dict() -> dict[str, str]:
        """Get the patch dict for an environ modification."""
        return {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_ENDPOINT": MockS3Bucket.endpoint,
            "AWS_REGION": MockS3Bucket.region,
        }


class TestGetS3FS(unittest.TestCase):
    """Test function to get an S3 filesystem."""

    @patch.dict(os.environ, MockS3Bucket.patch_dict())
    def test_get_s3fs(self) -> None:
        """Test that the function returns a filesystem."""
        with MockS3Bucket():
            fs = get_s3_fs()
            self.assertIsNotNone(fs)
            self.assertTrue(fs.isdir("s3://test-bucket/"))

class TestWriteToZarr(unittest.TestCase):
    """Test writing to a Zarr store."""

    @patch.dict(os.environ, MockS3Bucket.patch_dict())
    @unittest.skip("ot yet working")
    def test_writes_to_s3(self) -> None:
        """Test that the function writes to an S3 bucket."""

        da: xr.DataArray = xr.DataArray(
           coords={
               "time": [np.datetime64("2021-01-01T00:00:00")],
               "x_geostationary": [1, 2, 3, 4],
               "y_geostationary": [1, 2, 3, 4],
               "variable": ["VIS006", "IR_016"],
            },
           data=np.random.rand(1, 4, 4, 2),
        )

        with MockS3Bucket():
            write_to_zarr(da, "s3://test-bucket/test.zarr")
            fs = get_s3_fs()
            self.assertTrue(fs.isdir("s3://test-bucket/test.zarr/"))


