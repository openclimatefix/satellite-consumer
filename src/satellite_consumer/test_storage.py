import contextlib
import dataclasses
import tempfile
import unittest
from collections.abc import Generator
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

import numpy as np
import xarray as xr
from botocore.session import Session
from moto.server import ThreadedMotoServer

from satellite_consumer.config import Coordinates
from satellite_consumer.storage import create_empty_zarr, get_fs, write_to_zarr

if TYPE_CHECKING:
    from botocore.client import BaseClient as BotocoreClient


@contextlib.contextmanager
def mocks3() -> Generator[str]:
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
            yield "s3://test-bucket/"
        finally:
            s3_client.close()
            server.stop()

class TestStorage(unittest.TestCase):
    """Test the storage functions."""

    def test_get_s3fs(self) -> None:
        """Test that the function returns a filesystem."""
        with mocks3() as dst:
            fs = get_fs(path=dst)
            self.assertIsNotNone(fs)
            self.assertTrue(fs.isdir(dst))

    def test_create_empty_zarr(self) -> None:
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
                    self.assertTrue(get_fs(test["dst"]).isdir(test["dst"]))
                    self.assertTrue(
                        np.isnan(store_da.values).all(),
                        msg="All values in empty store should be NaN",
                    )
                    self.assertDictEqual(
                        dict(store_da.sizes), {k: len(v) for k, v in coords.to_dict().items()},
                    )
                    self.assertListEqual(
                        list(store_da.dims),
                        ["time", "y_geostationary", "x_geostationary", "variable"],
                        msg="Dimension ordering of emtpy store is incorrect",
                    )
                    for coord in list(coords.to_dict().keys()):
                        self.assertListEqual(
                            list(store_da.coords[coord].values), coords.to_dict()[coord],
                            msg="Coordinate values in empty store are incorrect",
                        )

    def test_write_to_zarr(self) -> None:
        """Test that the function writes to an S3 bucket."""

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

            da: xr.DataArray = xr.DataArray(
                name="data",
                coords=dataclasses.replace(
                   coords, time=[np.datetime64("2021-01-01T00:00", "ns")],
                ).to_dict(),
                data=np.ones(shape=(1, 1392, 3712, 2)),
            )

            for test in tests:
                with self.subTest(name=test["name"]):
                    # Create an empty zarr store
                    fs = get_fs(test["dst"])
                    store_da = create_empty_zarr(dst=test["dst"], coords=coords)
                    self.assertTrue(fs.isdir(test["dst"]), msg="Zarr store doesn't exist")
                    self.assertTrue(
                        np.isnan(store_da.isel(time=0)).all(),
                        msg="Empty store is not empty",
                    )
                    # Write ones to the first time coordinate
                    write_to_zarr(da=da, dst=test["dst"])
                    store_da = xr.open_dataarray(test["dst"], engine="zarr", consolidated=False)
                    self.assertTrue((store_da.isel(time=0).values == 1.0).all())

