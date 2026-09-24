import tempfile
import unittest
from unittest import mock
from typing import Any, TypedDict

import numpy as np
import xarray as xr
import zarr.codecs

from satellite_consumer import storage
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
                        write_new=True,
                    )
                    store_ds = xr.open_zarr(test["dst"], consolidated=False)
                    self.assertTrue((store_ds.data_vars["data"].isel(time=0).values == 1.0).all())


#: Stands in for any credential in the get_icechunk_repo tests.
FAKE_CREDENTIAL = "fake"


class TestGetIcechunkRepo(unittest.TestCase):
    """The store is opened where the path says, not at a fixed bucket."""

    def _storage_for(self, path: str, **kwargs: str) -> mock.MagicMock:
        with (
            mock.patch.object(storage.icechunk, "s3_storage") as s3,
            mock.patch.object(storage.icechunk, "gcs_storage") as gcs,
            mock.patch.object(storage.icechunk.Repository, "exists", return_value=True),
            mock.patch.object(storage.icechunk.Repository, "open"),
        ):
            storage.get_icechunk_repo(path, **kwargs)
        return s3 if s3.called else gcs

    def test_s3_bucket_with_dots_and_nested_prefix(self) -> None:
        s3 = self._storage_for(
            "s3://us-west-2.opendata.source.coop/bkr/geo/mtg_1000m.icechunk",
            aws_access_key_id=FAKE_CREDENTIAL,
            aws_secret_access_key=FAKE_CREDENTIAL,
            aws_region_name="us-west-2",
        )
        s3.assert_called_once_with(
            bucket="us-west-2.opendata.source.coop",
            prefix="bkr/geo/mtg_1000m.icechunk",
            access_key_id=FAKE_CREDENTIAL,
            secret_access_key=FAKE_CREDENTIAL,
            region="us-west-2",
            endpoint_url=None,
        )

    def test_s3_bucket_other_than_source_coop(self) -> None:
        s3 = self._storage_for("s3://my-bucket/stores/goes-east_2000m.icechunk")
        self.assertEqual(s3.call_args.kwargs["bucket"], "my-bucket")
        self.assertEqual(s3.call_args.kwargs["prefix"], "stores/goes-east_2000m.icechunk")

    def test_gcs_path(self) -> None:
        for protocol in ("gs", "gcs"):
            gcs = self._storage_for(f"{protocol}://bucket/a/b.icechunk", gcs_token=FAKE_CREDENTIAL)
            gcs.assert_called_once_with(
                bucket="bucket", prefix="a/b.icechunk", application_credentials=FAKE_CREDENTIAL,
            )

    def test_unsupported_protocol_raises(self) -> None:
        with self.assertRaises(OSError):
            storage.get_icechunk_repo("ftp://host/store.icechunk")

    def test_local_path_creates_a_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/store.icechunk"
            repo = storage.get_icechunk_repo(path)
            self.assertTrue(
                storage.icechunk.Repository.exists(
                    storage=storage.icechunk.local_filesystem_storage(path=path),
                ),
            )
            self.assertIsInstance(repo, storage.icechunk.Repository)
