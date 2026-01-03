"""Tests for duplicate timestamp handling functionality."""

import datetime as dt
import tempfile
import unittest

import numpy as np
import xarray as xr

from satellite_consumer.storage import (
    _get_satellite_longitude,
    _should_overwrite,
    remove_duplicate_times,
    should_overwrite_existing,
    write_to_store,
)
from satellite_consumer.test_mocks import mocks3


class TestGetSatelliteLongitude(unittest.TestCase):
    """Tests for _get_satellite_longitude function."""

    def test_returns_value_when_present(self) -> None:
        """Test that longitude is extracted when satellite_actual_longitude exists."""
        ds = xr.Dataset(
            data_vars={
                "satellite_actual_longitude": (["time"], [9.5]),
                "data": (["time", "x"], [[1.0, 2.0]]),
            },
            coords={
                "time": [np.datetime64("2021-01-01T00:00", "ns")],
                "x": [0.0, 1.0],
            },
        )
        result = _get_satellite_longitude(ds)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 9.5, places=1)

    def test_returns_none_when_missing(self) -> None:
        """Test that None is returned when satellite_actual_longitude is missing."""
        ds = xr.Dataset(
            data_vars={
                "data": (["time", "x"], [[1.0, 2.0]]),
            },
            coords={
                "time": [np.datetime64("2021-01-01T00:00", "ns")],
                "x": [0.0, 1.0],
            },
        )
        result = _get_satellite_longitude(ds)
        self.assertIsNone(result)

    def test_returns_first_value_for_array(self) -> None:
        """Test that first value is returned for array of longitudes."""
        ds = xr.Dataset(
            data_vars={
                "satellite_actual_longitude": (["time"], [0.0, 9.5]),
                "data": (["time", "x"], [[1.0, 2.0], [3.0, 4.0]]),
            },
            coords={
                "time": [
                    np.datetime64("2021-01-01T00:00", "ns"),
                    np.datetime64("2021-01-01T00:05", "ns"),
                ],
                "x": [0.0, 1.0],
            },
        )
        result = _get_satellite_longitude(ds)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=1)


class TestShouldOverwrite(unittest.TestCase):
    """Tests for _should_overwrite function."""

    def test_prefers_zero_degree_over_9_5(self) -> None:
        """Test that 0° satellite is preferred over 9.5° satellite."""
        # New at 0° should overwrite existing at 9.5°
        self.assertTrue(_should_overwrite(9.5, 0.0))

    def test_keeps_zero_degree_over_9_5(self) -> None:
        """Test that existing 0° data is kept when new is at 9.5°."""
        # New at 9.5° should NOT overwrite existing at 0°
        self.assertFalse(_should_overwrite(0.0, 9.5))

    def test_same_position_keeps_existing(self) -> None:
        """Test that same position keeps existing (first wins)."""
        self.assertFalse(_should_overwrite(0.0, 0.0))
        self.assertFalse(_should_overwrite(9.5, 9.5))

    def test_new_has_info_existing_none(self) -> None:
        """Test that new data with info overwrites existing without info."""
        self.assertTrue(_should_overwrite(None, 0.0))
        self.assertTrue(_should_overwrite(None, 9.5))

    def test_existing_has_info_new_none(self) -> None:
        """Test that existing data with info is kept when new has none."""
        self.assertFalse(_should_overwrite(0.0, None))
        self.assertFalse(_should_overwrite(9.5, None))

    def test_both_none_keeps_existing(self) -> None:
        """Test that neither having info keeps existing."""
        self.assertFalse(_should_overwrite(None, None))

    def test_prefers_closer_to_zero(self) -> None:
        """Test preference for position closer to 0°."""
        # 3° is closer to 0° than 9.5°
        self.assertTrue(_should_overwrite(9.5, 3.0))
        self.assertFalse(_should_overwrite(3.0, 9.5))

        # Negative longitudes also work (e.g., -5° is 5° away from 0°)
        self.assertTrue(_should_overwrite(9.5, -5.0))
        self.assertFalse(_should_overwrite(-5.0, 9.5))


class TestShouldOverwriteExisting(unittest.TestCase):
    """Tests for should_overwrite_existing function with actual stores."""

    def _create_test_dataset(
        self,
        time: np.datetime64,
        longitude: float,
    ) -> xr.Dataset:
        """Create a minimal test dataset."""
        return xr.Dataset(
            coords={
                "time": [time],
                "y_geostationary": np.linspace(-100.0, 100.0, 10),
                "x_geostationary": np.linspace(-100.0, 100.0, 10),
                "channel": ["VIS"],
            },
            data_vars={
                "data": (
                    ["time", "y_geostationary", "x_geostationary", "channel"],
                    np.ones(shape=(1, 10, 10, 1)),
                ),
                "instrument": (["time"], ["METEOSAT_TEST_SATELLITE_01"]),
                "cal_slope": (["time", "channel"], [[1.0]]),
                "cal_offset": (["time", "channel"], [[0.0]]),
                "satellite_actual_longitude": (["time"], [longitude]),
                "satellite_actual_latitude": (["time"], [0.0]),
                "satellite_actual_altitude": (["time"], [35786023.0]),
                "projection_longitude": (["time"], [longitude]),
                "projection_latitude": (["time"], [0.0]),
                "projection_altitude": (["time"], [35786023.0]),
            },
        )

    def test_returns_true_for_empty_store(self) -> None:
        """Test that overwrite is allowed for non-existent store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = should_overwrite_existing(
                dst=f"{tmpdir}/nonexistent.zarr",
                time_value=dt.datetime(2021, 1, 1, 0, 0, tzinfo=dt.UTC),
                new_satellite_longitude=0.0,
            )
            self.assertTrue(result)

    def test_returns_true_when_new_preferred(self) -> None:
        """Test that overwrite is allowed when new satellite is preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/test.zarr"
            time = np.datetime64("2021-01-01T00:00", "ns")

            # Write initial data at 9.5°
            ds = self._create_test_dataset(time, longitude=9.5)
            write_to_store(
                ds=ds,
                dst=zarr_path,
                append_dim="time",
                dims=["time", "y_geostationary", "x_geostationary", "channel"],
                chunks=[1, 10, 10, 1],
                shards=[1, 10, 10, 1],
            )

            # Check if 0° data should overwrite
            result = should_overwrite_existing(
                dst=zarr_path,
                time_value=dt.datetime(2021, 1, 1, 0, 0, tzinfo=dt.UTC),
                new_satellite_longitude=0.0,
            )
            self.assertTrue(result)

    def test_returns_false_when_existing_preferred(self) -> None:
        """Test that overwrite is NOT allowed when existing satellite is preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/test.zarr"
            time = np.datetime64("2021-01-01T00:00", "ns")

            # Write initial data at 0°
            ds = self._create_test_dataset(time, longitude=0.0)
            write_to_store(
                ds=ds,
                dst=zarr_path,
                append_dim="time",
                dims=["time", "y_geostationary", "x_geostationary", "channel"],
                chunks=[1, 10, 10, 1],
                shards=[1, 10, 10, 1],
            )

            # Check if 9.5° data should overwrite
            result = should_overwrite_existing(
                dst=zarr_path,
                time_value=dt.datetime(2021, 1, 1, 0, 0, tzinfo=dt.UTC),
                new_satellite_longitude=9.5,
            )
            self.assertFalse(result)


class TestRemoveDuplicateTimes(unittest.TestCase):
    """Tests for remove_duplicate_times function."""

    def test_no_duplicates_returns_empty(self) -> None:
        """Test that empty list is returned when no duplicates exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/test.zarr"

            # Create dataset with unique timestamps
            ds = xr.Dataset(
                coords={
                    "time": [
                        np.datetime64("2021-01-01T00:00", "ns"),
                        np.datetime64("2021-01-01T00:05", "ns"),
                    ],
                    "x": [0.0, 1.0],
                },
                data_vars={
                    "data": (["time", "x"], [[1.0, 2.0], [3.0, 4.0]]),
                    "satellite_actual_longitude": (["time"], [0.0, 0.0]),
                },
            )
            ds.to_zarr(zarr_path, mode="w", consolidated=False, zarr_format=3)

            result = remove_duplicate_times(dst=zarr_path, dry_run=True)
            self.assertEqual(len(result), 0)

    def test_file_not_found_raises(self) -> None:
        """Test that FileNotFoundError is raised for non-existent store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                remove_duplicate_times(dst=f"{tmpdir}/nonexistent.zarr")


class TestWriteToStoreWithDuplicates(unittest.TestCase):
    """Integration tests for write_to_store with duplicate handling."""

    def _create_test_dataset(
        self,
        time: np.datetime64,
        longitude: float,
        data_value: float = 1.0,
    ) -> xr.Dataset:
        """Create a minimal test dataset."""
        return xr.Dataset(
            coords={
                "time": [time],
                "y_geostationary": np.linspace(-100.0, 100.0, 10),
                "x_geostationary": np.linspace(-100.0, 100.0, 10),
                "channel": ["VIS"],
            },
            data_vars={
                "data": (
                    ["time", "y_geostationary", "x_geostationary", "channel"],
                    np.full(shape=(1, 10, 10, 1), fill_value=data_value),
                ),
                "instrument": (["time"], ["METEOSAT_TEST_SATELLITE_01"]),
                "cal_slope": (["time", "channel"], [[1.0]]),
                "cal_offset": (["time", "channel"], [[0.0]]),
                "satellite_actual_longitude": (["time"], [longitude]),
                "satellite_actual_latitude": (["time"], [0.0]),
                "satellite_actual_altitude": (["time"], [35786023.0]),
                "projection_longitude": (["time"], [longitude]),
                "projection_latitude": (["time"], [0.0]),
                "projection_altitude": (["time"], [35786023.0]),
            },
        )

    def test_write_unique_timestamps(self) -> None:
        """Test that unique timestamps are written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/test.zarr"
            dims = ["time", "y_geostationary", "x_geostationary", "channel"]
            chunks = [1, 10, 10, 1]
            shards = [1, 10, 10, 1]

            # Write first timestamp
            ds1 = self._create_test_dataset(
                time=np.datetime64("2021-01-01T00:00", "ns"),
                longitude=0.0,
                data_value=1.0,
            )
            write_to_store(
                ds=ds1,
                dst=zarr_path,
                append_dim="time",
                dims=dims,
                chunks=chunks,
                shards=shards,
            )

            # Write second timestamp
            ds2 = self._create_test_dataset(
                time=np.datetime64("2021-01-01T00:05", "ns"),
                longitude=0.0,
                data_value=2.0,
            )
            write_to_store(
                ds=ds2,
                dst=zarr_path,
                append_dim="time",
                dims=dims,
                chunks=chunks,
                shards=shards,
            )

            # Verify both timestamps exist
            result = xr.open_zarr(zarr_path, consolidated=False)
            self.assertEqual(len(result.coords["time"]), 2)


if __name__ == "__main__":
    unittest.main()
