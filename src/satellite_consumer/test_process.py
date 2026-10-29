import datetime as dt
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from satellite_consumer.exceptions import ValidationError
from satellite_consumer.models import SpectralChannel
from satellite_consumer.process import _map_scene_to_dataset


def _make_mock_scene(data_vars: list[str]) -> MagicMock:
    """Create a mock Scene whose to_xarray_dataset returns a dataset with the given vars."""
    ds = xr.Dataset(
        coords={
            "x": np.arange(10, dtype=np.float64),
            "y": np.arange(10, dtype=np.float64),
        },
        data_vars={
            name: (["y", "x"], np.random.rand(10, 10).astype(np.float32))
            for name in data_vars
        },
        attrs={
            "start_time": dt.datetime(2024, 1, 1, 0, 0, 0),
            "end_time": dt.datetime(2024, 1, 1, 0, 15, 0),
            "platform_name": "GK2A",
            "area": MagicMock(),
            "sensor": "ami",
        },
    )
    # Add per-variable attrs matching what satpy produces
    for name in data_vars:
        ds[name].attrs = {
            "units": "%",
            "wavelength": (0.45, 0.47, 0.49),
            "standard_name": "toa_bidirectional_reflectance",
            "calibration": "reflectance",
            "resolution": 1000,
            "orbital_parameters": {
                "satellite_actual_longitude": 128.2,
                "satellite_actual_latitude": 0.0,
                "satellite_actual_altitude": 35786023.0,
                "projection_longitude": 128.2,
                "projection_latitude": 0.0,
                "projection_altitude": 35786023.0,
            },
        }

    scene = MagicMock()
    scene.to_xarray_dataset.return_value = ds
    return scene


class TestMapSceneToDatasetMissingChannels(unittest.TestCase):
    """Test that _map_scene_to_dataset raises ValidationError for missing channels."""

    def test_raises_when_channel_missing(self):
        """A missing channel should raise ValidationError."""
        scene = _make_mock_scene(["IR105", "IR096"])
        channels = [
            SpectralChannel(representation="reflectance", satpy_index=0, name="IR105"),
            SpectralChannel(representation="reflectance", satpy_index=1, name="IR096"),
            SpectralChannel(representation="reflectance", satpy_index=2, name="IR087"),
        ]

        with self.assertRaises(ValidationError) as ctx:
            _map_scene_to_dataset(scene=scene, channels=channels)

        self.assertIn("IR087", str(ctx.exception))

    def test_raises_when_all_channels_missing(self):
        """All channels missing should raise ValidationError."""
        scene = _make_mock_scene(["OTHER"])
        channels = [
            SpectralChannel(representation="reflectance", satpy_index=0, name="IR105"),
            SpectralChannel(representation="reflectance", satpy_index=1, name="IR087"),
        ]

        with self.assertRaises(ValidationError) as ctx:
            _map_scene_to_dataset(scene=scene, channels=channels)

        self.assertIn("IR105", str(ctx.exception))
        self.assertIn("IR087", str(ctx.exception))

    def test_no_error_when_all_channels_present(self):
        """All channels present should not raise."""
        scene = _make_mock_scene(["IR105", "IR096"])
        channels = [
            SpectralChannel(representation="reflectance", satpy_index=0, name="IR105"),
            SpectralChannel(representation="reflectance", satpy_index=1, name="IR096"),
        ]

        # Should not raise
        ds = _map_scene_to_dataset(scene=scene, channels=channels)
        self.assertIsInstance(ds, xr.Dataset)
