#!/usr/bin/env python
"""Create a test Zarr store with duplicate timestamps for testing cleanup."""

import datetime as dt
import tempfile

import numpy as np
import xarray as xr

from satellite_consumer.storage import write_to_store


def create_test_store_with_duplicates(output_path: str) -> None:
    """Create a Zarr store with duplicate timestamps at different satellite positions.
    
    This simulates the scenario where a satellite transition (9.5° to 0°) creates
    duplicate timestamps in the store.
    """
    # Shared test data configuration
    dims = ["time", "y_geostationary", "x_geostationary", "channel"]
    chunks = [1, 10, 10, 1]
    shards = [1, 10, 10, 1]
    
    # Create 3 unique timestamps
    times = [
        np.datetime64("2021-01-01T00:00", "ns"),
        np.datetime64("2021-01-01T00:05", "ns"),
        np.datetime64("2021-01-01T00:10", "ns"),
    ]
    
    print(f"Creating test store at {output_path}")
    
    # Write first set of data (satellite at 9.5°)
    print("Writing data from satellite at 9.5° longitude...")
    for i, time in enumerate(times):
        ds = xr.Dataset(
            coords={
                "time": [time],
                "y_geostationary": np.linspace(-100.0, 100.0, 10),
                "x_geostationary": np.linspace(-100.0, 100.0, 10),
                "channel": ["VIS"],
            },
            data_vars={
                "data": (
                    dims,
                    np.full(shape=(1, 10, 10, 1), fill_value=float(i + 1)),
                ),
                "instrument": (["time"], ["METEOSAT_TEST_SATELLITE_01"]),
                "cal_slope": (["time", "channel"], [[1.0]]),
                "cal_offset": (["time", "channel"], [[0.0]]),
                "satellite_actual_longitude": (["time"], [9.5]),
                "satellite_actual_latitude": (["time"], [0.0]),
                "satellite_actual_altitude": (["time"], [35786023.0]),
                "projection_longitude": (["time"], [9.5]),
                "projection_latitude": (["time"], [0.0]),
                "projection_altitude": (["time"], [35786023.0]),
            },
        )
        write_to_store(ds=ds, dst=output_path, append_dim="time", dims=dims, chunks=chunks, shards=shards)
        print(f"  - Wrote timestamp {time} from 9.5° satellite")
    
    # Now create DUPLICATE timestamps from satellite at 0° (should overwrite middle one)
    # This simulates the satellite transition scenario
    print("\nSimulating satellite transition: Writing duplicate timestamp from 0° satellite...")
    duplicate_time = times[1]  # Use the middle timestamp
    
    ds_duplicate = xr.Dataset(
        coords={
            "time": [duplicate_time],
            "y_geostationary": np.linspace(-100.0, 100.0, 10),
            "x_geostationary": np.linspace(-100.0, 100.0, 10),
            "channel": ["VIS"],
        },
        data_vars={
            "data": (
                dims,
                np.full(shape=(1, 10, 10, 1), fill_value=999.0),  # Different value!
            ),
            "instrument": (["time"], ["METEOSAT_TEST_SATELLITE_01"]),
            "cal_slope": (["time", "channel"], [[1.0]]),
            "cal_offset": (["time", "channel"], [[0.0]]),
            "satellite_actual_longitude": (["time"], [0.0]),  # Different position!
            "satellite_actual_latitude": (["time"], [0.0]),
            "satellite_actual_altitude": (["time"], [35786023.0]),
            "projection_longitude": (["time"], [0.0]),
            "projection_latitude": (["time"], [0.0]),
            "projection_altitude": (["time"], [35786023.0]),
        },
    )
    
    print(f"  - Writing DUPLICATE at {duplicate_time} from 0° satellite (should be preferred)")
    print(f"  - This creates a duplicate that cleanup should handle")
    
    # Direct write to create the duplicate (bypassing the overwrite logic in consume.py)
    # This simulates what happened in the bug
    ds_duplicate.to_zarr(
        output_path,
        mode="a",
        append_dim="time",
        consolidated=False,
        zarr_format=3,
    )
    
    print(f"\n✓ Created test store with duplicates at {output_path}")
    print(f"  - Total timestamps written: 4 (3 unique + 1 duplicate)")
    print(f"  - Duplicate timestamp: {duplicate_time}")
    print(f"  - Satellite positions: 9.5° (original) and 0° (duplicate, preferred)")
    print(f"\nYou can now test cleanup with:")
    print(f"  uv run sat-cleanup {output_path} --dry-run")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "/tmp/test_duplicates.zarr"
    
    create_test_store_with_duplicates(output_path)
