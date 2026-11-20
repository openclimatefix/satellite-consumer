import unittest

import numpy as np
import xarray as xr

from satellite_consumer.consume import grid_mismatch


def _scan(rows: int, cols: int = 4, times: int = 1) -> xr.Dataset:
    return xr.Dataset(
        {"data": (("time", "y_geostationary", "x_geostationary"), np.zeros((times, rows, cols)))},
    )


class TestGridMismatch(unittest.TestCase):
    """A scan on a different grid from the store is found before it is written."""

    def test_a_scan_on_the_store_grid_matches(self):
        """Only time may differ."""
        self.assertEqual(grid_mismatch(_scan(11, times=1), _scan(11, times=30)), {})

    def test_a_scan_with_more_rows_is_reported(self):
        """Himawari 2022-11 decoded a scan with 17629 rows into an 11000-row store."""
        self.assertEqual(
            grid_mismatch(_scan(17), _scan(11)),
            {"y_geostationary": (17, 11)},
        )

    def test_nothing_to_compare_against(self):
        """The first scan of a new store sets its grid."""
        self.assertEqual(grid_mismatch(_scan(17), None), {})


if __name__ == "__main__":
    unittest.main()
