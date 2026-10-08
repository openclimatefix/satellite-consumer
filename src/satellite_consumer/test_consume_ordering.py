"""Tests that EUMETSAT scans are matched to the store's times and appended only in order."""

import asyncio
import datetime as dt
import re
import tempfile
import unittest
from typing import Any
from unittest import mock

import eumdac.errors
import eumdac.product
import numpy as np
import pandas as pd
import requests
import xarray as xr

from satellite_consumer import consume
from satellite_consumer.download_eumetsat import download_raw
from satellite_consumer.exceptions import DownloadError, NotYetAvailableError, ValidationError

DAY = "2026-10-08"


def _product(start: str) -> mock.Mock:
    """An MTG FCI scan sensed from `start` (HH:MM), as the Data Store lists it."""
    sensing_start = pd.Timestamp(f"{DAY}T{start}:06").to_pydatetime()
    product = mock.Mock(spec=eumdac.product.Product)
    product.sensing_start = sensing_start
    product.sensing_end = sensing_start + dt.timedelta(minutes=9, seconds=21)
    product._id = f"FCI_{start}"
    return product


def _dataset(time: str) -> xr.Dataset:
    """A processed scan, at the store time `time` (HH:MM): its repeat cycle's end."""
    return _store([time])


def _store(times: list[str]) -> xr.Dataset:
    return xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((len(times), 2, 2), dtype=np.float32))},
        coords={"time": [np.datetime64(f"{DAY}T{t}") for t in times]},
    )


def _nominal_end(start: str) -> str:
    return (pd.Timestamp(f"{DAY}T{start}") + pd.Timedelta(minutes=10)).strftime("%H:%M")


def _forbidden() -> eumdac.errors.EumdacError:
    """The error eumdac raises for a product the account may not download yet."""
    try:
        raise requests.HTTPError("403 Client Error: Forbidden")
    except requests.HTTPError as http_error:
        try:
            raise eumdac.errors.EumdacError(
                "Could not download Product - Unauthorised (403)",
                {"status": 403},
            ) from http_error
        except eumdac.errors.EumdacError as e:
            return e


def _not_yet_available() -> NotYetAvailableError:
    try:
        raise NotYetAvailableError("not yet released") from _forbidden()
    except NotYetAvailableError as e:
        return e


class TestConsumeOrdering(unittest.TestCase):
    """Runs `consume_to_store` over listed scans, with downloads and the store mocked."""

    def _run(
        self,
        store_times: list[str],
        products: list[mock.Mock],
        outcomes: dict[str, Any] | None = None,
        allow_out_of_order: bool = False,
    ) -> tuple[list[str], list[str], tuple[int, int, int]]:
        """Return the scans downloaded, the times written and the (skips, errs, total)."""
        outcomes = outcomes or {}
        downloaded: list[str] = []
        written: list[str] = []

        def download_and_process(product: mock.Mock, **_: object) -> object:
            start = product._id.removeprefix("FCI_")
            downloaded.append(start)
            return outcomes.get(start, _dataset(_nominal_end(start)))

        def write_to_store(ds: xr.Dataset, **_: object) -> None:
            written.extend(pd.Timestamp(t).strftime("%H:%M") for t in ds.time.values)

        with (
            mock.patch.object(
                consume.storage,
                "get_existing_dataset",
                return_value=_store(store_times) if store_times else None,
            ),
            mock.patch.object(consume.storage, "write_to_store", side_effect=write_to_store),
            mock.patch.object(consume, "get_products_iterator", return_value=iter(products)),
            mock.patch.object(consume, "_download_and_process", download_and_process),
            self.assertLogs("sat_consumer", level="INFO") as logs,
        ):
            asyncio.run(
                consume.consume_to_store(
                    dt_range=(
                        dt.datetime.fromisoformat(f"{DAY}T01:00:00+00:00"),
                        dt.datetime.fromisoformat(f"{DAY}T04:30:00+00:00"),
                    ),
                    jump_to_latest=False,
                    cadence_mins=10,
                    product_id="EO:EUM:DAT:0662",
                    filter_regex=r"\.nc$",
                    raw_zarr_paths=("raw", "store.zarr"),
                    keep_raw=False,
                    channels=[],
                    resolution_meters=2000,
                    crop_region_lonlat=None,
                    encoding={},
                    eumetsat_credentials=("key", "secret"),
                    buffer_size=3,
                    max_workers=1,
                    accum_writes=1,
                    executor="threads",
                    request_timeout=60,
                    source="eumetsat",
                    allow_out_of_order=allow_out_of_order,
                ),
            )
        summary = re.search(
            r"skips=(\d+), errs=(\d+), finished (\d+) writes",
            "\n".join(logs.output),
        )
        if summary is None:
            self.fail("the consumer logged no summary line")
        skips, errs, total = (int(g) for g in summary.groups())
        return downloaded, written, (skips, errs, total)

    def test_a_scan_is_matched_to_its_nominal_end_in_the_store(self) -> None:
        """The 03:30 scan is stored as 03:40, so it is not fetched again, but 03:40 is.

        The scan's floored sensing end (03:40 for the 03:40 scan) was compared with the
        store's times, which are the nominal ends: a scan was taken as stored when the one
        before it was, and one after a failed scan was never fetched.
        """
        downloaded, written, _ = self._run(
            store_times=["03:30", "03:40"],
            products=[_product("03:30"), _product("03:40")],
        )
        self.assertEqual(downloaded, ["03:40"])
        self.assertEqual(written, ["03:50"])

    def test_a_scan_older_than_the_stores_newest_is_skipped(self) -> None:
        """A gap behind the store's newest time is left, not filled out of order."""
        downloaded, written, (skips, errs, total) = self._run(
            store_times=["03:40", "04:10"],
            products=[_product(t) for t in ("03:30", "03:40", "03:50", "04:00", "04:10")],
        )
        self.assertEqual(downloaded, ["04:10"])
        self.assertEqual(written, ["04:20"])
        self.assertEqual((skips, errs, total - skips - errs), (2, 0, 1))

    def test_a_processed_scan_older_than_the_stores_newest_is_not_written(self) -> None:
        """The time the scan comes out with is checked too, not only the listing's."""
        downloaded, written, (skips, _, _) = self._run(
            store_times=["04:10"],
            products=[_product("04:10")],
            outcomes={"04:10": _dataset("04:00")},
        )
        self.assertEqual(downloaded, ["04:10"])
        self.assertEqual(written, [])
        self.assertEqual(skips, 1)

    def test_scans_after_one_not_yet_available_are_held_back(self) -> None:
        """The 04:00 scan is not appended ahead of the embargoed 03:40 and 03:50 ones."""
        downloaded, written, (skips, errs, total) = self._run(
            store_times=["03:40"],
            products=[_product(t) for t in ("03:30", "03:40", "03:50", "04:00", "04:10")],
            outcomes={
                "03:40": _not_yet_available(),
                "03:50": _not_yet_available(),
                "04:10": _not_yet_available(),
            },
        )
        self.assertEqual(written, [])
        self.assertEqual(downloaded[0], "03:40")
        # Nothing an embargo holds back counts as an error, nor as written
        self.assertEqual(errs, 0)
        self.assertEqual(total - skips - errs, 0)

    def test_the_scans_before_one_not_yet_available_are_written(self) -> None:
        _, written, (skips, errs, total) = self._run(
            store_times=["03:30"],
            products=[_product(t) for t in ("03:20", "03:30", "03:40", "03:50")],
            outcomes={"03:40": _not_yet_available()},
        )
        self.assertEqual(written, ["03:40"])
        self.assertEqual((skips, errs, total - skips - errs), (2, 0, 1))

    def test_scans_after_a_failed_download_are_held_back(self) -> None:
        _, written, (skips, errs, total) = self._run(
            store_times=["03:40"],
            products=[_product(t) for t in ("03:30", "03:40", "03:50", "04:00")],
            outcomes={"03:40": DownloadError("connection reset")},
        )
        self.assertEqual(written, [])
        self.assertEqual(errs, 1)
        self.assertEqual(total - skips - errs, 0)

    def test_an_invalid_scan_does_not_hold_back_the_rest(self) -> None:
        """Bad data will not get better, so the scans after it are appended."""
        _, written, (skips, errs, _) = self._run(
            store_times=["03:40"],
            products=[_product(t) for t in ("03:40", "03:50")],
            outcomes={"03:40": ValidationError("missing channels")},
        )
        self.assertEqual(written, ["04:00"])
        self.assertEqual((skips, errs), (1, 0))

    def test_out_of_order_appends_can_be_allowed_for_a_backfill(self) -> None:
        _, written, _ = self._run(
            store_times=["03:40", "04:10"],
            products=[_product(t) for t in ("03:40", "03:50", "04:00")],
            outcomes={"03:40": DownloadError("connection reset")},
            allow_out_of_order=True,
        )
        self.assertEqual(written, ["04:00"])


class TestDownloadForbidden(unittest.TestCase):
    """A 403 for a recent scan is not retried: the scan is not released yet."""

    def _download(self, sensed_ago: dt.timedelta) -> tuple[Exception, int]:
        product = mock.Mock(spec=eumdac.product.Product)
        product._id = "FCI"
        product.entries = ["scan.nc"]
        product.sensing_end = (dt.datetime.now(tz=dt.UTC) - sensed_ago).replace(tzinfo=None)
        product.open.side_effect = _forbidden()
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(DownloadError) as ctx:
            download_raw(product, folder=folder, filter_regex=r"\.nc$", nest_by_date=False)
        return ctx.exception, product.open.call_count

    def test_a_recent_scan_forbidden_is_not_yet_available(self) -> None:
        error, attempts = self._download(dt.timedelta(minutes=30))
        self.assertIsInstance(error, NotYetAvailableError)
        self.assertEqual(attempts, 1)

    def test_an_old_scan_forbidden_is_an_error(self) -> None:
        error, attempts = self._download(dt.timedelta(days=2))
        self.assertNotIsInstance(error, NotYetAvailableError)
        self.assertEqual(attempts, 6)

    def test_a_scan_not_yet_available_is_not_retried_as_a_filesystem_error(self) -> None:
        downloader = mock.Mock(side_effect=_not_yet_available())
        with mock.patch.dict(consume.DOWNLOADERS, {"eumetsat": downloader}):
            result = consume._download_and_process(
                _product("03:40"),
                folder="raw",
                filter_regex=r"\.nc$",
                channels=[],
                resolution_meters=2000,
                crop_region_lonlat=None,
                keep_raw=True,
            )
        self.assertIsInstance(result, NotYetAvailableError)
        self.assertEqual(downloader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
