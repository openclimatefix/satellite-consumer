"""Tests that concurrent Himawari scans keep their decompressed segments to themselves."""

import bz2
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import xarray as xr

from satellite_consumer import consume
from satellite_consumer.download_himawari import decompress_bz2
from satellite_consumer.exceptions import DownloadError


def _segment(folder: str, hhmm: str, seg: int) -> str:
    path = os.path.join(folder, f"HS_H09_20261006_{hhmm}_B13_FLDK_R20_S{seg:02d}10.DAT.bz2")
    with open(path, "wb") as f:
        f.write(bz2.compress(f"{hhmm}-{seg}".encode()))
    return path


def _dataset() -> xr.Dataset:
    return xr.Dataset(coords={"time": [np.datetime64("2026-10-06T15:10")]})


def _read(path: str) -> bytes:
    """Read a segment as satpy's ahi_hsd reader would, decompressing it if it is .bz2."""
    if path.endswith(".bz2"):
        from satpy.readers.core.utils import unzip_file

        path = unzip_file(path)
    with open(path, "rb") as f:
        return f.read()


class TestDecompressBz2(unittest.TestCase):
    def test_decompresses_into_the_directory_given(self) -> None:
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as dest:
            src = _segment(raw, "1500", 6)
            plain = os.path.join(raw, "other.nc")
            open(plain, "wb").close()

            out = decompress_bz2([src, plain], dest)

            self.assertEqual(
                out, [os.path.join(dest, "HS_H09_20261006_1500_B13_FLDK_R20_S0610.DAT"), plain],
            )
            with open(out[0], "rb") as f:
                self.assertEqual(f.read(), b"1500-6")


class TestConcurrentScans(unittest.TestCase):
    def setUp(self) -> None:
        self.cwd = os.getcwd()
        self.work = tempfile.TemporaryDirectory()
        os.chdir(self.work.name)
        self.raw = os.path.join(self.work.name, "raw")
        os.mkdir(self.raw)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.work.cleanup()

    def _run(self, product: list[str], process_raw: mock.Mock, download: object) -> object:
        with (
            mock.patch.dict(consume.DOWNLOADERS, {"himawari": download}),
            mock.patch.object(consume, "process_raw", process_raw),
        ):
            return consume._download_and_process(
                product,
                folder=self.raw,
                filter_regex=r"\.bz2$",
                channels=[],
                resolution_meters=2000,
                crop_region_lonlat=None,
                keep_raw=True,
                source="himawari",
            )

    def test_a_scan_finishing_leaves_another_scans_segments_in_place(self) -> None:
        """Scan 1500 reads its segments only after scan 1510 has finished and cleaned up.

        Scan 1510 is set up only once scan 1500 is being read, as worker threads overlap
        in a run: with satpy decompressing into its shared `tmp_dir`, scan 1500's
        segments were written to scan 1510's directory and deleted with it.
        """
        scans = {hhmm: [_segment(self.raw, hhmm, s) for s in (1, 6)] for hhmm in ("1500", "1510")}
        first_reading = threading.Event()
        second_done = threading.Event()

        def download(product: list[str], **_: object) -> list[str]:
            if "_1510_" in product[0]:
                self.assertTrue(first_reading.wait(timeout=10))
            return product

        def process_raw(paths: list[str], **_: object) -> xr.Dataset:
            name = os.path.basename(paths[0])
            if "_1500_" in name:
                first_reading.set()
                self.assertTrue(second_done.wait(timeout=10))
            contents = [_read(p) for p in paths]
            self.assertEqual(contents, [f"{name[16:20]}-{s}".encode() for s in (1, 6)])
            return _dataset()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                self._run, scans["1500"], mock.Mock(side_effect=process_raw), download,
            )
            second = pool.submit(
                self._run, scans["1510"], mock.Mock(side_effect=process_raw), download,
            )
            second_result = second.result(timeout=20)
            second_done.set()
            first_result = first.result(timeout=20)

        self.assertIsInstance(second_result, xr.Dataset)
        self.assertIsInstance(first_result, xr.Dataset)
        # Each scan's decompression directory is gone once the scan is processed
        self.assertEqual([d for d in os.listdir(".") if d != "raw"], [])


class TestRetry(unittest.TestCase):
    def setUp(self) -> None:
        self.cwd = os.getcwd()
        self.work = tempfile.TemporaryDirectory()
        os.chdir(self.work.name)
        self.product = [_segment(self.work.name, "1500", 1)]

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.work.cleanup()

    def _run(self, process_raw: mock.Mock) -> object:
        with (
            mock.patch.dict(consume.DOWNLOADERS, {"himawari": lambda product, **_: product}),
            mock.patch.object(consume, "process_raw", process_raw),
        ):
            return consume._download_and_process(
                self.product,
                folder=self.work.name,
                filter_regex=r"\.bz2$",
                channels=[],
                resolution_meters=2000,
                crop_region_lonlat=None,
                keep_raw=True,
                source="himawari",
            )

    @staticmethod
    def _missing_file() -> DownloadError:
        try:
            raise FileNotFoundError(2, "No such file or directory", "/work/tmpx/06uth0sz90")
        except FileNotFoundError as e:
            try:
                raise DownloadError(f"Error reading paths as satpy Scene: {e}") from e
            except DownloadError as wrapped:
                return wrapped

    def test_a_filesystem_error_is_retried_once(self) -> None:
        process_raw = mock.Mock(side_effect=[self._missing_file(), _dataset()])
        self.assertIsInstance(self._run(process_raw), xr.Dataset)
        self.assertEqual(process_raw.call_count, 2)

    def test_a_filesystem_error_that_persists_is_returned(self) -> None:
        process_raw = mock.Mock(side_effect=[self._missing_file(), self._missing_file()])
        self.assertIsInstance(self._run(process_raw), DownloadError)
        self.assertEqual(process_raw.call_count, 2)

    def test_other_errors_are_not_retried(self) -> None:
        process_raw = mock.Mock(side_effect=DownloadError("bad calibration"))
        self.assertIsInstance(self._run(process_raw), DownloadError)
        self.assertEqual(process_raw.call_count, 1)


if __name__ == "__main__":
    unittest.main()
