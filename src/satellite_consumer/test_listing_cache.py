import datetime as dt
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from satellite_consumer import download_gk2a, download_goes, download_himawari, storage
from satellite_consumer.storage import LISTING_CACHE_MAX_AGE, ListingCache

NOW = dt.datetime(2026, 9, 25, 12, 0, tzinfo=dt.UTC)
OLD = NOW - dt.timedelta(days=30)
RECENT = NOW - dt.timedelta(days=2)


class TestListingCache(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()

    def _cache(self, now: dt.datetime = NOW) -> ListingCache:
        return ListingCache(self.dir, "bucket", "product", now=now)

    def test_an_old_period_is_kept_across_runs(self) -> None:
        cache = self._cache()
        cache.put("k", OLD, ["a.nc"])
        cache.flush()
        self.assertEqual(self._cache().get("k", OLD), ["a.nc"])

    def test_a_period_within_the_week_is_never_written(self) -> None:
        cache = self._cache()
        cache.put("k", RECENT, ["a.nc"])
        cache.flush()
        self.assertIsNone(self._cache().get("k", RECENT))
        self.assertFalse(os.listdir(self.dir))

    def test_a_period_within_the_week_is_never_read(self) -> None:
        """Even one on disk, written by an older consumer or by hand."""
        path = storage._listing_cache_path(self.dir, "bucket", "product")
        with open(path, "w") as f:
            json.dump({"k": ["stale.nc"]}, f)
        self.assertIsNone(self._cache().get("k", RECENT))
        self.assertEqual(self._cache().get("k", OLD), ["stale.nc"])

    def test_the_week_is_counted_back_from_now(self) -> None:
        edge = NOW - LISTING_CACHE_MAX_AGE
        cache = self._cache()
        self.assertTrue(cache.cacheable(edge))
        self.assertFalse(cache.cacheable(edge + dt.timedelta(seconds=1)))

    def test_a_naive_time_is_utc(self) -> None:
        self.assertTrue(self._cache().cacheable(OLD.replace(tzinfo=None)))
        self.assertFalse(self._cache().cacheable(RECENT.replace(tzinfo=None)))

    def test_no_directory_disables_it(self) -> None:
        cache = ListingCache(None, "bucket", "product", now=NOW)
        cache.put("k", OLD, ["a.nc"])
        cache.flush()
        self.assertIsNone(cache.get("k", OLD))

    def test_two_writers_to_one_file_keep_both(self) -> None:
        """Two resolutions of one satellite share a cache file."""
        first, second = self._cache(), self._cache()
        first.put("a", OLD, ["a.nc"])
        second.put("b", OLD, ["b.nc"])
        first.flush()
        second.flush()
        reread = self._cache()
        self.assertEqual(reread.get("a", OLD), ["a.nc"])
        self.assertEqual(reread.get("b", OLD), ["b.nc"])

    def test_an_unreadable_file_is_a_relisting_not_a_failure(self) -> None:
        path = storage._listing_cache_path(self.dir, "bucket", "product")
        with open(path, "w") as f:
            f.write("{truncated")
        self.assertIsNone(self._cache().get("k", OLD))

    def test_it_writes_as_it_goes(self) -> None:
        """A long backfill that dies keeps most of what it listed."""
        cache = self._cache()
        for i in range(storage.LISTING_CACHE_FLUSH_EVERY):
            cache.put(f"k{i}", OLD, [])
        self.assertEqual(self._cache().get("k0", OLD), [])

    def test_glob_lists_only_what_it_must(self) -> None:
        fs = mock.Mock()
        fs.glob.return_value = ["a.nc"]
        cache = self._cache()
        cache.glob(fs, "s3://b/old/*", "old", OLD)
        cache.glob(fs, "s3://b/old/*", "old", OLD)
        cache.glob(fs, "s3://b/new/*", "new", RECENT)
        cache.glob(fs, "s3://b/new/*", "new", RECENT)
        self.assertEqual(
            [c.args[0] for c in fs.glob.call_args_list],
            ["s3://b/old/*", "s3://b/new/*", "s3://b/new/*"],
        )


class FakeS3:
    """Answers each glob with one file named for the period the glob lists."""

    def __init__(self, name_for: "callable") -> None:
        self.name_for = name_for
        self.globs: list[str] = []

    def glob(self, pattern: str) -> list[str]:
        self.globs.append(pattern)
        return [pattern.rsplit("/", 1)[0] + "/" + self.name_for(pattern)]


def _goes_name(pattern: str) -> str:
    year, doy, hour = re.search(r"/(\d{4})/(\d{3})/(\d{2})/", pattern).groups()
    stamp = f"{year}{doy}{hour}00000"
    return f"OR_ABI-L1b-RadF-M6C01_G16_s{stamp}_e{stamp}_c{stamp}.nc"


def _gk2a_name(pattern: str) -> str:
    ym, day, hour = re.search(r"/(\d{6})/(\d{2})/(\d{2})/", pattern).groups()
    return f"gk2a_ami_le1b_vi004_fd010ge_{ym}{day}{hour}00.nc"


def _himawari_name(pattern: str) -> str:
    year, month, day = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", pattern).groups()
    return f"HS_H09_{year}{month}{day}_0000_B01_FLDK_R10_S0110.DAT.bz2"


class TestTheDownloadersCache(unittest.TestCase):
    """GOES, GK-2A and Himawari: an old span is listed once, a new one always."""

    CASES = (
        (download_goes, "get_products_for_date_range_goes", "noaa-goes19", "ABI-L1b-RadF",
         ["C01"], _goes_name, dt.timedelta(hours=1)),
        (download_gk2a, "get_products_for_date_range_gk2a", "noaa-gk2a-pds", "AMI/L1B/FD",
         ["VI004"], _gk2a_name, dt.timedelta(hours=1)),
        (download_himawari, "get_products_for_date_range_himawari", "noaa-himawari9",
         "AHI-L1b-FLDK", ["B01"], _himawari_name, dt.timedelta(days=1)),
    )  # fmt: skip

    def _list(self, case, cache_dir: str, start: dt.datetime, end: dt.datetime) -> FakeS3:
        module, func, bucket, product, channels, name_for, _ = case
        fs = FakeS3(name_for)
        real_cache = storage.ListingCache

        def cache_at_now(*args, **kwargs):
            return real_cache(*args, now=NOW, **kwargs)

        with (
            mock.patch.object(module.s3fs, "S3FileSystem", return_value=fs),
            mock.patch.object(module, "ListingCache", cache_at_now),
        ):
            groups = list(
                getattr(module, func)(
                    bucket, product, start, end, channels=channels, cache_dir=cache_dir,
                ),
            )
        self.assertTrue(groups)
        return fs

    def test_an_old_span_is_listed_once(self) -> None:
        for case in self.CASES:
            with self.subTest(case[1]), tempfile.TemporaryDirectory() as cache_dir:
                end = OLD + 3 * case[-1]
                first = self._list(case, cache_dir, OLD, end)
                second = self._list(case, cache_dir, OLD, end)
                self.assertTrue(first.globs)
                self.assertEqual(second.globs, [])

    def test_a_span_within_the_week_is_listed_every_time(self) -> None:
        for case in self.CASES:
            with self.subTest(case[1]), tempfile.TemporaryDirectory() as cache_dir:
                end = RECENT + 3 * case[-1]
                first = self._list(case, cache_dir, RECENT, end)
                second = self._list(case, cache_dir, RECENT, end)
                self.assertEqual(second.globs, first.globs)
                self.assertTrue(second.globs)

    def test_a_span_across_the_week_relists_only_its_new_part(self) -> None:
        for case in self.CASES:
            with self.subTest(case[1]), tempfile.TemporaryDirectory() as cache_dir:
                start = NOW - LISTING_CACHE_MAX_AGE - 3 * case[-1]
                end = NOW - LISTING_CACHE_MAX_AGE + 3 * case[-1]
                first = self._list(case, cache_dir, start, end)
                second = self._list(case, cache_dir, start, end)
                self.assertTrue(0 < len(second.globs) < len(first.globs))

    def test_no_cache_dir_lists_everything_every_time(self) -> None:
        for case in self.CASES:
            with self.subTest(case[1]):
                end = OLD + 3 * case[-1]
                first = self._list(case, None, OLD, end)
                second = self._list(case, None, OLD, end)
                self.assertEqual(first.globs, second.globs)


if __name__ == "__main__":
    unittest.main()
