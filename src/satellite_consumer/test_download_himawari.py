import datetime as dt
import unittest
from unittest import mock

from satellite_consumer import download_himawari

DAY = "s3://noaa-himawari9/AHI-L1b-FLDK/2026/10/06"


def _scan(hhmm: str, bands: dict[str, range]) -> list[str]:
    return [
        f"{DAY}/{hhmm}/HS_H09_20261006_{hhmm}_B{band}_FLDK_R20_S{seg:02d}10.DAT.bz2"
        for band, segments in bands.items()
        for seg in segments
    ]


class FakeS3:
    def __init__(self, files: list[str]) -> None:
        self.files = files

    def glob(self, pattern: str) -> list[str]:
        return list(self.files)


def _groups(files: list[str], start: dt.datetime, end: dt.datetime, now: dt.datetime) -> list:
    clock = mock.Mock(wraps=dt.datetime)
    clock.now.return_value = now
    with (
        mock.patch.object(download_himawari.s3fs, "S3FileSystem", return_value=FakeS3(files)),
        mock.patch.object(download_himawari.dt, "datetime", clock),
    ):
        return list(
            download_himawari.get_products_for_date_range_himawari(
                "noaa-himawari9", "AHI-L1b-FLDK", start, end, channels=["B09", "B16"],
            ),
        )


def _at(hhmm: str) -> dt.datetime:
    return dt.datetime(2026, 10, 6, int(hhmm[:2]), int(hhmm[2:]), tzinfo=dt.UTC)


class TestMissingSegments(unittest.TestCase):
    def test_a_whole_scan_lacks_nothing(self) -> None:
        files = _scan("1700", {"09": range(1, 11), "16": range(1, 11)})
        self.assertEqual(download_himawari.missing_segments(files, ["B09", "B16"]), {})

    def test_a_scan_still_arriving_lacks_its_later_segments(self) -> None:
        files = _scan("1700", {"09": range(1, 11), "16": range(1, 7)})
        self.assertEqual(
            download_himawari.missing_segments(files, ["B09", "B16"]),
            {"B16": [7, 8, 9, 10]},
        )

    def test_an_absent_channel_lacks_everything(self) -> None:
        files = _scan("1700", {"09": range(1, 11)})
        self.assertEqual(download_himawari.missing_segments(files, ["B09", "B16"]), {"B16": []})


class TestScanGroups(unittest.TestCase):
    FILES = (
        _scan("0000", {"09": range(1, 11), "16": range(1, 11)})
        + _scan("1650", {"09": range(1, 11), "16": range(1, 11)})
        + _scan("1700", {"09": range(1, 11), "16": range(1, 11)})
        + _scan("2030", {"09": range(1, 11), "16": range(1, 11)})
        # Listed mid-upload: B09 whole, B16 only its first six segments.
        + _scan("2040", {"09": range(1, 11), "16": range(1, 7)})
    )

    def test_only_the_scans_in_the_window_are_yielded(self) -> None:
        groups = _groups(self.FILES, _at("1650"), _at("2030"), now=_at("2052"))
        starts = [download_himawari.get_timestamp_from_filename(g[0]) for g in groups]
        self.assertEqual(starts, [_at("1650"), _at("1700"), _at("2030")])

    def test_each_group_is_one_scan(self) -> None:
        for group in _groups(self.FILES, _at("1650"), _at("2040"), now=_at("2052")):
            self.assertEqual(len({f.split("/")[-1][7:20] for f in group}), 1)
            self.assertEqual(len(group), 20)

    def test_a_scan_still_arriving_is_left_for_a_later_run(self) -> None:
        groups = _groups(self.FILES, _at("2030"), _at("2050"), now=_at("2052"))
        self.assertEqual([g[0].split("/")[-2] for g in groups], ["2030"])

    def test_a_scan_long_incomplete_is_yielded_regardless(self) -> None:
        groups = _groups(self.FILES, _at("2030"), _at("2050"), now=_at("2052") + dt.timedelta(hours=2))
        self.assertEqual([g[0].split("/")[-2] for g in groups], ["2030", "2040"])


class DayGlobS3(FakeS3):
    """Answers a glob the way S3 does: only the files under the day it names."""

    def glob(self, pattern: str) -> list[str]:
        import fnmatch

        return [f for f in self.files if fnmatch.fnmatch(f, pattern)]


class TestWindowAcrossMidnight(unittest.TestCase):
    def test_both_days_are_listed(self) -> None:
        """A window from 23:40 to 00:20 must reach the scans after midnight.

        pd.date_range(start, end, freq="D") stepped a day from 23:40, so it
        listed only the first day: a live run between 21:00 and 23:30 lost
        every scan past midnight, and a backfill from 22:00 found nothing.
        """
        next_day = "s3://noaa-himawari9/AHI-L1b-FLDK/2026/10/07"
        files = _scan("2350", {"09": range(1, 11), "16": range(1, 11)}) + [
            f"{next_day}/0010/HS_H09_20261007_0010_B{band}_FLDK_R20_S{seg:02d}10.DAT.bz2"
            for band in ("09", "16")
            for seg in range(1, 11)
        ]
        start = _at("2340")
        end = dt.datetime(2026, 10, 7, 0, 20, tzinfo=dt.UTC)
        clock = mock.Mock(wraps=dt.datetime)
        clock.now.return_value = end + dt.timedelta(hours=2)
        with (
            mock.patch.object(download_himawari.s3fs, "S3FileSystem", return_value=DayGlobS3(files)),
            mock.patch.object(download_himawari.dt, "datetime", clock),
        ):
            groups = list(
                download_himawari.get_products_for_date_range_himawari(
                    "noaa-himawari9", "AHI-L1b-FLDK", start, end, channels=["B09", "B16"],
                ),
            )
        self.assertEqual([g[0].split("/")[-2] for g in groups], ["2350", "0010"])


if __name__ == "__main__":
    unittest.main()
