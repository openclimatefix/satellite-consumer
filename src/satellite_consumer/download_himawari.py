"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import itertools
import logging
import re
from collections.abc import Iterator

import fsspec
import pandas as pd
import s3fs

from satellite_consumer import models
from satellite_consumer.storage import ListingCache, get_fs

log = logging.getLogger("sat_consumer")


def get_timestamp_from_filename(filename: str) -> dt.datetime:
    """Extract timestamp from a filename.

    Args:
        filename: The filename to extract the timestamp from.

    Returns:
        The timestamp extracted from the filename.
    """
    # Example filename: 'HS_H09_20221105_0020_B01_FLDK_R10_S0110.DAT.bz2'

    match = re.search(r"_(\d{8})_(\d{4})_", filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not contain a valid timestamp.")

    start_str, end_str = match.groups()
    # Convert to datetime object from YYYYMMDDHHMM format
    start_dt = dt.datetime.strptime(start_str, "%Y%m%d").replace(tzinfo=dt.UTC)
    end_dt = dt.datetime.strptime(end_str, "%H%M").replace(tzinfo=dt.UTC)
    # Combine the date and time parts
    start_time = start_dt.replace(
        hour=end_dt.hour,
        minute=end_dt.minute,
        second=0,  # Assuming seconds are not in the filename
        microsecond=0,  # Assuming microseconds are not in the filename
    )
    return start_time


def get_products_for_date_range_himawari(
    bucket: str,
    product_id: str,
    start: dt.datetime,
    end: dt.datetime,
    channels: list[str] | None = None,
    cache_dir: str | None = None,
) -> Iterator[list[str]]:
    """Lazily yield product file groups for a given date range from an S3 bucket.

    Args:
        bucket: The S3 bucket to search in.
        product_id: The product ID to search for.
        start: Start time of the search.
        end: End time of the search.
        channels: List of channels to filter the products by.
        cache_dir: Optional directory to cache S3 listing results to disk.

    Yields:
        List of product file paths for each unique start time.
    """
    fs = s3fs.S3FileSystem(anon=True)

    cache = ListingCache(cache_dir, bucket, product_id)

    log.debug(
        "Searching for products in S3 buckets",
        product_id=product_id,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    found_any = False
    try:
        for date in pd.date_range(start, end, freq="D"):
            cache_key = f"{date.year}{date.month:02d}{date.day:02d}"

            pattern = (
                f"s3://{bucket}/{product_id}/{date.year}/{date.month:02d}"
                f"/{date.day:02d}/*/*.bz2"
            )
            log.debug(f"Searching for products in S3 bucket: {pattern}")
            # The glob is the whole day's, so the period listed ends with the day.
            day_end = date.floor("D").to_pydatetime() + dt.timedelta(days=1)
            results = cache.glob(fs, pattern, cache_key, day_end)

            if channels is not None:
                results = [
                    r for r in results if any("_" + channel + "_" in r for channel in channels)
                ]
            if not results:
                continue
            found_any = True
            # Combine by start time
            start_times = [get_timestamp_from_filename(f) for f in results]
            unique_start_times = sorted(set(start_times))
            start_lists = [[] for _ in range(len(unique_start_times))]
            for result in results:
                start_time = get_timestamp_from_filename(result.split("/")[-1])
                index = unique_start_times.index(start_time)
                start_lists[index].append(result)
            yield from start_lists
    finally:
        # Also when the consumer stops iterating early.
        cache.flush()
    if not found_any:
        log.warning(
            f"No products found for {product_id} in {bucket} "
            f"between {start.year}-{start.month:02d}-{start.day:02d} "
            f"and {end.year}-{end.month:02d}-{end.day:02d}.",
        )


def get_products_iterator_himawari(
    product_id: str,
    start: dt.datetime,
    end: dt.datetime,
    channels: list[models.SpectralChannel],
    cache_dir: str | None = None,
) -> Iterator[list[str]]:
    """Get a lazy iterator over the products for a given satellite in a given time range.

    Args:
        product_id: The product ID to search for.
        start: Start time of the search.
        end: End time of the search.
        channels: The channels to search for.
        cache_dir: Optional directory to cache S3 listing results to disk.

    Returns:
        Iterator over product file groups.
    """
    log.info(f"Searching for products between {start!s} and {end!s} for {product_id}")
    cnames: list[str] = [c.name for c in channels]
    start = start.replace(tzinfo=dt.UTC)
    end = end.replace(tzinfo=dt.UTC)
    himawari_cutoff = dt.datetime(2022, 11, 4, tzinfo=dt.UTC)

    if start < himawari_cutoff and end < himawari_cutoff:
        # Only Himawari8
        return get_products_for_date_range_himawari(
            "noaa-himawari8",
            product_id,
            start,
            end,
            channels=cnames,
            cache_dir=cache_dir,
        )
    elif start >= himawari_cutoff and end >= himawari_cutoff:
        # Only Himawari9
        return get_products_for_date_range_himawari(
            "noaa-himawari9",
            product_id,
            start,
            end,
            channels=cnames,
            cache_dir=cache_dir,
        )
    else:
        # Both Himawari8 and Himawari9
        himawari8_end = himawari_cutoff if end >= himawari_cutoff else end
        himawari9_start = himawari_cutoff if start < himawari_cutoff else start
        return itertools.chain(
            get_products_for_date_range_himawari(
                "noaa-himawari8",
                product_id,
                start,
                himawari8_end,
                channels=cnames,
                cache_dir=cache_dir,
            ),
            get_products_for_date_range_himawari(
                "noaa-himawari9",
                product_id,
                himawari9_start,
                end,
                channels=cnames,
                cache_dir=cache_dir,
            ),
        )


def download_raw_himawari(
    product: list[str],
    folder: str,
    filter_regex: str,
    nest_by_date: bool = True,
    retries: int = 6,
    existing_times: list[dt.datetime] | None = None,
) -> list[str]:
    """Download a product to an S3 bucket.

    EUMDAC products are collections of files, with a `.nat` file containing the data,
    and with `.xml` files containing metadata.
    This function only downloads the `.nat` files,
    skipping any files that are already present in the folder
    or that correspond to already existing times.

    Args:
        product: Product to download.
        folder: Folder to download the product to. Can be local path or S3 URL.
        filter_regex: Regular expression to filter the files to download.
        retries: Number of times to retry downloading the product.
        existing_times: List of existing times that do not need to be redownloaded.

    Returns:
        Path to the downloaded file, or None if the download failed.
    """
    fs = get_fs(path=folder)
    fs_s3 = fsspec.filesystem("s3", anon=True)
    # Filter to only product files we care about
    raw_files = [p for p in product if re.search(filter_regex, p)]
    if not raw_files:
        log.warning(
            f"No files found for product '{product}' with filter '{filter_regex}'. "
            "Skipping download.",
        )
        return []
    downloaded_files: list[str] = []

    if existing_times is not None:
        rounded_time = (
            pd.Timestamp(get_timestamp_from_filename(raw_files[0]))
            .floor("10min")
            .to_pydatetime()
            .replace(tzinfo=dt.UTC)
        )
        if rounded_time in existing_times:
            log.debug(
                "Skipping product that exists in store",
                rounded_time=rounded_time.strftime("%Y-%m-%dT%H:%M"),
            )
            return []

    for i, raw_file in enumerate(raw_files):
        filename = raw_file.split("/")[-1]
        filepath: str = f"{folder}/{filename}"
        raw_file = "s3://" + raw_file if not raw_file.startswith("s3://") else raw_file
        try:
            if fs.exists(filepath):
                log.debug("Skipping already downloaded file", filename=raw_file)
                downloaded_files.append(filepath)
                continue
        except Exception as e:
            raise OSError(
                f"Could not determine if file '{filepath}' exists: '{e}'"
                "Ensure you have the required access permissions.",
            ) from e

        log.debug(
            "Downloading raw file",
            src=raw_file,
            dst=filepath,
            num=f"{i + 1}/{len(raw_files)}",
        )
        for i in range(retries + 1):
            try:
                # Copying to temp then putting seems to be quicker than copying to fs
                fs_s3.download(raw_file, filepath)
                downloaded_files.append(filepath)
                break
            except Exception as e:
                log.warning(
                    f"Error downloading product '{product}' (attempt {i}/{retries}): '{e}'",
                )

        if i == retries:
            log.error(
                f"Failed to download output '{raw_file}' after {retries} attempts.",
            )
            return []

    return downloaded_files
