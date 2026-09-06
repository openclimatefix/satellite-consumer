"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import itertools
import logging
import re
from collections.abc import Iterator

import fsspec
import pandas as pd
import s3fs

from satellite_consumer.config import SatelliteMetadata
from satellite_consumer.storage import get_fs, load_listing_cache, save_listing_cache

log = logging.getLogger("sat_consumer")

HISTORY_RANGE = {
    "goes16": (
        dt.datetime(2017, 12, 18, tzinfo=dt.UTC),
        dt.datetime(2025, 4, 7, tzinfo=dt.UTC),
    ),  # GOES-16 operational from Dec 18, 2017
    "goes17": (
        dt.datetime(2019, 2, 12, tzinfo=dt.UTC),
        dt.datetime(2023, 1, 4, tzinfo=dt.UTC),
    ),  # GOES-17 operational from Feb 12, 2019
    "goes18": (
        dt.datetime(2023, 1, 4, tzinfo=dt.UTC),
        None,
    ),  # GOES-18 operational from Jan 4, 2023
    "goes19": (
        dt.datetime(2025, 4, 7, tzinfo=dt.UTC),
        None,
    ),  # GOES-19 operational from Apr 7, 2025
}


def get_timestamp_from_filename(filename: str) -> dt.datetime:
    """Extract timestamp from a filename.

    Args:
        filename: The filename to extract the timestamp from.

    Returns:
        The timestamp extracted from the filename.
    """
    # Example filename:
    # 'goes16_ABI-L2-MCMIPC-M6_G16_s20233010000123_e20233010000456_c20233010000456.nc'
    match = re.search(r"s(\d{14})_e(\d{14})", filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not contain a valid timestamp.")

    start_str, _ = match.groups()
    start_time = dt.datetime.strptime(start_str[:-1], "%Y%j%H%M%S").replace(tzinfo=dt.UTC)
    return start_time


def get_products_for_date_range_goes(
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
        channels: The channels to search for.
        cache_dir: Optional directory to cache S3 listing results to disk.

    Yields:
        List of product file paths for each unique start time.
    """
    fs = s3fs.S3FileSystem(anon=True)
    start = start.replace(tzinfo=dt.UTC)
    end = end.replace(tzinfo=dt.UTC)

    cache: dict[str, list[str]] = {}
    if cache_dir is not None:
        cache = load_listing_cache(cache_dir, bucket, product_id)

    log.debug(
        "Searching for products in S3 buckets",
        product_id=product_id,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    found_any = False
    for date in pd.date_range(start, end, freq="h"):
        if "goes16" in bucket:
            if date >= dt.datetime(2025, 1, 1, tzinfo=dt.UTC):
                product_id = "ABI-L1b-RadF"
            else:
                product_id = "ABI-L1b-RadF-Reproc"
        if "goes17" in bucket:
            if date >= dt.datetime(2023, 1, 1, tzinfo=dt.UTC):
                product_id = "ABI-L1b-RadF"
            else:
                product_id = "ABI-L1b-RadF-Reproc"
        if "goes18" in bucket or "goes19" in bucket:
            product_id = "ABI-L1b-RadF"

        cache_key = (
            f"{bucket}_{product_id}_{date.year}"
            f"_{date.timetuple().tm_yday:03d}_{date.hour:02d}"
        )

        if cache_key in cache:
            results = cache[cache_key]
        else:
            log.debug(
                "Searching for products for date in bucket: "
                f"s3://{bucket}/{product_id}/{date.year}"
                f"/{date.timetuple().tm_yday:03d}/{date.hour:02d}/*.nc",
            )
            results = fs.glob(
                f"s3://{bucket}/{product_id}/{date.year}"
                f"/{date.timetuple().tm_yday:03d}/{date.hour:02d}/*.nc",
            )
            if cache_dir is not None:
                cache[cache_key] = results
                save_listing_cache(cache_dir, bucket, product_id, cache)

        # Filter out non-channel files
        if channels is not None:
            results = [r for r in results if any(channel + "_" in r for channel in channels)]
        if not results or len(results) < len(channels): # One file per channel
            continue
        found_any = True
        # Combine by start time
        start_times = [get_timestamp_from_filename(f.split("/")[-1]) for f in results]
        unique_start_times = sorted(set(start_times))
        start_lists = [[] for _ in range(len(unique_start_times))]
        for result in results:
            start_time = get_timestamp_from_filename(result.split("/")[-1])
            index = unique_start_times.index(start_time)
            start_lists[index].append(result)
        yield from start_lists
    if not found_any:
        log.warning(
            f"No products found for {product_id} in {bucket} "
            f"between {start.year}-{start.month:02d}-{start.day:02d} "
            f"and {end.year}-{end.month:02d}-{end.day:02d}.",
        )


def get_products_iterator_goes(
    sat_metadata: SatelliteMetadata,
        cadence_mins: int,
        credentials: tuple[str, str],
        product_id: str,
    start: dt.datetime,
    end: dt.datetime,
    missing_product_threshold: float = 0.1,
    resolution_meters: int = 2000,
    cache_dir: str | None = None,
) -> Iterator[list[str]]:
    """Get a lazy iterator over the products for a given satellite in a given time range.

    Args:
        sat_metadata: Metadata for the satellite to search for.
        start: Start time of the search.
        end: End time of the search.
        missing_product_threshold: Percentage of missing products allowed without error.
        resolution_meters: Resolution of the products to search for.
        cache_dir: Optional directory to cache S3 listing results to disk.

    Returns:
        Iterator over product file groups.
    """
    log.info(
        f"Searching for products between {start!s} and {end!s} for {sat_metadata.product_id}",
    )
    cnames: list[str] = [
        c.name for c in sat_metadata.channels if resolution_meters in c.resolution_meters
    ]
    start = start.replace(tzinfo=dt.UTC)
    end = end.replace(tzinfo=dt.UTC)

    if "goes-east" in sat_metadata.region.lower():
        if start < HISTORY_RANGE["goes16"][1] and end < HISTORY_RANGE["goes16"][1]:
            # Only GOES-16
            return get_products_for_date_range_goes(
                "noaa-goes16",
                sat_metadata.product_id,
                start,
                end,
                channels=cnames,
                cache_dir=cache_dir,
            )
        elif start >= HISTORY_RANGE["goes16"][1] and end >= HISTORY_RANGE["goes16"][1]:
            # Only GOES-19
            return get_products_for_date_range_goes(
                "noaa-goes19",
                sat_metadata.product_id,
                start,
                end,
                channels=cnames,
                cache_dir=cache_dir,
            )
        else:
            # Both GOES-16 and GOES-19
            goes_16_end = (
                HISTORY_RANGE["goes16"][1] if HISTORY_RANGE["goes16"][1] < end else end
            )
            goes_19_start = (
                HISTORY_RANGE["goes19"][0] if HISTORY_RANGE["goes19"][0] > start else start
            )
            return itertools.chain(
                get_products_for_date_range_goes(
                    "noaa-goes16",
                    sat_metadata.product_id,
                    start,
                    goes_16_end,
                    channels=cnames,
                    cache_dir=cache_dir,
                ),
                get_products_for_date_range_goes(
                    "noaa-goes19",
                    sat_metadata.product_id,
                    goes_19_start,
                    end,
                    channels=cnames,
                    cache_dir=cache_dir,
                ),
            )
    else:
        if "goes-west" not in sat_metadata.region.lower():
            raise ValueError(
                f"Unknown region '{sat_metadata.region}' "
                f"for satellite {sat_metadata.product_id}."
                "Expected 'goes-east' or 'goes-west'.",
            )
        if start < HISTORY_RANGE["goes17"][1] and end < HISTORY_RANGE["goes17"][1]:
            # Only GOES-17
            return get_products_for_date_range_goes(
                "noaa-goes17",
                sat_metadata.product_id,
                start,
                end,
                channels=cnames,
                cache_dir=cache_dir,
            )
        elif start >= HISTORY_RANGE["goes17"][1] and end >= HISTORY_RANGE["goes17"][1]:
            # Only GOES-18
            return get_products_for_date_range_goes(
                "noaa-goes18",
                sat_metadata.product_id,
                start,
                end,
                channels=cnames,
                cache_dir=cache_dir,
            )
        else:
            goes_17_end = (
                HISTORY_RANGE["goes17"][1] if HISTORY_RANGE["goes17"][1] < end else end
            )
            goes_18_start = (
                HISTORY_RANGE["goes18"][0] if HISTORY_RANGE["goes18"][0] > start else start
            )
            return itertools.chain(
                get_products_for_date_range_goes(
                    "noaa-goes17",
                    sat_metadata.product_id,
                    start,
                    goes_17_end,
                    channels=cnames,
                    cache_dir=cache_dir,
                ),
                get_products_for_date_range_goes(
                    "noaa-goes18",
                    sat_metadata.product_id,
                    goes_18_start,
                    end,
                    channels=cnames,
                    cache_dir=cache_dir,
                ),
            )


def download_raw_goes(
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
    s3_fs = fsspec.filesystem("s3", anon=True)
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
                s3_fs.download(raw_file, filepath)
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
