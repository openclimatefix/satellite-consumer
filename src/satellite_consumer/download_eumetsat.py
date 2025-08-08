"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import eumdac
import pandas as pd
from loguru import logger as log

from satellite_consumer.config import SatelliteMetadata
from satellite_consumer.exceptions import DownloadError
from satellite_consumer.storage import get_fs

if TYPE_CHECKING:
    from eumdac.collection import Collection, SearchResults


def get_products_iterator(
    sat_metadata: SatelliteMetadata,
    start: dt.datetime,
    end: dt.datetime,
    missing_product_threshold: float = 0.1,
    resolution_meters: int = 3000,  # noqa: ARG001
) -> Iterator[eumdac.product.Product]:
    """Get an iterator over the products for a given satellite in a given time range.

    Checks that the number of products returned matches the expected number of products.

    Args:
        sat_metadata: Metadata for the satellite to search for.
        start: Start time of the search.
        end: End time of the search.
        missing_product_threshold: Percentage of missing products allowed without error.
        resolution_meters: Resolution of the products to search for, in meters,
         not used in this one as all EUMETSAT files are in a single file.

    Returns:
        Tuple of the iterator over the products and the total number of products found.
    """
    log.info(
        f"Searching for products between {start!s} and {end!s} for {sat_metadata.product_id}",
    )
    expected_products_count = int((end - start) / dt.timedelta(minutes=sat_metadata.cadence_mins))
    token = _gen_token()
    try:
        collection: Collection = eumdac.datastore.DataStore(token=token).get_collection(
            sat_metadata.product_id,
        )
        search_results: SearchResults = collection.search(
            dtstart=start,
            dtend=end,
            sort="start,time,1",  # Sort by ascending start time
        )
    except Exception as e:
        raise DownloadError(
            f"Error searching for products for '{sat_metadata.product_id}': '{e}'",
        ) from e
    if search_results.total_results == 0:
        raise DownloadError(
            f"No products found for {sat_metadata.product_id} "
            f"in the given time range '{start!s}-{end!s}.",
        )
    if (1 - search_results.total_results / expected_products_count) > missing_product_threshold:
        raise DownloadError(
            f"Threshold for missing products exceeded: "
            f"found {search_results.total_results}/{expected_products_count} products "
            f"for {sat_metadata.product_id}. ",
        )
    log.info(
        f"Found {search_results.total_results}/{expected_products_count} products "
        f"for {sat_metadata.product_id} ",
    )
    return search_results.__iter__()


def download_customisation(
    customisation: eumdac.customisation.Customisation,
    folder: str,
    filter_regex: str,
    timeout_seconds: int = 60,
    retries: int = 6,
) -> list[str] | None:
    """Download a customisation to an S3 bucket.

    Customizations are a wrapper around a DataTailor job, which can only be downloaded
    once it is completed. As such, this function waits for completion before downloading.

    Args:
        customisation: Customisation to download.
        folder: Folder to download the product to. Can be local path or S3 URL.
        retries: Number of times to retry downloading the output.
        filter_regex: Regular expression to filter the files to download.
        timeout_seconds: Maximum time to wait for the customisation to complete.

    Returns:
        Path to the downloaded file, or None if the download failed.
    """
    fs = get_fs(path=folder)

    raw_files: list[str] = [p for p in customisation.outputs if re.search(filter_regex, p)]

    downloaded_filepaths: list[str] = []
    for raw_file in raw_files:
        filepath: str = f"{folder}/{raw_file}"
        try:
            if fs.exists(filepath):
                log.debug("Skipping already downloaded file", filename=raw_file)
                downloaded_filepaths.append(filepath)
        except Exception as e:
            raise OSError(
                f"Could not determine if file '{filepath}' exists: '{e}'"
                "Ensure you have the required access permissions.",
            ) from e

        timer: int = 0
        while timer < timeout_seconds:
            if customisation.status in ["QUEUED", "RUNNING"]:
                time.sleep(1)
                timer += 1
            else:
                break

        # Check if status is still queued after timeout
        if customisation.status in ["QUEUED", "RUNNING"]:
            try:
                log.error(
                    f"Customisation exceeded timeout ({timeout_seconds}s). Killing customisation.",
                )
                customisation.kill()
            except Exception as e:
                log.error(f"Failed to kill customisation: {e}")
            return None

        log.debug("Downloading raw file", src=raw_file, dst=filepath)
        for i in range(retries + 1):
            try:
                # Copying to temp then putting seems to be quicker than copying to fs
                with tempfile.NamedTemporaryFile(suffix=".nat") as fdst:
                    for chunk in customisation.stream_output_iter_content(output=raw_file):  # type: ignore
                        fdst.write(chunk)
                        fdst.flush()
                    fs.put(fdst.name, filepath)
                downloaded_filepaths.append(filepath)
            except Exception as e:
                log.warning(
                    f"Error downloading output '{raw_file}' (attempt {i}/{retries}): '{e}'",
                )

            if i == retries:
                raise DownloadError(
                    f"Failed to download output '{raw_file}' after {retries} attempts.",
                )

    return downloaded_filepaths


def download_raw(
    product: eumdac.product.Product,
    folder: str,
    filter_regex: str,
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
    # Filter to only product files we care about
    raw_files: list[str] = [p for p in product.entries if re.search(filter_regex, p)]

    downloaded_files: list[str] = []

    if existing_times is not None:
        rounded_time: dt.datetime = (
            pd.Timestamp(product.sensing_end).round("5 min").to_pydatetime().replace(tzinfo=dt.UTC)
        )
        if rounded_time in existing_times:
            log.debug(
                "Skipping product that exists in store",
                time=product.sensing_end.strftime("%Y-%m-%dT%H:%M"),
                rounded_time=rounded_time.strftime("%Y-%m-%dT%H:%M"),
            )
            return []

    for i, raw_file in enumerate(raw_files):
        if product.qualityStatus != "NOMINAL":
            log.warning(
                f"Encountered product '{product!s}' with non-nominal quality status "
                f"'{product.qualityStatus}'. ",
                quality=product.qualityStatus,
            )
            continue

        filepath: str = f"{folder}/{raw_file}"
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
                with (
                    product.open(raw_file) as fsrc,
                    tempfile.NamedTemporaryFile() as fdst,
                ):
                    shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                    fs.put(fdst.name, filepath)
                    if os.stat(fdst.name).st_size != fs.info(filepath).get("size", 0):
                        raise DownloadError(
                            f"Downloaded file size mismatch for '{raw_file}'. "
                            f"Expected {os.stat(fdst.name).st_size}, "
                            f"got {fs.info(filepath).get('size', 0)}.",
                        )
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


def _gen_token() -> eumdac.token.AccessToken:
    """Generated an aces token from environment variables."""
    for var in ["EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET"]:
        if var not in os.environ:
            raise DownloadError(
                "Cannot download data from EUMETSAT due to missing "
                f"required authorization environment variable '{var}'",
            )
    consumer_key: str = os.environ["EUMETSAT_CONSUMER_KEY"]
    consumer_secret: str = os.environ["EUMETSAT_CONSUMER_SECRET"]
    token = eumdac.token.AccessToken(credentials=(consumer_key, consumer_secret))

    return token
