"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import eumdac.customisation
import eumdac.product
import eumdac.token
import pandas as pd

from satellite_consumer.exceptions import DownloadError
from satellite_consumer.storage import get_fs

if TYPE_CHECKING:
    from eumdac.collection import Collection, SearchResults

log = logging.getLogger(__name__)

def get_products_iterator(
    product_id: str,
    cadence_mins: int,
    start: dt.datetime,
    end: dt.datetime,
    credentials: tuple[str, str],
) -> Iterator[eumdac.product.Product]:
    """Get an iterator over the products for a given satellite in a given time range."""
    expected_products_count = int((end - start) / dt.timedelta(minutes=cadence_mins))
    token = eumdac.token.AccessToken(credentials=credentials)
    try:
        collection: Collection = eumdac.datastore.DataStore(token=token).get_collection(
            product_id,
        )
        search_results: SearchResults = collection.search(
            dtstart=start,
            dtend=end,
            sort="start,time,1",  # Sort by ascending start time
        )
    except Exception as e:
        raise DownloadError(
            f"Error searching for products for '{product_id}': '{e}'",
        ) from e
    if search_results.total_results == 0:
        raise DownloadError(
            f"No products found for {product_id} "
            f"in the given time range '{start!s}-{end!s}.",
        )
    if search_results.total_results != expected_products_count:
        log.warning(
            f"only {search_results.total_results} / {expected_products_count} products found "
            f"in upstream for {product_id}",
        )

    return search_results.__iter__()

def download_raw(
    product: eumdac.product.Product,
    folder: str,
    filter_regex: str,
    retries: int = 6,
    existing_times: list[dt.datetime] | None = None,
    cadence_mins: int = 5,
) -> list[str]:
    """Download a product to filesystem.

    EUMDAC products are collections of files, with a `.nat` file containing the data,
    and with `.xml` files containing metadata.
    This function only downloads the `.nat` files, skipping any files that are already present in
    the folder or that correspond to already existing times.
    """
    fs = get_fs(path=folder)
    # Filter to only product files we care about
    raw_files: list[str] = [p for p in product.entries if re.search(filter_regex, p)]
    rounded_time: dt.datetime = (
        pd.Timestamp(product.sensing_end)
            .round(f"{cadence_mins} min")
            .to_pydatetime()
            .astimezone(tz=dt.UTC)
    )

    downloaded_files: list[str] = []
    if existing_times is not None and rounded_time in existing_times:
        log.debug(
            "time %s exists in store, skipping",
            rounded_time.strftime("%Y-%m-%dT%H:%M"),
        )
        return []

    for raw_file in raw_files:
        if product.qualityStatus != "NOMINAL":
            log.warning("%s not nominal, skipping", product)
            continue

        date_folder: str = rounded_time.strftime("%Y/%m/%d")
        filepath: str = f"{folder}/{date_folder}/{raw_file}"
        try:
            if fs.exists(filepath):
                log.debug("file %s exists, skipping", raw_file)
                downloaded_files.append(filepath)
                continue
        except Exception as e:
            raise OSError(
                f"Could not determine if file '{filepath}' exists: '{e}'"
                "Ensure you have the required access permissions.",
            ) from e

        for i in range(retries):
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
                    f"error downloading product '{product}' (attempt {i}/{retries}): '{e}'",
                )

        if i == retries:
            raise DownloadError(
                f"Failed to download output '{raw_file}' after {retries} attempts.",
            )

    return downloaded_files

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
                log.debug("file %s exists, skipping", raw_file)
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

