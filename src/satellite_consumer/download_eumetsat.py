"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import hashlib
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
from fsspec.implementations.local import LocalFileSystem

from satellite_consumer.exceptions import DownloadError
from satellite_consumer.storage import get_fs

if TYPE_CHECKING:
    from eumdac.collection import Collection, SearchResults

log = logging.getLogger(__name__)


def calculate_md5(filepath: str) -> str:
    """Calculate MD5 hash of a file.

    Args:
        filepath: Path to the file to hash.

    Returns:
        Hexadecimal MD5 hash string.
    """
    hash_md5 = hashlib.md5()  # noqa: S324
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_md5_hash(filepath: str, expected_hash: str) -> bool:
    """Verify that a file's MD5 hash matches the expected value.

    Args:
        filepath: Path to the file to verify.
        expected_hash: Expected MD5 hash string.

    Returns:
        True if the hash matches, False otherwise.
    """
    if not expected_hash:
        log.debug("No expected hash provided, skipping MD5 verification")
        return True

    actual_hash = calculate_md5(filepath)
    if actual_hash.lower() != expected_hash.lower():
        log.warning(
            f"MD5 hash mismatch for '{filepath}': "
            f"expected '{expected_hash}', got '{actual_hash}'"
        )
        return False
    log.debug(f"MD5 verification passed for '{filepath}'")
    return True


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
            f"No products found for {product_id} in the given time range '{start!s}-{end!s}.",
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
    cadence_mins: int = 5,
    nest_by_date: bool = True,
    verify_md5: bool = False,
) -> list[str]:
    """Download a product to filesystem.

    EUMDAC products are collections of files, with a `.nat` file containing the data,
    and with `.xml` files containing metadata.
    This function downloads the whole product as a zipped archive and unpacks it locally to reduce
    network calls.

    Nesting the raw files by date is recommended if keeping files after processing as it prevents
    the creation of overly populated folders on disk.

    Args:
        product: EUMDAC product to download.
        folder: Folder to download the product to.
        filter_regex: Regular expression to filter the files to download.
        retries: Number of times to retry downloading the product.
        cadence_mins: Cadence of the product in minutes.
        nest_by_date: Whether to nest the raw files by date.
        verify_md5: Whether to verify the MD5 hash of the downloaded archive.
            Uses the hash provided by EUMETSAT to detect corrupted downloads.
            Disabled by default for backward compatibility.

    Returns:
        List of paths to the downloaded files.
    """
    fs = get_fs(path=folder)
    fs.mkdirs(path=folder, exist_ok=True)
    # Filter to only product files we care about
    product_files: list[str] = [p for p in product.entries if re.search(filter_regex, p)]
    rounded_time: dt.datetime = (
        pd.Timestamp(product.sensing_end)
        .round(f"{cadence_mins} min")
        .to_pydatetime()
        .astimezone(tz=dt.UTC)
    )

    save_folder: str = f"{folder}/{rounded_time.strftime('%Y/%m/%d')}" if nest_by_date else folder
    expected_files: list[str] = [f"{save_folder}/{name}" for name in product_files]

    try:
        if all(fs.exists(f) for f in expected_files):
            log.debug(
                "all files for product %s exist in %s, skipping",
                product,
                save_folder,
            )
            return expected_files
    except Exception as e:
        raise OSError(
            f"Could not determine if file exists: '{e}'"
            "Ensure you have the required access permissions.",
        ) from e

    # If saving raw files to S3, use a local temp directory to unzip the archives.
    # Otherwise handle the unzipping in the raw archive folder path.
    with tempfile.TemporaryDirectory(
        dir=folder if isinstance(fs, LocalFileSystem) else None,
    ) as tmpdir:
        for i in range(retries):
            try:
                # Copying to temp then putting seems to be quicker than copying to fs
                with (
                    product.open() as fsrc,
                    tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".zip") as fdst,
                ):
                    shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)

                    # Make sure the file is flushed to disk before verification
                    fdst.flush()

                    # Verify MD5 hash if enabled and hash is available
                    if verify_md5:
                        # Get expected hash from product metadata (if available)
                        expected_hash = getattr(product, "hash", None) or getattr(
                            fsrc, "headers", {}
                        ).get("Content-MD5")
                        if expected_hash:
                            if not verify_md5_hash(fdst.name, expected_hash):
                                raise DownloadError(
                                    f"MD5 hash verification failed for product "
                                    f"'{product._id}'. The download may be corrupted.",
                                )
                        else:
                            log.debug(
                                f"MD5 verification requested but no hash available "
                                f"for product '{product._id}'"
                            )

                    shutil.unpack_archive(fdst.name, tmpdir, "zip")

                    for file in product_files:
                        save_path: str = f"{save_folder}/{file}"
                        fs.put(f"{tmpdir}/{file}", save_path)
                        if os.stat(f"{tmpdir}/{file}").st_size != fs.info(save_path).get("size", 0):
                            raise DownloadError(
                                f"Downloaded file size mismatch for '{save_path}'. "
                                f"Expected {os.stat(f'{tmpdir}/{file}').st_size}, "
                                f"got {fs.info(save_path).get('size', 0)}.",
                            )
                break
            except Exception as e:
                log.debug(
                    f"error downloading product '{product._id}' (attempt {i}/{retries}): '{e}'",
                )
                if i + 1 == retries:
                    raise DownloadError(
                        f"Failed to download output '{product._id}': '{e}'",
                    ) from e

    return expected_files


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
