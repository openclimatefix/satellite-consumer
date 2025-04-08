"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import TYPE_CHECKING

import eumdac
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
) -> Iterator[eumdac.product.Product]:
    """Get an iterator over the products for a given satellite in a given time range.

    Checks that the number of products returned matches the expected number of products.

    Args:
        sat_metadata: Metadata for the satellite to search for.
        start: Start time of the search.
        end: End time of the search.
        missing_product_threshold: Percentage of missing products allowed without error.

    Returns:
        Tuple of the iterator over the products and the total number of products found.
    """
    log.info(
        f"Searching for products between {start!s} and {end!s} "
        f"for {sat_metadata.product_id} ",
    )
    expected_products_count = int((end - start) / dt.timedelta(minutes=sat_metadata.cadence_mins))
    token = _gen_token()
    try:
        collection: Collection = eumdac.datastore.DataStore(token=token)\
            .get_collection(sat_metadata.product_id)
        search_results: SearchResults = collection.search(
            dtstart=start, dtend=end,
            sort="start,time,1", # Sort by ascending start time
        )
    except Exception as e:
        raise DownloadError(
            f"Error searching for products for '{sat_metadata.product_id}': '{e}'",
        ) from e
    if search_results.total_results == 0:
        raise DownloadError(
            f"No products found for {sat_metadata.product_id} in the given time range '{start!s}-{end!s}.",
        )
    if (1 - search_results.total_results/expected_products_count) > missing_product_threshold:
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

def download_nat(
    product: eumdac.product.Product,
    folder: str,
    retries: int = 6,
) -> str | None:
    """Download a product to an S3 bucket.

    EUMDAC products are collections of files, with a `.nat` file containing the data,
    and with `.xml` files containing metadata. This function only downloads the `.nat` files,
    skipping any files that are already present in the folder.

    Args:
        product: Product to download.
        folder: Folder to download the product to. Can be local path or S3 URL.
        retries: Number of times to retry downloading the product.

    Returns:
        Path to the downloaded file, or None if the download failed.
    """
    fs = get_fs(path=folder)

    nat_files: list[str] = [p for p in product.entries if p.endswith(".nat")]
    if len(nat_files) != 1:
        raise DownloadError(
            f"Couldn't download product '{product!s}' as it contains more "
            f"than one '.nat' file: '{nat_files}'. New functionality is needed ",
            "to determine how to act in this case.",
        )
    nat_filename: str = nat_files[0]

    if product.qualityStatus != "NOMINAL":
        log.warning(
            f"Encountered product '{product!s}' with non-nominal quality status "
            f"'{product.qualityStatus}'. ",
            quality=product.qualityStatus,
        )
        return None

    filepath: str = f"{folder}/{nat_filename}"
    try:
        if fs.exists(filepath):
            log.debug("Skipping already downloaded file", filename=nat_filename)
            return filepath
    except Exception as e:
        raise OSError(
            f"Could not determine if file '{filepath}' exists: '{e}'"
            "Ensure you have the required access permissions.",
        ) from e

    log.debug("Downloading raw file", src=nat_filename, dst=filepath)
    for i in range(retries):
        try:
            # Copying to temp then putting seems to be quicker than copying to fs
            with (product.open(nat_filename) as fsrc,
                  tempfile.NamedTemporaryFile(suffix=".nat") as fdst):
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024)
                fs.put(fdst.name, filepath)
            return filepath
        except Exception as e:
            log.warning(
                f"Error downloading product '{product}' (attempt {i}/{retries}): '{e}'",
            )

    raise DownloadError(f"Failed to download product '{product}' after {retries} attempts.")


def _gen_token() -> eumdac.token.AccessToken:
    """Generated an aces token from environment variables."""
    for var in ["EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET"]:
        if var not in os.environ:
            raise DownloadError(
                "Cannot download data from EUMETSET due to missing "
                f"required authorization environment variable '{var}'",
            )
    consumer_key: str = os.environ["EUMETSAT_CONSUMER_KEY"]
    consumer_secret: str = os.environ["EUMETSAT_CONSUMER_SECRET"]
    token = eumdac.token.AccessToken(credentials=(consumer_key, consumer_secret))

    return token

