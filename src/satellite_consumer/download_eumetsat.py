"""Functions for interfacing with EUMETSAT's API and data."""

import datetime as dt
import os
import pathlib
import shutil
from collections.abc import Iterator
from typing import TYPE_CHECKING

import eumdac
from loguru import logger as log

from satellite_consumer.config import SatelliteMetadata

if TYPE_CHECKING:
    from eumdac.collection import Collection, SearchResults

def get_products_iterator(
    sat_metadata: SatelliteMetadata,
    start: dt.datetime,
    end: dt.datetime,
    missing_product_threshold: float = 0.1,
) -> tuple[Iterator[eumdac.product.Product], int]:
    """Get an iterator over the products for a given satellite in a given time range.

    Checks that the number of products returned matches the expected number of products.

    Args:
        sat_metadata: Metadata for the satellite to search for.
        start: Start time of the search.
        end: End time of the search.
        token: EUMETSAT access token.
        missing_product_threshold: Percentage of missing products allowed without error.

    Returns:
        Tuple of the iterator over the products and the total number of products found.
    """
    log.info(
        f"Searching for products between {start!s} and {end!s} "
        f"for {sat_metadata.product_id} ",
    )
    expected_products_count = int((end - start) / dt.timedelta(minutes=sat_metadata.cadence_mins))
    collection: Collection = eumdac.datastore.DataStore(token=_gen_token())\
        .get_collection(sat_metadata.product_id)
    search_results: SearchResults = collection.search(
        dtstart=start, dtend=end,
        sort="start,time,1", # Sort by ascending start time
    )
    if (1 - search_results.total_results/expected_products_count) > missing_product_threshold:
        raise ValueError(
            f"Threshold for missing products exceeded: "
            f"found {search_results.total_results}/{expected_products_count} products "
            f"for {sat_metadata.product_id} ",
        )
    log.info(
        f"Found {search_results.total_results}/{expected_products_count} products "
        f"for {sat_metadata.product_id} ",
    )
    return search_results.__iter__(), search_results.total_results

def download_nat(
    product: eumdac.product.Product,
    folder: pathlib.Path,
    retries: int = 6,
) -> pathlib.Path | None:
    """Download a product to a folder.

    EUMDAC products are collections of files, with a `.nat` file containing the data,
    and with `.xml` files containing metadata. This function only downloads the `.nat` files,
    skipping any files that are already present in the folder.

    Args:
        product: Product to download.
        folder: Folder to download the product to.
        retries: Number of times to retry downloading the product.

    Returns:
        Path to the downloaded file, or None if the download failed.
    """
    folder.mkdir(parents=True, exist_ok=True)
    nat_files: list[str] = [p for p in product.entries if p.endswith(".nat")]
    if len(nat_files) != 1:
        log.warning(
            f"Product '{product}' contains {len(nat_files)} .nat files. "
            "Expected 1. Skipping download.",
        )
        return None
    nat_filename: str = nat_files[0]

    filepath: pathlib.Path = folder / nat_filename
    if filepath.exists():
        log.debug(f"Skipping existing file: {filepath}")
        return filepath

    for i in range(retries):
        try:
            with (product.open(nat_filename) as fsrc, filepath.open("wb") as fdst):
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024)
            return filepath
        except Exception as e:
            log.warning(
                f"Error downloading product '{product}' (attempt {i}/{retries}): '{e}'",
            )

    log.error(f"Failed to download product '{product}' after {retries} attempts.")
    return None

def _gen_token() -> eumdac.token.AccessToken:
    """Generated an aces token from environment variables."""
    consumer_key: str = os.environ["EUMETSAT_CONSUMER_KEY"]
    consumer_secret: str = os.environ["EUMETSAT_CONSUMER_SECRET"]
    token = eumdac.token.AccessToken(credentials=(consumer_key, consumer_secret))

    return token

