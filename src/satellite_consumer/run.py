"""Pipeline for downloading, processing, and saving archival satellite data.

Consolidates the old cli_downloader, backfill_hrv and backfill_nonhrv scripts.
"""

import datetime as dt
from importlib.metadata import PackageNotFoundError, version

import eumdac.product
import icechunk
from joblib import Parallel, delayed
from loguru import logger as log

from satellite_consumer.config import (
    ConsumeCommandOptions,
    MergeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_raw,
    get_products_iterator,
)
from satellite_consumer.exceptions import ValidationError
from satellite_consumer.process import process_raw
from satellite_consumer import storage
from satellite_consumer.validate import validate

from icechunk.xarray import to_icechunk

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

def _consume_to_store(command_opts: ConsumeCommandOptions) -> None:
    """Logic for the consume command (and the archive command)."""
    window = command_opts.time_window
    product_iter = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    processed_filepaths: list[str] = []
    num_skipped: int = 0

    if command_opts.icechunk:
        repo, was_created = storage.get_icechunk_repo(path=command_opts.zarr_path)
        session: icechunk.Session = repo.writable_session(branch="main")
        # Download products in parallel
        # raw_filegroups: list[list[str]] = Parallel(
        #     n_jobs=command_opts.num_workers,
        #     return_as="list",
        #     prefer="threads",
        # )(delayed(download_raw)(
        #     product,
        #     folder=f"{command_opts.workdir}/raw",
        #     filter_regex=command_opts.satellite_metadata.file_filter_regex,
        # ) for product in product_iter)

        raw_filegroups = [
            download_raw(
                product=p,
                folder=f"{command_opts.workdir}/raw",
                filter_regex=command_opts.satellite_metadata.file_filter_regex,
            ) for p in product_iter
        ]
        for i, raw_filepaths in enumerate(raw_filegroups):
            if len(raw_filepaths) == 0:
                num_skipped += 1
                continue
            da = process_raw(
                paths=raw_filepaths,
                channels=command_opts.satellite_metadata.channels,
                resolution_meters=command_opts.resolution,
                normalize=False,
            )
            validate(src=da)
            if (i == 0 and was_created):
                to_icechunk(da.to_dataset(promote_attrs=True), session=session)
            else:
                to_icechunk(da.to_dataset(promote_attrs=True), session=session, append_dim="time")
            commit_id: str = session.commit(message="append data")
            log.debug(
                "Committed data to icechunk store",
                dst=command_opts.zarr_path, commit_id=commit_id,
            )
            processed_filepaths.extend(raw_filepaths)

        log.info(
            "Finished population of icechunk store",
            dst=command_opts.zarr_path, num_skipped=num_skipped,
        )

    else:
        fs = storage.get_fs(path=command_opts.zarr_path)
        # Use existing zarr store if it exists
        if fs.exists(command_opts.zarr_path.replace("s3://", "")):
            log.info("Using existing store", dst=command_opts.zarr_path)
        else:
            # Create new store
            log.info("Creating new zarr store", dst=command_opts.zarr_path)
            _ = storage.create_empty_zarr(
                dst=command_opts.zarr_path,
                coords=command_opts.as_coordinates(),
            )

        def _etl(product: eumdac.product.Product) -> list[str]:
            """Download, process, and save a single NAT file."""
            raw_filepaths = download_raw(
                product,
                folder=f"{command_opts.workdir}/raw",
                filter_regex=command_opts.satellite_metadata.file_filter_regex,
            )
            if len(raw_filepaths) == 0:
                return []
            da = process_raw(
                paths=raw_filepaths,
                channels=command_opts.satellite_metadata.channels,
                resolution_meters=command_opts.resolution,
                normalize=command_opts.rescale,
            )
            storage.write_to_zarr(da=da, dst=command_opts.zarr_path)
            return raw_filepaths

        # Iterate through all products in search
        for raw_filepaths in Parallel(
            n_jobs=command_opts.num_workers, return_as="generator",
            prefer="threads",
        )(delayed(_etl)(product) for product in product_iter):
            if len(raw_filepaths) == 0:
                num_skipped += 1
            else:
                processed_filepaths.extend(raw_filepaths)

        # Might not need this as skipped files are all NaN
        # and the validation step should catch it
        if num_skipped / (num_skipped + len(raw_filepaths)) > 0.05:
            raise ValidationError(
                f"Too many files had to be skipped "
                f"({num_skipped}/{num_skipped + len(raw_filepaths)}). "
                "Use dataset at your own risk!",
            )

        log.info(
            "Finished population of zarr store",
            dst=command_opts.zarr_path, num_skipped=num_skipped,
        )

        if command_opts.validate:
            validate(src=command_opts.zarr_path)

    if command_opts.delete_raw:
        if command_opts.workdir.startswith("s3://"):
            log.warning("delete-raw was specified, but deleting S3 files is not yet implemented")
        else:
            log.info(
                f"Deleting {len(raw_filepaths)} raw files in {command_opts.raw_folder}",
                num_files=len(raw_filepaths), dst=command_opts.raw_folder,
            )
            _ = [f.unlink() for f in raw_filepaths]

def _merge_command(command_opts: MergeCommandOptions) -> None:
    """Logic for the merge command."""
    zarr_paths = command_opts.zarr_paths
    log.info(
        f"Merging {len(zarr_paths)} stores",
        num=len(zarr_paths), consume_missing=command_opts.consume_missing,
    )
    fs = storage.get_fs(path=zarr_paths[0])

    for zarr_path in zarr_paths:
        if not fs.exists(zarr_path):
            if command_opts.consume_missing:
                _consume_to_store(command_opts=ConsumeCommandOptions(
                    time=dt.datetime.strptime(
                        zarr_path.split("/")[-1].split("_")[0], "%Y%m%dT%H%M",
                    ).replace(tzinfo=dt.UTC),
                    satellite=command_opts.satellite,
                    workdir=command_opts.workdir,
                    validate=True,
                    rescale=True, # TODO: Make this an option
                    resolution= command_opts.resolution,
                ))
            else:
                raise FileNotFoundError(f"Zarr store not found at {zarr_path}")

    dst = storage.create_latest_zip(srcs=zarr_paths)
    log.info("Created latest.zip", dst=dst)


def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}'",
        version=__version__, start_time=str(prog_start), opts=config.command_options.__str__(),
    )

    if isinstance(config.command_options, ConsumeCommandOptions):
        _consume_to_store(command_opts=config.command_options)
    elif isinstance(config.command_options, MergeCommandOptions):
        _merge_command(command_opts=config.command_options)
    else:
        pass

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")

