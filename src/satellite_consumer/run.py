import datetime as dt
import sentry_sdk
from importlib.metadata import PackageNotFoundError, version
import eumdac.product
from joblib import Parallel, delayed
from loguru import logger as log

from satellite_consumer.config import (
    ArchiveCommandOptions,
    ConsumeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.download_eumetsat import (
    download_nat,
    get_products_iterator,
)
from satellite_consumer.process import process_nat
from satellite_consumer.storage import create_empty_zarr, create_latest_zip, get_fs, write_to_zarr
from satellite_consumer.validate import validate

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"

# Initialize Sentry


sentry_sdk.init(
    dsn="https://12927b5f211046b575ee51fd8b1ac34f@o1.ingest.sentry.io/1",  
    traces_sample_rate=1.0,  
    environment="production"
)


def _consume_command(command_opts: ArchiveCommandOptions | ConsumeCommandOptions) -> None:
    """Run the download and processing pipeline."""
    fs = get_fs(path=command_opts.zarr_path)
    window = command_opts.time_window
    
    sentry_sdk.set_tag("satellite", command_opts.satellite_metadata.name)
    sentry_sdk.set_tag("time_window", f"{window[0]} to {window[1]}")

    product_iter = get_products_iterator(
        sat_metadata=command_opts.satellite_metadata,
        start=window[0],
        end=window[1],
    )

    if fs.exists(command_opts.zarr_path.replace("s3://", "")):
        log.info("Using existing zarr store", dst=command_opts.zarr_path)
    else:
        log.info("Creating new zarr store", dst=command_opts.zarr_path)
        _ = create_empty_zarr(dst=command_opts.zarr_path, coords=command_opts.as_coordinates())

    def _etl(product: eumdac.product.Product) -> str:
        """Download, process, and save a single NAT file."""
        try:
            sentry_sdk.add_breadcrumb(message="Starting ETL for product", category="processing")
            nat_filepath = download_nat(product, folder=f"{command_opts.workdir}/raw")
            da = process_nat(path=nat_filepath, hrv=command_opts.hrv)
            write_to_zarr(da=da, dst=command_opts.zarr_path)
            return nat_filepath
        except Exception as e:
            sentry_sdk.capture_exception(e)
            log.error(f"Error processing product: {e}")
            return ""

    nat_filepaths: list[str] = []
    for nat_filepath in Parallel(
        n_jobs=command_opts.num_workers, return_as="generator",
    )(delayed(_etl)(product) for product in product_iter):
        if nat_filepath:
            nat_filepaths.append(nat_filepath)

    log.info("Finished population of zarr store", dst=command_opts.zarr_path)

    if command_opts.validate:
        try:
            validate(dataset_path=command_opts.zarr_path)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            log.error(f"Validation failed: {e}")

    if isinstance(command_opts, ConsumeCommandOptions) and command_opts.latest_zip:
        try:
            zippath: str = create_latest_zip(dst=command_opts.zarr_path)
            log.info(f"Created latest.zip at {zippath}", dst=zippath)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            log.error(f"Failed to create latest.zip: {e}")

    if command_opts.delete_raw:
        try:
            if command_opts.workdir.startswith("s3://"):
                log.warning("delete-raw was specified, but deleting S3 files is not yet implemented")
            else:
                log.info(
                    f"Deleting {len(nat_filepaths)} raw files in {command_opts.raw_folder}",
                    num_files=len(nat_filepaths), dst=command_opts.raw_folder,
                )
                _ = [f.unlink() for f in nat_filepaths]  # type:ignore
        except Exception as e:
            sentry_sdk.capture_exception(e)
            log.error(f"Failed to delete raw files: {e}")

def run(config: SatelliteConsumerConfig) -> None:
    """Run the download and processing pipeline."""
    prog_start = dt.datetime.now(tz=dt.UTC)

    log.info(
        f"Starting satellite consumer with command '{config.command}'",
        version=__version__, start_time=str(prog_start), opts=config.command_options.__str__(),
    )

    try:
        if config.command == "archive" or config.command == "consume":
            _consume_command(command_opts=config.command_options)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        log.error(f"Pipeline execution failed: {e}")

    runtime = dt.datetime.now(tz=dt.UTC) - prog_start
    log.info(f"Completed satellite consumer run in {runtime!s}.")

