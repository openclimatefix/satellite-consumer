"""Entrypoint to the satellite consumer."""

import argparse
from importlib.metadata import PackageNotFoundError, version

from loguru import logger as log

from satellite_consumer.config import (
    SATELLITE_METADATA,
    ArchiveCommandOptions,
    DownloadCommandOptions,
    SatelliteConsumerConfig,
)

try:
    __version__ = version("satellite-consumer")
except PackageNotFoundError:
    __version__ = "v?"



def run(config: SatelliteConsumerConfig) -> None:
    """Main entrypoint to the application."""
    log.info(f"Starting satellite consumer with command {config.command}", version=__version__)
    log.info(f"Config: {config}")
    if config.command == "archive":
        log.info("Getting time for window", window=config.command_options.get_time_window())
        log.info(type(config.command_options))

def cli_entrypoint() -> None:
    """Handle the program using CLI arguments."""
    parser = argparse.ArgumentParser(description="Satellite consumer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive_parser = subparsers.add_parser(
        "archive",
        help="Create monthly archive of satellite data",
    )
    archive_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    archive_parser.add_argument("month", type=str, help="Month to download (YYYY-MM)")
    archive_parser.add_argument("--delete-raw", action="store_true")
    archive_parser.add_argument("--validate", action="store_true")
    archive_parser.add_argument("--hrv", action="store_true")
    archive_parser.add_argument("--rescale", action="store_true")
    archive_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    download_parser.add_argument("--time", type=str, help="Time to download (YYYY-MM-DDTHH:MM:SS)")
    download_parser.add_argument("--delete-raw", action="store_true")
    download_parser.add_argument("--validate", action="store_true")
    download_parser.add_argument("--hrv", action="store_true")
    download_parser.add_argument("--rescale", action="store_true")
    download_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")

    args = parser.parse_args()

    command_opts: ArchiveCommandOptions | DownloadCommandOptions
    if args.command == "archive":
        command_opts = ArchiveCommandOptions(
            satellite=args.satellite,
            month=args.month,
            delete_raw=args.delete_raw,
            validate=args.validate,
            hrv=args.hrv,
            rescale=args.rescale,
            workdir=args.workdir,
        )
    else:
        command_opts = DownloadCommandOptions(
            satellite=args.satellite,
            time=args.time,
            delete_raw=args.delete_raw,
            validate=args.validate,
            hrv=args.hrv,
            rescale=args.rescale,
            workdir=args.workdir,
        )
    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=args.command, command_options=command_opts,
    )
    return run(config)





