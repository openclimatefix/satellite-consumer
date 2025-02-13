"""Entrypoint to the satellite consumer."""

import argparse
import datetime as dt
from loguru import logger as log

from satellite_consumer.config import (
    SATELLITE_METADATA,
    ArchiveCommandOptions,
    ConsumeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.run import run


@log.catch
def cli_entrypoint() -> None:
    """Handle the program using CLI arguments."""
    parser = argparse.ArgumentParser(description="Satellite consumer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive_parser = subparsers.add_parser("archive",
        help="Create monthly archive of satellite data",
    )
    archive_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    archive_parser.add_argument("month", type=str, help="Month to download (YYYY-MM)")
    archive_parser.add_argument("--delete-raw", action="store_true")
    archive_parser.add_argument("--validate", action="store_true")
    archive_parser.add_argument("--hrv", action="store_true")
    archive_parser.add_argument("--rescale", action="store_true")
    archive_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")

    consume_parser = subparsers.add_parser("consume",
        help="Consume satellite data for a given time",
    )
    consume_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    consume_parser.add_argument("--time", "-t",
            type=dt.datetime.fromisoformat, required=False,
            help="Time to consume (YYYY-MM-DDTHH:MM:SS)",
    )
    consume_parser.add_argument("--delete-raw", action="store_true")
    consume_parser.add_argument("--validate", action="store_true")
    consume_parser.add_argument("--hrv", action="store_true")
    consume_parser.add_argument("--rescale", action="store_true")
    consume_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")
    consume_parser.add_argument("--zip", action="store_true")

    args = parser.parse_args()

    command_opts: ArchiveCommandOptions | ConsumeCommandOptions
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
        command_opts = ConsumeCommandOptions(
            satellite=args.satellite,
            time=args.time,
            delete_raw=args.delete_raw,
            validate=args.validate,
            hrv=args.hrv,
            rescale=args.rescale,
            workdir=args.workdir,
            latest_zip=args.zip,
        )
    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=args.command, command_options=command_opts,
    )

    return run(config)





