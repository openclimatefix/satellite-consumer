"""Entrypoint to the satellite consumer."""

import argparse
import datetime as dt
import os

from loguru import logger as log

from satellite_consumer.config import (
    SATELLITE_METADATA,
    ArchiveCommandOptions,
    Command,
    ConsumeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.run import run


@log.catch(onerror=lambda e: log.error(f"Error: {e}"))
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
    archive_parser.add_argument("--eumetsat-key", type=str, required=True)
    archive_parser.add_argument("--eumetsat-secret", type=str, required=True)

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
    archive_parser.add_argument("--eumetsat-key", type=str, required=True)
    archive_parser.add_argument("--eumetsat-secret", type=str, required=True)

    args = parser.parse_args()

    os.environ["EUMETSAT_CONSUMER_KEY"] = args.eumetsat_key
    os.environ["EUMETSAT_CONSUMER_SECRET"] = args.eumetsat_secret

    command_opts: ArchiveCommandOptions | ConsumeCommandOptions
    command = Command(args.command)
    if command == "archive":
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
        command=command, command_options=command_opts,
    )

    return run(config)

@log.catch(onerror=lambda e: log.error(f"Error: {e}"))
def env_entrypoint() -> None:
    """Handle the program using environment variables."""
    try:
        command = Command(os.environ["SATCONS_COMMAND"])
        command_opts: ArchiveCommandOptions | ConsumeCommandOptions
        if command == "archive":
            command_opts = ArchiveCommandOptions(
                satellite=os.environ["SATCONS_SATELLITE"],
                month=os.environ["SATCONS_MONTH"],
                delete_raw=os.environ.get("SATCONS_DELETE_RAW", "false") == "true",
                validate=os.environ.get("SATCONS_VALIDATE", "false") == "true",
                hrv=os.environ.get("SATCONS_HRV", "false") == "true",
                rescale=os.environ.get("SATCONS_RESCALE", "false") == "true",
                workdir=os.environ.get("SATCONS_WORKDIR", "/mnt/disks/sat"),
            )
        else:
            if os.environ.get("SATCONS_TIME") is None:
                t: dt.datetime | None = None
            else:
                t = dt.datetime.fromisoformat(os.environ["SATCONS_TIME"])
            command_opts = ConsumeCommandOptions(
                satellite=os.environ["SATCONS_SATELLITE"],
                time=t,
                delete_raw=os.environ.get("SATCONS_DELETE_RAW", "false") == "true",
                validate=os.environ.get("SATCONS_VALIDATE", "false") == "true",
                hrv=os.environ.get("SATCONS_HRV", "false") == "true",
                rescale=os.environ.get("SATCONS_RESCALE", "false") == "true",
                workdir=os.environ.get("SATCONS_WORKDIR", "/mnt/disks/sat"),
                latest_zip=os.environ.get("SATCONS_ZIP", "false") == "true",
            )
    except KeyError as e:
        log.error(f"Missing environment variable: {e}")
        return
    except Exception as e:
        raise e

    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=command, command_options=command_opts,
    )
    return run(config)


