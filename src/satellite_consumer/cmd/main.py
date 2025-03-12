"""Entrypoint to the satellite consumer."""

import argparse
import datetime as dt
import os
import sys
import traceback

from loguru import logger as log

from satellite_consumer.config import (
    SATELLITE_METADATA,
    ArchiveCommandOptions,
    Command,
    ConsumeCommandOptions,
    SatelliteConsumerConfig,
    MergeCommandOptions,
)
from satellite_consumer.run import run


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
    archive_parser.add_argument("--num-workers", type=int, default=1)
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
    consume_parser.add_argument("--num-workers", type=int, default=1)
    consume_parser.add_argument("--eumetsat-key", type=str, required=True)
    consume_parser.add_argument("--eumetsat-secret", type=str, required=True)

    merge_parser = subparsers.add_parser("merge",
        help="Merge satellite data for a given window",
    )
    merge_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    merge_parser.add_argument("--window-mins",
        type=int, help="Merge window size in minutes", default=210,
    )
    merge_parser.add_argument("--window-end", "-t",
        type=dt.datetime.fromisoformat,
        help="End of merge window (YYYY-MM-DDTHH:MM:SS)",
    )
    merge_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")
    merge_parser.add_argument("--hrv", action="store_true")

    args = parser.parse_args()

    os.environ["EUMETSAT_CONSUMER_KEY"] = args.eumetsat_key
    os.environ["EUMETSAT_CONSUMER_SECRET"] = args.eumetsat_secret

    command_opts: ArchiveCommandOptions | ConsumeCommandOptions | MergeCommandOptions
    command = Command(args.command.upper())
    match command:
        case Command.ARCHIVE:
            command_opts = ArchiveCommandOptions(
                satellite=args.satellite,
                month=args.month,
                delete_raw=args.delete_raw,
                validate=args.validate,
                hrv=args.hrv,
                rescale=args.rescale,
                workdir=args.workdir,
            )
        case Command.CONSUME:
            command_opts = ConsumeCommandOptions(
                satellite=args.satellite,
                time=args.time,
                delete_raw=args.delete_raw,
                validate=args.validate,
                hrv=args.hrv,
                rescale=args.rescale,
                workdir=args.workdir,
            )
        case Command.MERGE:
            command_opts = MergeCommandOptions(
                satellite=args.satellite,
                window_mins=args.window_mins,
                window_end=args.window_end,
                hrv=args.hrv,
                workdir=args.workdir,
            )

    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=command, command_options=command_opts,
    )

    try:
        run(config)
        sys.exit(0)
    except Exception as e:
        tb: str = traceback.format_exc()
        log.error(f"Error: {e}", traceback=tb)
        sys.exit(1)

def env_entrypoint() -> None:
    """Handle the program using environment variables."""
    try:
        command = Command(os.environ["SATCONS_COMMAND"])
        command_opts: ArchiveCommandOptions | ConsumeCommandOptions | MergeCommandOptions
        match command:
            case Command.ARCHIVE:
                command_opts = ArchiveCommandOptions(
                    satellite=os.environ["SATCONS_SATELLITE"],
                    month=os.environ["SATCONS_MONTH"],
                    delete_raw=os.getenv("SATCONS_DELETE_RAW", "false") == "true",
                    validate=os.getenv("SATCONS_VALIDATE", "false") == "true",
                    hrv=os.getenv("SATCONS_HRV", "false") == "true",
                    rescale=os.getenv("SATCONS_RESCALE", "false") == "true",
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    num_workers=int(os.getenv("SATCONS_NUM_WORKERS", default="1")),
                )
            case Command.CONSUME:
                if os.getenv("SATCONS_TIME") is None:
                    t: dt.datetime | None = None
                else:
                    t = dt.datetime.fromisoformat(os.environ["SATCONS_TIME"])

                command_opts = ConsumeCommandOptions(
                    satellite=os.environ["SATCONS_SATELLITE"],
                    time=t,
                    delete_raw=os.getenv("SATCONS_DELETE_RAW", "false") == "true",
                    validate=os.getenv("SATCONS_VALIDATE", "false") == "true",
                    hrv=os.getenv("SATCONS_HRV", "false") == "true",
                    rescale=os.getenv("SATCONS_RESCALE", "false") == "true",
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    num_workers=int(os.getenv("SATCONS_NUM_WORKERS", default="1")),
                )
            case Command.MERGE:
                # Use SATCONS_TIME if SATCONS_WINDOW_END is not set
                if os.getenv("SATCONS_WINDOW_END") is None:
                    if os.getenv("SATCONS_TIME") is None:
                        window_end: dt.datetime | None = None
                    else:
                        window_end = dt.datetime.fromisoformat(os.environ["SATCONS_TIME"])
                else:
                    window_end = dt.datetime.fromisoformat(os.environ["SATCONS_TIME"])

                command_opts = MergeCommandOptions(
                    satellite=os.environ["SATCONS_SATELLITE"],
                    window_mins=int(os.getenv("SATCONS_WINDOW_MINS", default="210")),
                    window_end=window_end,
                    hrv=os.getenv("SATCONS_HRV", "false") == "true",
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                )

    except KeyError as e:
        log.error(f"Missing environment variable: {e}")
        return
    except Exception as e:
        raise e

    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=command, command_options=command_opts,
    )

    try:
        run(config)
        sys.exit(0)
    except Exception as e:
        tb: str = traceback.format_exc()
        log.error(f"Error: {e}", traceback=tb)
        sys.exit(1)


