"""Entrypoint to the satellite consumer."""

import argparse
import datetime as dt
import os
import sys
import traceback

from loguru import logger as log

from satellite_consumer.config import (
    SATELLITE_METADATA,
    Command,
    ConsumeCommandOptions,
    MergeCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.run import run


def cli_entrypoint() -> None:
    """Handle the program using CLI arguments.

    Maps the provided CLI arguments to an appropriate Config object.
    """
    parser = argparse.ArgumentParser(description="Satellite consumer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consume_parser = subparsers.add_parser("consume",
        help="Consume satellite data for a given time",
    )
    consume_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    consume_parser.add_argument("--time", "-t",
            type=dt.datetime.fromisoformat, required=False,
            help="Time to consume (YYYY-MM-DDTHH:MM:SS)",
            default=dt.datetime.now(tz=dt.UTC),
    )
    window_ex_group = consume_parser.add_mutually_exclusive_group(required=False)
    window_ex_group.add_argument("--window-mins",
        type=int, help="Window size in minutes", default=0,
    )
    window_ex_group.add_argument("--window-months",
        type=int, help="Window size in months", default=0,
    )
    consume_parser.add_argument("--delete-raw", action="store_true")
    consume_parser.add_argument("--validate", action="store_true")
    consume_parser.add_argument("--resolution", type=int, default=3000)
    consume_parser.add_argument("--rescale", action="store_true")
    consume_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")
    consume_parser.add_argument("--num-workers", type=int, default=1)
    consume_parser.add_argument("--eumetsat-key", type=str, required=True)
    consume_parser.add_argument("--eumetsat-secret", type=str, required=True)
    consume_parser.add_argument("--icechunk", action="store_true")

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
    merge_parser.add_argument("--resolution", type=int, default=3000)
    merge_parser.add_argument("--consume-missing", action="store_true")

    args = parser.parse_args()

    os.environ["EUMETSAT_CONSUMER_KEY"] = args.eumetsat_key
    os.environ["EUMETSAT_CONSUMER_SECRET"] = args.eumetsat_secret

    command_opts: ConsumeCommandOptions | MergeCommandOptions
    command = Command(args.command.upper())
    match command:
        case Command.CONSUME:
            command_opts = ConsumeCommandOptions(
                satellite=args.satellite,
                time=args.time,
                window_mins=args.window_mins,
                window_months=args.window_months,
                delete_raw=args.delete_raw,
                validate=args.validate,
                resolution=args.resolution,
                rescale=args.rescale,
                workdir=args.workdir,
                icechunk=args.icechunk,
            )
        case Command.MERGE:
            command_opts = MergeCommandOptions(
                satellite=args.satellite,
                window_mins=args.window_mins,
                window_end=args.window_end,
                resolution=args.resolution,
                workdir=args.workdir,
                consume_missing=args.consume_missing,
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
    """Handle the program using environment variables.

    Maps the environemnt variables to an appropriate Config object.
    """
    try:
        command = Command(os.environ["SATCONS_COMMAND"])
        command_opts: ConsumeCommandOptions | MergeCommandOptions
        match command:
            case Command.CONSUME:
                if os.getenv("SATCONS_TIME") is None:
                    t: dt.datetime = dt.datetime.now(tz=dt.UTC)
                else:
                    t = dt.datetime.fromisoformat(os.environ["SATCONS_TIME"])

                command_opts = ConsumeCommandOptions(
                    satellite=os.environ["SATCONS_SATELLITE"],
                    time=t,
                    window_mins=int(os.getenv("SATCONS_WINDOW_MINS", default="0")),
                    window_months=int(os.getenv("SATCONS_WINDOW_MONTHS", default="0")),
                    delete_raw=os.getenv("SATCONS_DELETE_RAW", "false").lower() == "true",
                    validate=os.getenv("SATCONS_VALIDATE", "false").lower() == "true",
                    resolution=int(os.getenv("SATCONS_RESOLUTION", default="3000")),
                    rescale=os.getenv("SATCONS_RESCALE", "false").lower() == "true",
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    num_workers=int(os.getenv("SATCONS_NUM_WORKERS", default="1")),
                    icechunk=os.getenv("SATCONS_ICECHUNK", "false").lower() == "true",
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
                    resolution=int(os.getenv("SATCONS_RESOLUTION", default="3000")),
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    consume_missing=os.getenv("SATCONS_CONSUME_MISSING", "false").lower() == "true",
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


