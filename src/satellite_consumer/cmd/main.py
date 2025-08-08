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
    ExtractLatestCommandOptions,
    SatelliteConsumerConfig,
)
from satellite_consumer.run import run


def cli_entrypoint() -> None:
    """Handle the program using CLI arguments.

    Maps the provided CLI arguments to an appropriate Config object.
    """
    parser = argparse.ArgumentParser(description="Satellite consumer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consume_parser = subparsers.add_parser(
        "consume",
        help="Consume satellite data for a given time",
    )
    consume_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    consume_parser.add_argument(
        "--time",
        "-t",
        type=dt.datetime.fromisoformat,
        required=False,
        help="Time to consume (YYYY-MM-DDTHH:MM:SS)",
        default=dt.datetime.now(tz=dt.UTC),
    )
    window_ex_group = consume_parser.add_mutually_exclusive_group(required=False)
    window_ex_group.add_argument(
        "--window-mins",
        type=int,
        help="Window size in minutes",
        default=0,
    )
    window_ex_group.add_argument(
        "--window-months",
        type=int,
        help="Window size in months",
        default=0,
    )
    consume_parser.add_argument("--validate", action="store_true")
    consume_parser.add_argument("--resolution", type=int, default=3000)
    consume_parser.add_argument("--rescale", action="store_true")
    consume_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")
    consume_parser.add_argument("--num-workers", type=int, default=1)
    consume_parser.add_argument("--eumetsat-key", type=str, required=True)
    consume_parser.add_argument("--eumetsat-secret", type=str, required=True)
    consume_parser.add_argument("--icechunk", action="store_true")
    consume_parser.add_argument("--crop-region", type=str, default="")

    extractlatest_parser = subparsers.add_parser(
        "extractlatest",
        help="Extract the latest windown of data from an existing scan store.",
    )
    extractlatest_parser.add_argument("satellite", choices=list(SATELLITE_METADATA.keys()))
    extractlatest_parser.add_argument(
        "--window-mins",
        type=int,
        help="Merge window size in minutes",
        default=210,
    )
    extractlatest_parser.add_argument("--workdir", type=str, default="/mnt/disks/sat")
    extractlatest_parser.add_argument("--resolution", type=int, default=3000)
    extractlatest_parser.add_argument("--crop-region", type=str, default="")

    args = parser.parse_args()

    os.environ["EUMETSAT_CONSUMER_KEY"] = args.eumetsat_key
    os.environ["EUMETSAT_CONSUMER_SECRET"] = args.eumetsat_secret

    command_opts: ConsumeCommandOptions | ExtractLatestCommandOptions
    command = Command(args.command.lower())
    match command:
        case Command.CONSUME:
            command_opts = ConsumeCommandOptions(
                satellite=args.satellite,
                time=args.time,
                window_mins=args.window_mins,
                window_months=args.window_months,
                validate=args.validate,
                resolution=args.resolution,
                rescale=args.rescale,
                workdir=args.workdir,
                icechunk=args.icechunk,
                crop_region=args.crop_region.lower(),
            )
        case Command.EXTRACTLATEST:
            command_opts = ExtractLatestCommandOptions(
                satellite=args.satellite,
                window_mins=args.window_mins,
                workdir=args.workdir,
                resolution=args.resolution,
            )

    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=command,
        command_options=command_opts,
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
        command_opts: ConsumeCommandOptions | ExtractLatestCommandOptions
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
                    validate=os.getenv("SATCONS_VALIDATE", "false").lower() == "true",
                    resolution=int(os.getenv("SATCONS_RESOLUTION", default="3000")),
                    rescale=os.getenv("SATCONS_RESCALE", "false").lower() == "true",
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    num_workers=int(os.getenv("SATCONS_NUM_WORKERS", default="1")),
                    icechunk=os.getenv("SATCONS_ICECHUNK", "false").lower() == "true",
                    crop_region=os.getenv("SATCONS_CROP_REGION", "").lower(),
                )

            case Command.EXTRACTLATEST:
                command_opts = ExtractLatestCommandOptions(
                    satellite=os.environ["SATCONS_SATELLITE"],
                    window_mins=int(os.getenv("SATCONS_WINDOW_MINS", default="210")),
                    workdir=os.getenv("SATCONS_WORKDIR", "/mnt/disks/sat"),
                    resolution=int(os.getenv("SATCONS_RESOLUTION", default="3000")),
                    crop_region=os.getenv("SATCONS_CROP_REGION", "").lower(),
                )

    except KeyError as e:
        log.error(f"Missing environment variable: {e}")
        return
    except Exception as e:
        raise e

    config: SatelliteConsumerConfig = SatelliteConsumerConfig(
        command=command,
        command_options=command_opts,
    )

    try:
        run(config)
        sys.exit(0)
    except Exception as e:
        tb: str = traceback.format_exc()
        log.error(f"Error: {e}", traceback=tb)
        sys.exit(1)
