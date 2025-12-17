"""Entrypoint to the satellite consumer."""

import datetime as dt
import importlib.metadata
import importlib.resources
import logging
import time

import pyhocon
import sentry_sdk

from satellite_consumer import consume, models

log = logging.getLogger(__name__)


def main() -> None:
    """Entrypoint to the consumer.

    Parse configuration from environment and run consumer acording to it.
    """
    conf_file = importlib.resources.files("satellite_consumer.cmd").joinpath("application.conf")
    conf: pyhocon.ConfigTree = pyhocon.ConfigFactory.parse_string(conf_file.read_text())

    if conf.get_string("sentry.dsn") != "":
        sentry_sdk.init(
            dsn=conf.get_string("sentry.dsn"),
            environment=conf.get_string("sentry.environment"),
            traces_sample_rate=1,
        )
        sentry_sdk.set_tag("app_name", "satellite_consumer")
        sentry_sdk.set_tag("app_version", importlib.metadata.version("satellite_consumer"))

    sat: str = conf.get_string("consumer.satellite")
    dt_range: tuple[dt.datetime, dt.datetime] = (
        dt.datetime.fromisoformat(conf.get_string("consumer.start_timestamp")),
        dt.datetime.fromisoformat(conf.get_string("consumer.end_timestamp")),
    )

    sensor: str = conf.get_string(f"satellites.{sat}.sensor")
    resolution: int = conf.get_int("consumer.resolution_meters")
    channels: list[models.SpectralChannel] = [
        models.SpectralChannel(
            name=ch,
            representation=conf.get_string(f"sensors.{sensor}.channels.{ch}.repr"),
            satpy_index=conf.get_int(f"sensors.{sensor}.channels.{ch}.satpy_idx"),
        )
        for ch in conf.get_config(f"sensors.{sensor}.channels")
        if resolution in conf.get_list(f"sensors.{sensor}.channels.{ch}.res")
    ]
    if len(channels) == 0:
        raise ValueError(
            f"No channels found for satellite {sat} at resolution {resolution!s} meters",
        )

    # Get the lon-lat crop bounds from the region string
    crop_region_lonlat: tuple[float, float, float, float] | None = None
    crop_region: str = conf.get_string("consumer.crop_region")
    if crop_region != "":
        crop_region_lonlat = (
            conf.get_float(f"regions.{crop_region}.left"),
            conf.get_float(f"regions.{crop_region}.bottom"),
            conf.get_float(f"regions.{crop_region}.right"),
            conf.get_float(f"regions.{crop_region}.top"),
        )

    prog_start = time.time()

    log.info(
        "sat consumer %s starting for %s from %s to %s",
        importlib.metadata.version("satellite_consumer"),
        sat,
        dt_range[0].isoformat(),
        dt_range[1].isoformat(),
    )

    consume.consume_to_store(
        dt_range=dt_range,
        cadence_mins=conf.get_int(f"satellites.{sat}.cadence_mins"),
        product_id=conf.get_string(f"satellites.{sat}.product_id"),
        filter_regex=conf.get_string(f"satellites.{sat}.file_filter_regex"),
        raw_zarr_paths=(
            conf.get_string("consumer.raw_path"),
            conf.get_string("consumer.zarr_path"),
        ),
        channels=channels,
        resolution_meters=resolution,
        crop_region_lonlat=crop_region_lonlat,
        eumetsat_credentials=(
            conf.get_string("credentials.eumetsat.key"),
            conf.get_string("credentials.eumetsat.secret"),
        ),
        use_icechunk=conf.get_bool("consumer.use_icechunk"),
        aws_credentials=(
            conf.get_string("credentials.aws.access_key_id", None),
            conf.get_string("credentials.aws.secret_access_key", None),
            conf.get_string("credentials.aws.endpoint_url", None),
            conf.get_string("credentials.aws.region", None),
        ),
        gcs_credentials=conf.get_string("credentials.gcs.application_credentials", None),
        dims_chunks_shards=(
            conf.get_list(f"satellites.{sat}.dimensions"),
            conf.get_list(f"satellites.{sat}.chunks"),
            conf.get_list(f"satellites.{sat}.shards"),
        ),
    )

    log.info(f"sat consumer finished in {time.time() - prog_start!s}")


if __name__ == "__main__":
    main()
