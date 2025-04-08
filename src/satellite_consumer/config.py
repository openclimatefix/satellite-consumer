"""Configuration for running the application."""

import dataclasses
import datetime as dt
from enum import StrEnum, auto

import numpy as np
import pandas as pd
from loguru import logger as log


@dataclasses.dataclass
class Coordinates:
    """Coordinates describing the shape of a satellite dataset.

    The order of the fields determines their order when mapped to a dict.
    """

    time: list[np.datetime64]
    y_geostationary: list[float]
    x_geostationary: list[float]
    variable: list[str]

    def to_dict(self) -> dict[str, list[float] | list[str] | list[np.datetime64]]:
        """Convert the coordinates to a dictionary."""
        return dataclasses.asdict(self)

    def shape(self) -> tuple[int, ...]:
        """Get the shape of the dataset."""
        return tuple([len(v) for v in self.to_dict().values()])

    def dims(self) -> list[str]:
        """Get the dimensions of the dataset."""
        return list(self.to_dict().keys())

    def shards(self) -> tuple[int, ...]:
        """Get the shard size for each dimension.

        In order for the validate function to work, the shard size must be
        1 along the dimensions that are not core input dimensions;
        'time' and 'variable'.
        """
        return tuple([
            1 if k in ["time", "variable"]
            else len(v)
            for k, v in self.to_dict().items()
        ])

    def chunks(self) -> tuple[int, ...]:
        """Get the chunk size for each dimension."""
        return tuple([
            1 if k in ["time", "variable"]
            else 116 if k in ["x_geostationary", "y_geostationary"]
            else len(v)
            for k, v in self.to_dict().items()
        ])

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if len(self.time) == 0:
            raise ValueError("Time coordinate must have at least one value.")
        if len(self.x_geostationary) == 0:
            raise ValueError("X coordinate must have at least one value.")
        if len(self.y_geostationary) == 0:
            raise ValueError("Y coordinate must have at least one value.")
        if len(self.variable) == 0:
            raise ValueError("Variable coordinate must have at least one value.")
        self.x_geostationary = sorted(self.x_geostationary, reverse=True)
        self.y_geostationary = sorted(self.y_geostationary)
        self.variable = sorted(self.variable)

class Command(StrEnum):
    """The available commands for the satellite consumer."""

    ARCHIVE = auto()
    """Download and process all scans into an archive store for a given month.

    Archive stores are a single Zarr store containing data for all scans
    in the given time period.
    """
    CONSUME = auto()
    """Download and process a single scan into a scan store.

    Scan stores are a single Zarr store containing data for an individual scan.
    """
    MERGE = auto()
    """Merge multiple consumed stores into a single zarr store."""

@dataclasses.dataclass
class ArchiveCommandOptions:
    """Options for the archive command."""

    satellite: str
    """The satellite to create an archive dataset for."""
    month: str
    """The month to create an archive dataset for (YYYY-MM)."""
    delete_raw: bool = False
    """Whether to delete the raw data after creating the archive dataset."""
    validate: bool = False
    """Whether to validate the archive dataset after creating it."""
    hrv: bool = False
    """Whether to pull the high resolution channel data instead of the low res."""
    rescale: bool = False
    """Whether to rescale the data."""
    workdir: str = "/mnt/disks/sat"
    """The parent folder to store downloaded and processed data in.

    Can be either a local path or an S3 path (s3://bucket-name/path).
    """
    num_workers: int = 1
    """The number of workers to use for downloading and processing the data."""

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.satellite not in SATELLITE_METADATA:
            raise ValueError(
                f"Invalid satellite '{self.satellite}'. Must be one of {SATELLITE_METADATA.keys()}",
            )
        try:
            month = dt.datetime.strptime(self.month, "%Y-%m").replace(tzinfo=dt.UTC)
        except ValueError as e:
            raise ValueError("Invalid month format. Must be YYYY-MM.") from e
        if month > dt.datetime.now(tz=dt.UTC):
            raise ValueError("Month must be in the past.")

    @property
    def satellite_metadata(self) -> "SatelliteMetadata":
        """Get the metadata for the chosen satellite."""
        return SATELLITE_METADATA[self.satellite]

    @property
    def time_window(self) -> tuple[dt.datetime, dt.datetime]:
        """Get the time window for the given month."""
        start: dt.datetime = dt.datetime.strptime(self.month, "%Y-%m").replace(tzinfo=dt.UTC)
        end: dt.datetime = (start + pd.DateOffset(months=1, minutes=-1))
        return start, end

    @property
    def zarr_path(self) -> str:
        """Get the path to the zarr store for the given month."""
        resstr: str = "hrv" if self.hrv else "nonhrv"
        satstr: str = "" if self.satellite == "rss" else self.satellite
        return f"{self.workdir}/data/{self.month}_{resstr}_{satstr}.zarr"

    @property
    def raw_folder(self) -> str:
        """Get the path to the raw data folder for the given time."""
        return f"{self.workdir}/raw"

    def as_coordinates(self) -> Coordinates:
        """Return the coordinates of the data associated with the command options."""
        start, end = self.time_window
        return Coordinates(
            time=[
                ts.to_numpy() for ts in pd.date_range(
                start=start, end=end, freq=f"{self.satellite_metadata.cadence_mins}min",
            )], # TODO: Determine inclusive bounds
            y_geostationary=self.satellite_metadata.spatial_coordinates["y_geostationary"],
            x_geostationary=self.satellite_metadata.spatial_coordinates["x_geostationary"],
            variable=[ch.name for ch in SEVIRI_CHANNELS if ch.is_high_res == self.hrv],
        )

@dataclasses.dataclass
class ConsumeCommandOptions:
    """Options for the consume command."""

    satellite: str
    """The satellite to consume data from."""
    time: dt.datetime | None = None
    """The time to download data for."""
    delete_raw: bool = False
    """Whether to delete the raw data after downloading it."""
    validate: bool = False
    """Whether to validate the downloaded data after downloading it."""
    hrv: bool = False
    """Whether to pull the high resolution channel data (defaults to low res)."""
    rescale: bool = False
    """Whether to rescale the data."""
    workdir: str = "/mnt/disks/sat"
    """The parent folder to store downloaded and processed data in.

    Can be either a local path or an S3 path (s3://bucket-name/path).
    """
    num_workers: int = 1
    """The number of workers to use for downloading and processing the data."""

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.satellite not in SATELLITE_METADATA:
            raise ValueError(
                f"Invalid satellite '{self.satellite}'. Must be one of {SATELLITE_METADATA.keys()}",
            )
        if self.time is not None:
            if self.time.replace(tzinfo=dt.UTC) > dt.datetime.now(tz=dt.UTC):
                raise ValueError(f"Invalid time '{self.time}'. Must be in the past.")
        else:
            self.time = dt.datetime.now(tz=dt.UTC)

        if self.time.minute % self.satellite_metadata.cadence_mins != 0 or self.time.second != 0:
            newtime: dt.datetime = (self.time - dt.timedelta(
                minutes=self.time.minute % self.satellite_metadata.cadence_mins,
            )).replace(second=0, microsecond=0)
            log.debug(
                "Input time is not a multiple of the chosen satellite's image cadence. " + \
                "Adjusting to nearest image time.",
                input_time=str(self.time),
                adjusted_time=str(newtime),
            )
            self.time = newtime

    @property
    def satellite_metadata(self) -> "SatelliteMetadata":
        """Get the metadata for the chosen satellite."""
        return SATELLITE_METADATA[self.satellite]

    @property
    def time_window(self) -> tuple[dt.datetime, dt.datetime]:
        """Get the time window for the given time."""
        # Round the start time down to the nearest interval given by the channel cadence
        start: dt.datetime = (
            self.time - pd.DateOffset(minutes=self.satellite_metadata.cadence_mins) # type: ignore
        )
        return start, self.time # type:ignore  # safe due to post_init

    @property
    def zarr_path(self) -> str:
        """Get the path to the zarr store for the given time."""
        resstr: str = "hrv" if self.hrv else "nonhrv"
        return f"{self.workdir}/data/{self.time.strftime('%Y%m%dT%H%M')}_{resstr}.zarr" # type:ignore

    @property
    def raw_folder(self) -> str:
        """Get the path to the raw data folder for the given time."""
        return f"{self.workdir}/raw"

    def as_coordinates(self) -> Coordinates:
        """Return the coordinates of the data associated with the command options."""
        start, end = self.time_window
        return Coordinates(
            time=[ts.to_numpy() for ts in pd.date_range(
                inclusive="right", start=start, end=end,
                freq=f"{self.satellite_metadata.cadence_mins}min",
            )],
            y_geostationary=self.satellite_metadata.spatial_coordinates["y_geostationary"],
            x_geostationary=self.satellite_metadata.spatial_coordinates["x_geostationary"],
            variable=[ch.name for ch in SEVIRI_CHANNELS if ch.is_high_res == self.hrv],
        )

@dataclasses.dataclass
class MergeCommandOptions:
    """Options for the merge command."""

    satellite: str
    """The satellite to merge data for."""
    window_mins: int = 210
    """The time window of consumed data to merge."""
    window_end: dt.datetime | None = None
    """The end time of the time window to merge data for."""
    workdir: str = "/mnt/disks/sat"
    """The parent folder to store downloaded and processed data in.

    Can be either a local path or an S3 path (s3://bucket-name/path).
    """
    hrv: bool = False
    """Whether to merge the high resolution channel data (defaults to low res)."""
    consume_missing: bool = False
    """Whether to consume missing data instead of erroring before merging."""

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.satellite not in SATELLITE_METADATA:
            raise ValueError(
                f"Invalid satellite '{self.satellite}'. Must be one of {SATELLITE_METADATA.keys()}",
            )
        if self.window_mins <= 0:
            raise ValueError("Window size must be positive.")
        if self.window_end is not None:
            if self.window_end.replace(tzinfo=dt.UTC) > dt.datetime.now(tz=dt.UTC):
                raise ValueError("Window end must be in the past.")
        else:
            self.window_end = dt.datetime.now(tz=dt.UTC)

        if self.window_end.minute % self.satellite_metadata.cadence_mins != 0 or self.window_end.second != 0:
            newtime: dt.datetime = (self.window_end - dt.timedelta(
                minutes=self.window_end.minute % self.satellite_metadata.cadence_mins,
            )).replace(second=0, microsecond=0)
            log.debug(
                "Input window end is not a multiple of the chosen satellite's image cadence. " + \
                "Adjusting to nearest image time.",
                input_window_end=str(self.window_end),
                adjusted_time=str(newtime),
            )
            self.window_end = newtime

    @property
    def satellite_metadata(self) -> "SatelliteMetadata":
        """Get the metadata for the chosen satellite."""
        return SATELLITE_METADATA[self.satellite]

    @property
    def time_window(self) -> tuple[dt.datetime, dt.datetime]:
        """Get the time window for the given window end and size."""
        end: dt.datetime = self.window_end # type: ignore
        start: dt.datetime = end - pd.DateOffset(minutes=self.window_mins)
        return start, end

    @property
    def zarr_paths(self) -> list[str]:
        """Get the path to the zarr stores for the given time window."""
        resstr: str = "hrv" if self.hrv else "nonhrv"
        return [
            f"{self.workdir}/data/{ts.strftime('%Y%m%dT%H%M')}_{resstr}.zarr"
            for ts in pd.date_range(
                start=self.time_window[0], end=self.time_window[1],
                freq=f"{SATELLITE_METADATA[self.satellite].cadence_mins}min",
                inclusive="right",
            )
        ]

    @property
    def merged_path(self) -> str:
        """Get the path to the merged zarr store for the given time window."""
        return f"{self.workdir}/data/latest.zarr.zip"


@dataclasses.dataclass
class SatelliteConsumerConfig:
    """Configuration for the satellite consumer."""

    command: Command
    """The operational mode of the consumer."""
    command_options:  ArchiveCommandOptions | ConsumeCommandOptions | MergeCommandOptions
    """Options for the chosen command."""


@dataclasses.dataclass
class SatelliteMetadata:
    """Metadata for a satellite's data set.

    Note that the same physical satellite may provide multiple different datasets,
    each with their own SatelliteMetadata object.
    """

    region: str
    """The region the satellite images cover."""
    cadence_mins: int
    """The cadence in minutes at which the satellite images are taken."""
    longitude: float
    """The longitude of the satellite."""
    product_id: str
    """The product ID of the satellite image set."""
    description: str
    """A description of the satellite data set."""
    spatial_coordinates: dict[str, list[float]]
    """The spatial coordinates of the satellite data set."""

SATELLITE_METADATA: dict[str, SatelliteMetadata] = {
    "rss": SatelliteMetadata(
        region="europe",
        cadence_mins=5,
        longitude=9.5,
        product_id="EO:EUM:DAT:MSG:MSG15-RSS",
        description="".join((
            "Rectified (level 1.5) Meteosat SEVIRI Rapid Scan image data for Europe. ",
            "The image region is the top 1/3 of the earth's disk, "
            "covering a latitude range from approximately 15 degrees to 70 degrees. ",
            "The data is transmitted as High Rate transmissions in 12 spectral channels "
            "(11 low and one high resolution). ",
            "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:MSG15-RSS",
        )),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(1395187.45153809, 5568748.13049316, 1392)),
        },

    ),
    "iodc": SatelliteMetadata(
        region="india",
        cadence_mins=15,
        longitude=45.5,
        product_id="EO:EUM:DAT:MSG:HRSEVIRI-IODC",
        description="".join((
            "Rectified (level 1.5) Meteosat SEVIRI image data for the Indian Ocean. ",
            "The data is transmitted as High Rate transmissions in 12 spectral channels "
            "(11 low and one high resolution). ",
            "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:HRSEVIRI-IODC",
        )),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(-5565747.79846191, 5568748.01721191, 3712)),
        },
    ),
    "odegree": SatelliteMetadata(
        region="europe, africa",
        cadence_mins=15,
        longitude=0.0,
        product_id="EO:EUM:DAT:MSG:HRSEVIRI",
        description="".join((
            "Rectified (level 1.5) Meteosat SEVIRI image data for Europe and Africa. ",
            "The data is transmitted as High Rate transmissions in 12 spectral channels ",
            "(11 low and one high resolution). ",
            "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:HRSEVIRI",
        )),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(-5565747.79846191, 5568748.01721191, 3712)),
        },
    ),
}
"""Metadata for the available satellite data sets."""

@dataclasses.dataclass
class SpectralChannelMetadata:
    """Metadata for a spectral channel."""

    name: str
    """The name of the channel."""
    minimum: float
    """The approximate minimum pixel value for the channel."""
    maximum: float
    """The approximate maximum pixel value for the channel."""
    is_high_res: bool = False
    """Whether the channel is high resolution."""

    @property
    def range(self) -> float:
        """The approximate range of pixel values for the channel."""
        return self.maximum - self.minimum


SEVIRI_CHANNELS: list[SpectralChannelMetadata] = [
    SpectralChannelMetadata("IR_016", -2.5118103, 69.60857),
    SpectralChannelMetadata("IR_039", -64.83977, 339.15588),
    SpectralChannelMetadata("IR_087", 63.404694, 340.26526),
    SpectralChannelMetadata("IR_097", 2.844452, 317.86752),
    SpectralChannelMetadata("IR_108", 199.10002, 313.2767),
    SpectralChannelMetadata("IR_120", -17.254883, 315.99194),
    SpectralChannelMetadata("IR_134", -26.29155, 274.82297),
    SpectralChannelMetadata("VIS006", -1.1009827, 93.786545),
    SpectralChannelMetadata("VIS008", -2.4184198, 101.34922),
    SpectralChannelMetadata("WV_062", 199.57048, 249.91806),
    SpectralChannelMetadata("WV_073", 198.95093, 286.96323),
    SpectralChannelMetadata("HRV", -1.2278595, 103.90016, True),
]
"""Metadata for the available spectral channels from SEVIRI satellites.

The minimum and maximum values are approximate, based on calculated values
from a snapshot of the data.
"""

