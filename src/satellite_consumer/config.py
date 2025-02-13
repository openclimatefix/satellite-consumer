"""Configuration for running the application."""

import dataclasses
import datetime as dt
from typing import Literal


@dataclasses.dataclass
class ArchiveCommandOptions:
    """Options for the archive command."""

    satellite: Literal["seviri", "iodc", "odegree"]
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

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        try:
            month = dt.datetime.strptime(self.month, "%Y-%m").replace(tzinfo=dt.UTC)
        except ValueError as e:
            raise ValueError("Invalid month format. Must be YYYY-MM.") from e
        if month > dt.datetime.now(tz=dt.UTC):
            raise ValueError("Month must be in the past.")


@dataclasses.dataclass
class DownloadCommandOptions:
    """Options for the download command."""

    satellite: Literal["seviri", "iodc", "odegree"]
    """The satellite to download data from."""
    time: dt.datetime | None = None
    """The time to download data for. Pulls 3.5 hours of data up to this time."""
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

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.time is not None:
            if self.time > dt.datetime.now(tz=dt.UTC):
                raise ValueError("Time must be in the past.")
        else:
            self.time = dt.datetime.now(tz=dt.UTC)


@dataclasses.dataclass
class SatelliteConsumerConfig:
    """Configuration for the satellite consumer."""

    command: Literal["archive", "download"]
    """The operational mode of the consumer."""
    command_options:  ArchiveCommandOptions | DownloadCommandOptions
    """Options for the chosen command."""

