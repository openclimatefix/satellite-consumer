"""Configuration for running the application."""

import dataclasses
import datetime as dt
from enum import StrEnum, auto

import numpy as np
import pandas as pd
from loguru import logger as log


@dataclasses.dataclass
class SpectralChannelMetadata:
    """Metadata for a spectral channel."""

    name: str
    """The name of the channel."""
    resolution_meters: list[int]
    """The available resolutions of the channel in meters."""
    minimum: float = 0.0
    """The approximate minimum pixel value for the channel."""
    maximum: float = 0.0
    """The approximate maximum pixel value for the channel."""

    @property
    def range(self) -> float:
        """The approximate range of pixel values for the channel."""
        return self.maximum - self.minimum


SEVIRI_CHANNELS: list[SpectralChannelMetadata] = [
    SpectralChannelMetadata("IR_016", [3000], -2.5118103, 69.60857),
    SpectralChannelMetadata("IR_039", [3000], -64.83977, 339.15588),
    SpectralChannelMetadata("IR_087", [3000], 63.404694, 340.26526),
    SpectralChannelMetadata("IR_097", [3000], 2.844452, 317.86752),
    SpectralChannelMetadata("IR_108", [3000], 199.10002, 313.2767),
    SpectralChannelMetadata("IR_120", [3000], -17.254883, 315.99194),
    SpectralChannelMetadata("IR_134", [3000], -26.29155, 274.82297),
    SpectralChannelMetadata("VIS006", [3000], -1.1009827, 93.786545),
    SpectralChannelMetadata("VIS008", [3000], -2.4184198, 101.34922),
    SpectralChannelMetadata("WV_062", [3000], 199.57048, 249.91806),
    SpectralChannelMetadata("WV_073", [3000], 198.95093, 286.96323),
    SpectralChannelMetadata("HRV", [1000], -1.2278595, 103.90016),
]
"""Metadata for the available spectral channels from SEVIRI satellites.

The minimum and maximum values are approximate, based on calculated values
from a snapshot of the data. The resolutions are derived from the data sheet at
https://user.eumetsat.int/s3/eup-strapi-media/pdf_ten_05105_msg_img_data_e7c8b315e6.pdf
"""

MTG_1KM_CHANNELS: list[SpectralChannelMetadata] = [
    SpectralChannelMetadata("vis_04", [1000]),
    SpectralChannelMetadata("vis_05", [1000]),
    SpectralChannelMetadata("vis_06", [1000, 500]),
    SpectralChannelMetadata("vis_08", [1000]),
    SpectralChannelMetadata("vis_09", [1000]),
    SpectralChannelMetadata("nir_13", [1000]),
    SpectralChannelMetadata("nir_16", [1000]),
    SpectralChannelMetadata("nir_22", [1000, 500]),
    SpectralChannelMetadata("wv_63", [2000]),
    SpectralChannelMetadata("wv_73", [2000]),
    SpectralChannelMetadata("ir_38", [2000, 1000]),
    SpectralChannelMetadata("ir_87", [2000]),
    SpectralChannelMetadata("ir_97", [2000]),
    SpectralChannelMetadata("ir_105", [2000, 1000]),
    SpectralChannelMetadata("ir_123", [2000]),
    SpectralChannelMetadata("ir_133", [2000]),
]
"""Metadata for the available spectral channels from MTG-1 satellites.

The minimum and maximum values are approximate, based on calculated values
from a snapshot of the data. The resolutions are derived from the data sheet at
https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide
"""


@dataclasses.dataclass
class Coordinates:
    """Coordinates describing the shape of a satellite dataset.

    The order of the fields determines their order when mapped to a dict.
    """

    time: list[np.datetime64]
    y_geostationary: list[float] | list[int]
    x_geostationary: list[float] | list[int]
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
        return tuple(
            [1 if k in ["time", "variable"] else len(v) for k, v in self.to_dict().items()],
        )

    def chunks(self) -> tuple[int, ...]:
        """Get the chunk size for each dimension."""

        def _get_factor_near(n: int, initial_divisor: int = 8) -> int:
            for i in range(initial_divisor, n, 1):
                if n % i == 0:
                    return i
            return 1

        return tuple(
            [
                1
                if k in ["time", "variable"]
                else len(v) // _get_factor_near(len(v), initial_divisor=8)
                if k in ["x_geostationary", "y_geostationary"]
                else len(v)
                for k, v in self.to_dict().items()
            ],
        )

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

    CONSUME = auto()
    """Download and process a single scan into a scan store.

    Scan stores are a single Zarr store containing data for all scana
    withing a given time window.
    """
    EXTRACTLATEST = auto()
    """Extract the latest window of data from a scan store."""


@dataclasses.dataclass
class ConsumeCommandOptions:
    """Options for the consume command."""

    satellite: str
    """The satellite to consume data from."""
    time: dt.datetime = dataclasses.field(default_factory=dt.datetime.now)
    """The time to download data for."""
    window_mins: int = 0
    """The time window to fetch data for (defaults to a single time)."""
    window_months: int = 0
    """The number of months to fetch data for (overrides window_mins)."""
    validate: bool = False
    """Whether to validate the downloaded data after downloading it."""
    resolution: int = 3000
    """The resolution in meters to use for the data."""
    rescale: bool = False
    """Whether to rescale the data."""
    workdir: str = "/mnt/disks/sat"
    """The parent folder to store downloaded and processed data in.

    Can be either a local path or an S3 path (s3://bucket-name/path).
    """
    num_workers: int = 1
    """The number of workers to use for downloading and processing the data."""
    icechunk: bool = False
    """Whether to use icechunk for storage."""
    crop_region: str = ""
    """The name of the region to crop to. An empty string means no cropping."""

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.satellite not in SATELLITE_METADATA:
            raise ValueError(
                f"Invalid satellite '{self.satellite}'. Must be one of {SATELLITE_METADATA.keys()}",
            )

        if self.time.replace(tzinfo=dt.UTC) > dt.datetime.now(tz=dt.UTC):
            raise ValueError(f"Invalid time '{self.time}'. Must be in the past.")

        if self.time.minute % self.satellite_metadata.cadence_mins != 0 or self.time.second != 0:
            newtime: dt.datetime = (
                self.time
                - dt.timedelta(
                    minutes=self.time.minute % self.satellite_metadata.cadence_mins,
                )
            ).replace(second=0, microsecond=0)
            log.debug(
                "Input time is not a multiple of the chosen satellite's image cadence. "
                + "Adjusting to nearest image time.",
                input_time=str(self.time),
                adjusted_time=str(newtime),
            )
            self.time = newtime

        if self.window_months > 0 and self.window_mins > 0:
            raise ValueError("Cannot specify both window_months and window_mins.")
        if self.window_months < 0 or self.window_mins < 0:
            raise ValueError("Window size must be positive.")

        if (
            len(
                [
                    c
                    for c in self.satellite_metadata.channels
                    if self.resolution in c.resolution_meters
                ],
            )
            == 0
        ):
            raise ValueError(
                f"No channels found for resolution {self.resolution} in the provided satellite.",
            )

    @property
    def satellite_metadata(self) -> "SatelliteMetadata":
        """Get the metadata for the chosen satellite."""
        return SATELLITE_METADATA[self.satellite]

    @property
    def time_window(self) -> tuple[dt.datetime, dt.datetime]:
        """Get the time window for the given time."""
        if self.window_months > 0:
            # Ignore the window_mins if window_months is set
            start: dt.datetime = self.time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end: dt.datetime = start + pd.DateOffset(months=self.window_months)
        else:
            start = self.time
            end = self.time + pd.DateOffset(
                minutes=self.window_mins + self.satellite_metadata.cadence_mins,
            )
        return start, end

    @property
    def zarr_path(self) -> str:
        """Get the path to the zarr store for the given time."""
        resstr: str = f"{self.resolution}m"
        satstr: str = self.satellite
        suffix: str = ".icechunk" if self.icechunk else ".zarr"
        cropstr: str = f"{self.crop_region}"
        match self.window_mins, self.window_months, self.icechunk:
            case 0, 0, False:
                windowstr: str = self.time.strftime("_%Y%m%dT%H%M")
            case _, 0, False:
                windowstr = f"_{self.time:%Y%m%dT%H%M}_window{self.window_mins}mins"
            case _, 1, False:
                windowstr = f"_{self.time:%Y%m}"
            case _, _, False:
                windowstr = f"_{self.time:%Y%m}_window{self.window_months}months"
            case _, _, True:
                windowstr = ""  # Append all times to the same icechunk store

        return f"{self.workdir}/{satstr}_{cropstr}{resstr}{windowstr}{suffix}"

    @property
    def raw_folder(self) -> str:
        """Get the path to the raw data folder for the given time."""
        return f"{self.workdir}/raw"

    def as_coordinates(self) -> Coordinates:
        """Return the coordinates of the data associated with the command options."""
        start, end = self.time_window
        bounds = self.crop_region_geos

        return Coordinates(
            # The bounds are "inclusive='right' here because for historical reasons
            # the image time is set to be the end time of the scan (rounded to the
            # satellite's cadence) rather than the start...
            # This means functionally that the time coordinates in the zarr store
            # do not line up with the time coordinates of the file names!
            time=[
                ts.to_numpy()
                for ts in pd.date_range(
                    inclusive="right",
                    start=start,
                    end=end,
                    freq=f"{self.satellite_metadata.cadence_mins}min",
                )
            ],
            y_geostationary=[
                y
                for y in self.satellite_metadata.spatial_coordinates["y_geostationary"]
                if (y >= bounds[1] and y <= bounds[3])
            ]
            if bounds is not None
            else self.satellite_metadata.spatial_coordinates["y_geostationary"],
            x_geostationary=[
                x
                for x in self.satellite_metadata.spatial_coordinates["x_geostationary"]
                if (x >= bounds[0] and x <= bounds[2])
            ]
            if bounds is not None
            else self.satellite_metadata.spatial_coordinates["x_geostationary"],
            variable=[
                ch.name
                for ch in self.satellite_metadata.channels
                if self.resolution in ch.resolution_meters
            ],
        )

    @property
    def crop_region_geos(self) -> tuple[float, float, float, float] | None:
        """Get the bounds of the crop region (if any) in geostationary coordinates.

        Returns a float tuple of (min_x, min_y, max_x, max_y) in geostationary coordinates.
        If no cropping is wanted, returns None.
        """
        crop_region_map: dict[str, dict[str, int]] = {
            "uk": {"left": -17, "bottom": 44, "right": 11, "top": 73},
            "west-europe": {"left": -17, "bottom": 35, "right": 26, "top": 73},
            "india": {"left": 60, "bottom": 6, "right": 97, "top": 37},
        }
        if self.crop_region in crop_region_map:
            import pyproj

            transformer = pyproj.Transformer.from_proj(
                pyproj.Proj(proj="latlong", datum="WGS84"),
                pyproj.Proj(
                    proj="geos",
                    h=self.satellite_metadata.height,
                    lon_0=self.satellite_metadata.longitude,
                    sweep="y",
                ),
            )
            geos_bounds = transformer.transform_bounds(**crop_region_map[self.crop_region])  # type: ignore

            # Check the produced bounds are within the satellite's spatial coordinates
            cs = self.satellite_metadata.spatial_coordinates
            if (
                geos_bounds[0] < min(cs["x_geostationary"])
                or geos_bounds[1] < min(cs["y_geostationary"])
                or geos_bounds[2] > max(cs["x_geostationary"])
                or geos_bounds[3] > max(cs["y_geostationary"])
            ):
                raise ValueError(
                    f"Crop region {self.crop_region} is outside the bounds of the satellite data.",
                )
            return geos_bounds
        elif self.crop_region == "":
            return None
        else:
            raise ValueError(
                f"Unknown crop region '{self.crop_region}'. "
                f"Expected one of {list(crop_region_map.keys())}.",
            )


@dataclasses.dataclass
class ExtractLatestCommandOptions:
    """Options for the extract latest command."""

    satellite: str
    """The satellite to merge data for."""
    window_mins: int = 210
    """The time window of consumed data to merge."""
    workdir: str = "/mnt/disks/sat"
    """The parent folder to store downloaded and processed data in.

    Can be either a local path or an S3 path (s3://bucket-name/path).
    """
    resolution: int = 3000
    """The resolution in meters to use for the data."""
    crop_region: str = ""
    """The name of the region to crop to. An empty string means no cropping."""

    def __post_init__(self) -> None:
        """Perform some validation on the input data."""
        if self.satellite not in SATELLITE_METADATA:
            raise ValueError(
                f"Invalid satellite '{self.satellite}'. Must be one of {SATELLITE_METADATA.keys()}",
            )
        if self.window_mins <= 0:
            raise ValueError("Window size must be positive.")

        if (
            len(
                [
                    c
                    for c in self.satellite_metadata.channels
                    if self.resolution in c.resolution_meters
                ],
            )
            == 0
        ):
            raise ValueError(
                f"No channels found for resolution {self.resolution} in the provided satellite.",
            )

    @property
    def satellite_metadata(self) -> "SatelliteMetadata":
        """Get the metadata for the chosen satellite."""
        return SATELLITE_METADATA[self.satellite]

    @property
    def time_window(self) -> tuple[dt.datetime, dt.datetime]:
        """Get the time window for the given window end and size."""
        end: dt.datetime = self.window_end  # type: ignore
        start: dt.datetime = end - pd.DateOffset(minutes=self.window_mins)
        return start, end

    @property
    def zarr_path(self) -> str:
        """Get the path to the zarr store for the given time."""
        resstr: str = f"{self.resolution}m"
        satstr: str = self.satellite
        suffix: str = ".icechunk"
        cropstr: str = f"{self.crop_region}"

        return f"{self.workdir}/data/{satstr}_{cropstr}{resstr}{suffix}"

    @property
    def latest_path(self) -> str:
        """Get the path to the latest zarr store for the given time window."""
        return f"{self.workdir}/data/latest.zarr.zip"


@dataclasses.dataclass
class SatelliteConsumerConfig:
    """Configuration for the satellite consumer."""

    command: Command
    """The operational mode of the consumer."""
    command_options: ConsumeCommandOptions | ExtractLatestCommandOptions
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
    height: float
    """The height of the satelite above the Earth's surface in meters."""
    product_id: str | list[str]
    """The product ID of the satellite image set."""
    description: str
    """A description of the satellite data set."""
    channels: list[SpectralChannelMetadata]
    """The spectral channels available in the satellite data set."""
    spatial_coordinates: dict[str, list[float]]
    """The spatial coordinates of the full satellite data set."""
    file_filter_regex: str
    """A pattern to filter what files are downloaded for the product."""


SATELLITE_METADATA: dict[str, SatelliteMetadata] = {
    "rss": SatelliteMetadata(
        region="europe",
        cadence_mins=5,
        longitude=9.5,
        height=35785831,
        product_id="EO:EUM:DAT:MSG:MSG15-RSS",
        channels=SEVIRI_CHANNELS,
        description="".join(
            (
                "Rectified (level 1.5) Meteosat SEVIRI Rapid Scan image data for Europe. ",
                "The image region is the top 1/3 of the earth's disk, "
                "covering a latitude range from approximately 15 degrees to 70 degrees. ",
                "The data is transmitted as High Rate transmissions in 12 spectral channels "
                "(11 low and one high resolution). ",
                "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:MSG15-RSS",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(1395187.45153809, 5568748.13049316, 1392)),
        },
        file_filter_regex=r"\S+\.nat$",  # Only want the .nat file
    ),
    "iodc": SatelliteMetadata(
        region="india",
        cadence_mins=15,
        longitude=45.5,
        height=35785831,
        product_id="EO:EUM:DAT:MSG:HRSEVIRI-IODC",
        channels=SEVIRI_CHANNELS,
        description="".join(
            (
                "Rectified (level 1.5) Meteosat SEVIRI image data for the Indian Ocean. ",
                "The data is transmitted as High Rate transmissions in 12 spectral channels "
                "(11 low and one high resolution). ",
                "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:HRSEVIRI-IODC",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(-5565747.79846191, 5568748.01721191, 3712)),
        },
        file_filter_regex=r"\S+\.nat$",  # Only want the .nat file
    ),
    "odegree": SatelliteMetadata(
        region="europe, africa",
        cadence_mins=15,
        longitude=0.0,
        height=35785831,
        product_id="EO:EUM:DAT:MSG:HRSEVIRI",
        channels=SEVIRI_CHANNELS,
        description="".join(
            (
                "Rectified (level 1.5) Meteosat SEVIRI image data for Europe and Africa. ",
                "The data is transmitted as High Rate transmissions in 12 spectral channels ",
                "(11 low and one high resolution). ",
                "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:HRSEVIRI",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(5565747.79846191, -5568748.01721191, 3712)),
            "y_geostationary": list(np.linspace(-5565747.79846191, 5568748.01721191, 3712)),
        },
        file_filter_regex=r"\S+\.nat$",  # Only want the .nat file
    ),
    "odegree-12": SatelliteMetadata(
        region="europe, africa",
        cadence_mins=10,
        longitude=0.0,
        height=35786400,
        product_id="EO:EUM:DAT:0662",
        channels=MTG_1KM_CHANNELS,
        description="".join(
            (
                "Rectified (level 1c) Meteosat-12 image data for Europe and Africa. ",
                "The data is transmitted as High Rate transmissions in 16 spectral channels ",
                "(12 low and 4 high resolution). ",
                "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:0662",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 11136)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 11136)),
        },
        file_filter_regex=r"\S+\.nc$",
    ),
    "odegree-12-highres": SatelliteMetadata(
        region="europe, africa",
        cadence_mins=10,
        longitude=0.0,
        height=35786400,
        product_id="EO:EUM:DAT:0665",
        channels=MTG_1KM_CHANNELS,
        description="".join(
            (
                "Rectified (level 1c) Meteosat-12 image data for Europe and Africa. ",
                "The data is transmitted as High Rate transmissions in 16 spectral channels ",
                "(12 low and 4 high resolution). ",
                "See https://user.eumetsat.int/catalogue/EO:EUM:DAT:0662",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 22272)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 22272)),
        },
        file_filter_regex=r"\S+\.nc$",
    ),
    "goes-east": SatelliteMetadata(
        region="goes-east, Americas, Atlantic Ocean",
        cadence_mins=10,
        longitude=-75.2,
        height=35786023,
        product_id="ABI-L1b-RadF",
        channels=[
            SpectralChannelMetadata("C01", [1000]),
            SpectralChannelMetadata("C02", [500]),
            SpectralChannelMetadata("C03", [1000]),
            SpectralChannelMetadata("C04", [2000]),
            SpectralChannelMetadata("C05", [1000]),
            SpectralChannelMetadata("C06", [2000]),
            SpectralChannelMetadata("C07", [2000]),
            SpectralChannelMetadata("C08", [2000]),
            SpectralChannelMetadata("C09", [2000]),
            SpectralChannelMetadata("C10", [2000]),
            SpectralChannelMetadata("C11", [2000]),
            SpectralChannelMetadata("C12", [2000]),
            SpectralChannelMetadata("C13", [2000]),
            SpectralChannelMetadata("C14", [2000]),
            SpectralChannelMetadata("C15", [2000]),
            SpectralChannelMetadata("C16", [2000]),
        ],
        description="".join(
            (
                "GOES-East ABI Level-1b Radiance data for North America and the Atlantic Ocean.",
                "The data is transmitted in two files: one for the full disk and one for the"
                " CONUS region.",
                "See https://www.ncei.noaa.gov/products/goes/abi-l1b-rad",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 22272)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 22272)),
        },
        file_filter_regex=r"\S+\.nc$",
    ),
    "goes-west": SatelliteMetadata(
        region="goes-west, Americas, Pacific Ocean",
        cadence_mins=10,
        longitude=-137.2,
        height=35786023,
        product_id="ABI-L1b-RadF",
        channels=[
            SpectralChannelMetadata("C01", [1000]),
            SpectralChannelMetadata("C02", [500]),
            SpectralChannelMetadata("C03", [1000]),
            SpectralChannelMetadata("C04", [2000]),
            SpectralChannelMetadata("C05", [1000]),
            SpectralChannelMetadata("C06", [2000]),
            SpectralChannelMetadata("C07", [2000]),
            SpectralChannelMetadata("C08", [2000]),
            SpectralChannelMetadata("C09", [2000]),
            SpectralChannelMetadata("C10", [2000]),
            SpectralChannelMetadata("C11", [2000]),
            SpectralChannelMetadata("C12", [2000]),
            SpectralChannelMetadata("C13", [2000]),
            SpectralChannelMetadata("C14", [2000]),
            SpectralChannelMetadata("C15", [2000]),
            SpectralChannelMetadata("C16", [2000]),
        ],
        description="".join(
            (
                "GOES-West ABI Level-1b Radiance data for North America and the Pacific Ocean.",
                "The data is transmitted in two files: one for the full disk and "
                "one for the CONUS region.",
                "See https://www.ncei.noaa.gov/products/goes/abi-l1b-rad",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 22272)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 22272)),
        },
        file_filter_regex=r"\S+\.nc$",
    ),
    "himawari": SatelliteMetadata(
        region="himawari, Asia, Pacific Ocean",
        cadence_mins=10,
        longitude=140.7,
        height=35786023,
        product_id="AHI-L1b-FLDK",
        channels=[
            SpectralChannelMetadata("B01", [1000]),
            SpectralChannelMetadata("B02", [1000]),
            SpectralChannelMetadata("B03", [500]),
            SpectralChannelMetadata("B04", [1000]),
            SpectralChannelMetadata("B05", [2000]),
            SpectralChannelMetadata("B06", [2000]),
            SpectralChannelMetadata("B07", [2000]),
            SpectralChannelMetadata("B08", [2000]),
            SpectralChannelMetadata("B09", [2000]),
            SpectralChannelMetadata("B10", [2000]),
            SpectralChannelMetadata("B11", [2000]),
            SpectralChannelMetadata("B12", [2000]),
            SpectralChannelMetadata("B13", [2000]),
            SpectralChannelMetadata("B14", [2000]),
            SpectralChannelMetadata("B15", [2000]),
            SpectralChannelMetadata("B16", [2000]),
        ],
        description="".join(
            (
                "Himawari ABI Level-1b Radiance data for Asia and Pacific Ocean.",
                "The data is transmitted in sixteen channels, with varying resolutions.",
                "",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 22272)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 22272)),
        },
        file_filter_regex=r"\S+\.bz2$",
    ),
    "gk2a": SatelliteMetadata(
        region="gk2a, Asia, Pacific Ocean",
        cadence_mins=10,
        longitude=140.7,
        height=35786023,
        product_id="AMI/L1B/FD",
        channels=[
            SpectralChannelMetadata("VI008", [1000]),
            SpectralChannelMetadata("VI005", [1000]),
            SpectralChannelMetadata("VI006", [500]),
            SpectralChannelMetadata("VI004", [1000]),
            SpectralChannelMetadata("IR087", [2000]),
            SpectralChannelMetadata("IR096", [2000]),
            SpectralChannelMetadata("IR105", [2000]),
            SpectralChannelMetadata("IR112", [2000]),
            SpectralChannelMetadata("IR123", [2000]),
            SpectralChannelMetadata("IR133", [2000]),
            SpectralChannelMetadata("NR013", [2000]),
            SpectralChannelMetadata("NR016", [2000]),
            SpectralChannelMetadata("SW038", [2000]),
            SpectralChannelMetadata("WV063", [2000]),
            SpectralChannelMetadata("WV069", [2000]),
            SpectralChannelMetadata("WV073", [2000]),
        ],
        description="".join(
            (
                "GK-2A Level-1b Radiance data for Asia and Pacific Ocean.",
                "The data is transmitted in sixteen channels, with varying resolutions.",
                "",
            ),
        ),
        spatial_coordinates={
            "x_geostationary": list(np.linspace(-5567499.9985508835, 5567499.998550878, 22272)),
            "y_geostationary": list(np.linspace(-5567499.998550887, 5567499.998550878, 22272)),
        },
        file_filter_regex=r"\S+\.nc$",
    ),
}
"""Metadata for the available satellite data sets."""
