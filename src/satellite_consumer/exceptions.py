"""Exceptions for the satellite_consumer package."""


class DownloadError(Exception):
    """Error experienced dutring download."""

    pass


class ValidationError(Exception):
    """Error experienced during validation."""

    pass


class NotYetAvailableError(DownloadError):
    """A product exists upstream but this account may not download it yet.

    EUMETSAT's Data Store answers 403 for a recent Meteosat scan an unlicensed account is
    not yet entitled to: only the hourly scans are released at once, the rest after a delay.
    """

    pass
