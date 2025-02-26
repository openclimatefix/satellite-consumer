"""Exceptions for the satellite_consumer package."""

class DownloadError(Exception):
    """Error experienced dutring download."""
    pass

class ValidationError(Exception):
    """Error experienced during validation."""
    pass
