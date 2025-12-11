"""Models internal to the package."""
import dataclasses


@dataclasses.dataclass
class SpectralChannel:
    """Object to store metadata about a spectral channel."""
    representation: str
    satpy_index: int
    name: str
