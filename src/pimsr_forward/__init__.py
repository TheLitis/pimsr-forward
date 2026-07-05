"""pimsr-forward: MT + gravity forward modeling and dataset builder."""

from .dataset import build_dataset, gravity_stations
from .gravity import bouguer_slab_gz, layered_prism_bounds, prism_gz
from .mt1d import (
    default_period_band,
    impedance_to_apparent,
    mt1d_impedance,
    mt1d_response,
)
from .sensors import SensorModel

__all__ = [
    "mt1d_impedance",
    "mt1d_response",
    "impedance_to_apparent",
    "default_period_band",
    "prism_gz",
    "layered_prism_bounds",
    "bouguer_slab_gz",
    "SensorModel",
    "build_dataset",
    "gravity_stations",
]

__version__ = "0.1.0"
