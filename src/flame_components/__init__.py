"""
Flame Components — fire behavior calculations for surface and crown fires.

Public API is implemented in :mod:`flame_components.core`; this module
re-exports it.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('flame-components')
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from a source checkout)
    __version__ = 'unknown'

from flame_components.core import (
    flame_component_array_multiprocessing,
    get_flame_depth,
    get_flame_height,
    get_flame_length,
    get_flame_residence_time,
    get_flame_tilt,
    get_mid_flame_ws,
)

__all__ = [
    'flame_component_array_multiprocessing',
    'get_flame_depth',
    'get_flame_height',
    'get_flame_length',
    'get_flame_residence_time',
    'get_flame_tilt',
    'get_mid_flame_ws',
]
