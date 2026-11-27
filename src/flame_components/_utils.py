"""Shared input validation for the flame calculations."""
from numbers import Integral, Real

import numpy as np


def numeric_array(name, value, *, optional=False):
    """Validate a real scalar or array and combine any existing mask with NaNs."""
    if optional and value is None:
        return None
    if isinstance(value, Real):
        # Keep a one-element axis for scalar calculations and scalar result extraction.
        value = np.array([value])
        if value.dtype.kind == "O":
            value = value.astype(float)
    elif not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be a real number or a real numeric NumPy array")
    return np.ma.array(value, mask=np.isnan(np.ma.getdata(value)))


def nonnegative_result(value, return_array):
    """Clamp negative results, restore NaNs, and return the caller's output type."""
    result = np.ma.maximum(value, 0)
    if np.ma.is_masked(result):
        if result.dtype.kind in "biu":
            result = result.astype(float)
        data = np.ma.filled(result, np.nan)
    else:
        data = np.ma.getdata(result)
    return data if return_array else data[0]


def check_choice(name, value, choices, *, optional=False):
    """Check a string parameter against the choices supplied by its caller."""
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")


def require_inputs(model, **inputs):
    """Report all missing parameters for the selected model."""
    missing = [name for name, value in inputs.items() if value is None]
    if missing:
        raise ValueError(f"The {model} model requires {', '.join(missing)}")


def check_bool(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def check_int(name, value, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def check_positive(name, values):
    if np.ma.any(values <= 0):
        raise ValueError(f"{name} must be greater than zero")


def fire_type_array(value):
    """Convert Nelson fire type labels or numeric codes to a masked array."""
    codes = {"surface": 1, "passive crown": 2, "active crown": 3}
    if isinstance(value, str):
        check_choice("fire_type", value, codes)
        value = codes[value]
    result = numeric_array("fire_type", value)
    if np.any(~np.ma.getmaskarray(result) & ~np.isin(result.data, list(codes.values()))):
        raise ValueError("fire_type must be 1 (surface), 2 (passive crown), or 3 (active crown)")
    return result


def common_row_count(inputs):
    """Check that array arguments can be partitioned along a shared first axis."""
    shape = None
    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.shape[0] == 0:
                raise ValueError(f"{name} must have at least one row")
            if shape is not None and value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            shape = value.shape
    if shape is None:
        raise ValueError("Multiprocessing requires at least one array input")
    return shape[0]
