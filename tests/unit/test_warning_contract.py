"""Warning behavior at the public calculation boundary."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

from flame_components import get_flame_residence_time, get_mid_flame_ws


def test_arithmetic_overflow_returns_masked_result_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = get_flame_residence_time(1e-310, 1, 1, "sec")
    assert np.isnan(result)


def test_canopy_warning_points_to_call_site():
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", UserWarning)
        call_line = inspect.currentframe().f_lineno + 1
        get_mid_flame_ws(20, 50, 0, 2)
    assert len(emitted) == 1
    assert emitted[0].category is UserWarning
    assert "canopy_ht" in str(emitted[0].message)
    assert Path(emitted[0].filename).resolve() == Path(__file__).resolve()
    assert emitted[0].lineno == call_line


def test_canopy_warning_precedes_later_calculation_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="canopy_ht"):
            get_mid_flame_ws(20, 50, np.array([0, 1, 2]), np.array([1, 2]))


def test_explicit_numpy_raise_policy_is_preserved():
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow"):
        get_flame_residence_time(1e-310, 1, 1, "sec", backend="python")


def test_canopy_warning_stops_calculation_before_numpy_error_callback():
    callback_errors = []

    def raise_from_callback(error, flags):
        callback_errors.append(error)
        raise RuntimeError(error)

    with np.errstate(over="call", call=raise_from_callback), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="canopy_ht"):
            get_mid_flame_ws(20, 1e308, 0, -5, units="IMP", backend="python")

    assert not callback_errors
