# tests/unit/test_get_flame_residence_time.py
"""Unit tests for get_flame_residence_time."""
import numpy as np
import pytest

from flame_components import get_flame_residence_time


class TestGetFlameResidenceTimeScalar:
    def test_known_value_seconds(self):
        """
        Formula: 0.39 * W^0.25 * u^1.51 / (ros/60)
        W=1, u=1, ros=1: 0.39 * 1 * 1 / (1/60) = 23.4 sec
        """
        result = get_flame_residence_time(ros=1.0, fuel_consumption=1.0,
                                          midflame_ws=1.0, units='sec')
        assert result == pytest.approx(23.4, rel=1e-4)

    def test_known_value_minutes(self):
        """Same inputs with units='min': 23.4 / 60 = 0.39 min."""
        result = get_flame_residence_time(ros=1.0, fuel_consumption=1.0,
                                          midflame_ws=1.0, units='min')
        assert result == pytest.approx(0.39, rel=1e-4)

    def test_returns_float_not_array(self):
        result = get_flame_residence_time(1.0, 1.0, 1.0, 'sec')
        assert isinstance(result, (float, np.floating))

    def test_negative_ros_clamped_to_zero(self):
        """Negative ros flips the formula's sign; result is floored at exactly 0."""
        result = get_flame_residence_time(ros=-1.0, fuel_consumption=1.0, midflame_ws=1.0, units='sec')
        assert result == pytest.approx(0.0)

    def test_zero_ros_is_undefined_returns_nan(self):
        """
        ros=0 means no fire spread; residence time is undefined (division by zero
        in the ros/60 denominator). Documents the library's NaN-for-undefined convention
        rather than raising — ros=0 is numerically degenerate, not a type/domain error.
        """
        result = get_flame_residence_time(ros=0.0, fuel_consumption=1.0, midflame_ws=1.0, units='sec')
        assert np.isnan(result)


class TestGetFlameResidenceTimeArray:
    def test_array_output_shape(self):
        result = get_flame_residence_time(
            np.array([1.0, 2.0]),
            np.array([0.5, 1.0]),
            np.array([1.0, 1.5]),
            'sec'
        )
        assert result.shape == (2,)

    def test_array_all_non_negative(self):
        result = get_flame_residence_time(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 0.75, 1.0]),
            np.array([1.0, 1.5, 2.0]),
            'sec'
        )
        assert np.all(result >= 0.0)

    def test_nan_propagates(self):
        result = get_flame_residence_time(
            np.array([1.0, np.nan]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            'sec'
        )
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetFlameResidenceTimeErrors:
    def test_string_ros_raises(self):
        with pytest.raises(TypeError):
            get_flame_residence_time('fast', 1.0, 1.0, 'sec')

    def test_invalid_units_raises(self):
        """
        Regression test: 'units' previously had NO validation at all — any string
        other than 'min' silently fell through to the 'sec' behavior, so a typo like
        'seconds' would silently return a value in the wrong unit with no error.
        """
        with pytest.raises(ValueError):
            get_flame_residence_time(1.0, 1.0, 1.0, units='seconds')

    def test_non_string_units_raises(self):
        with pytest.raises(TypeError):
            get_flame_residence_time(1.0, 1.0, 1.0, units=42)
