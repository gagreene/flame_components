# tests/unit/test_get_flame_depth.py
"""Unit tests for get_flame_depth. Formula: fd = ros * res_time."""
import numpy as np
import pytest

from flame_components import get_flame_depth


class TestGetFlameDepthScalar:
    def test_known_value(self):
        """fd = ros * res_time: 2.0 * 3.0 = 6.0 exactly."""
        assert get_flame_depth(2.0, 3.0) == pytest.approx(6.0)

    def test_zero_ros(self):
        """Zero rate of spread → zero flame depth."""
        assert get_flame_depth(0.0, 5.0) == pytest.approx(0.0)

    def test_zero_res_time(self):
        """Zero residence time → zero flame depth."""
        assert get_flame_depth(5.0, 0.0) == pytest.approx(0.0)

    def test_returns_float_not_array(self):
        """Scalar inputs return a single numeric value, not an ndarray."""
        result = get_flame_depth(2.0, 3.0)
        assert isinstance(result, (float, np.floating))

    def test_negative_ros_clamped_to_zero(self):
        """Negative ros makes the product negative (-1.0*2.0=-2.0); floored at exactly 0."""
        result = get_flame_depth(-1.0, 2.0)
        assert result == pytest.approx(0.0)

    def test_negative_res_time_clamped_to_zero(self):
        """Negative res_time makes the product negative (5.0*-2.0=-10.0); floored at exactly 0."""
        result = get_flame_depth(5.0, -2.0)
        assert result == pytest.approx(0.0)


class TestGetFlameDepthArray:
    def test_array_output_shape(self):
        """Array inputs return array with same shape."""
        result = get_flame_depth(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        assert result.shape == (3,)

    def test_array_known_values(self):
        """Element-wise: [1*2, 2*3, 3*4] = [2, 6, 12]."""
        result = get_flame_depth(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result, [2.0, 6.0, 12.0])

    def test_nan_propagates(self):
        """NaN in input propagates to NaN in output."""
        result = get_flame_depth(np.array([1.0, np.nan]), np.array([2.0, 3.0]))
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetFlameDepthErrors:
    def test_string_ros_raises_type_error(self):
        with pytest.raises(TypeError):
            get_flame_depth('fast', 3.0)

    def test_string_res_time_raises_type_error(self):
        with pytest.raises(TypeError):
            get_flame_depth(2.0, 'long')
