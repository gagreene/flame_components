# tests/unit/test_get_flame_length.py
"""Unit tests for get_flame_length."""
import numpy as np
import pytest

from flame_components import get_flame_length


class TestGetFlameLengthScalar:
    def test_byram_head_known_value(self):
        """
        Byram_HEAD: fl = 0.0775 * I^0.46
        I=100: 0.0775 * 100^0.46 = 0.0775 * 8.318 = 0.6446 m
        """
        result = get_flame_length('Byram_HEAD', 100.0)
        assert result == pytest.approx(0.6446, rel=1e-3)

    def test_zero_intensity_returns_zero(self):
        """Zero fire intensity → zero flame length (0^0.46 = 0)."""
        result = get_flame_length('Byram_HEAD', 0.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_returns_float_not_array(self):
        result = get_flame_length('Byram_HEAD', 1000.0)
        assert isinstance(result, (float, np.floating))

    def test_non_negative(self):
        result = get_flame_length('Nelson1_HEAD', 500.0)
        assert result >= 0.0

    def test_params_only_returns_tuple(self):
        """params_only=True returns the (a, b) model parameter tuple."""
        result = get_flame_length('Byram_HEAD', 100.0, params_only=True)
        assert isinstance(result, tuple)
        assert result == pytest.approx((0.0775, 0.4600))

    def test_finney_head_without_flame_depth_raises(self):
        """Finney_HEAD requires flame_depth; omitting it raises ValueError."""
        with pytest.raises(ValueError):
            get_flame_length('Finney_HEAD', 1000.0)

    def test_finney_head_known_value(self):
        """
        Finney_HEAD: fl = 0.01051 * I^0.774 / D^0.161
        I=1000, D=5: 0.01051 * 1000^0.774 / 5^0.161 = 1.7024 m
        Anchors the coefficient/exponent tuple for the special three-parameter model.
        """
        expected = 0.01051 * (1000.0 ** 0.774) / (5.0 ** 0.161)
        result = get_flame_length('Finney_HEAD', 1000.0, flame_depth=5.0)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_finney_head_zero_flame_depth_is_undefined_returns_nan(self):
        """
        Finney_HEAD divides by flame_depth^0.161; flame_depth=0 is a division by
        zero. Documents the NaN-for-undefined convention (zero depth is numerically
        degenerate, not a type/domain error) rather than raising.
        """
        result = get_flame_length('Finney_HEAD', 1000.0, flame_depth=0.0)
        assert np.isnan(result)

    def test_negative_fire_intensity_is_undefined_returns_nan(self):
        """
        Fire intensity is physically non-negative. A negative value raises a
        fractional power of a negative number (undefined in the reals); documents
        that this returns NaN rather than a silently-wrong or complex value.
        """
        result = get_flame_length('Byram_HEAD', -100.0)
        assert np.isnan(result)

    def test_params_only_non_bool_raises_type_error(self):
        with pytest.raises(TypeError):
            get_flame_length('Byram_HEAD', 100.0, params_only='yes')


class TestGetFlameLengthArray:
    def test_array_output_shape(self):
        result = get_flame_length('Byram_HEAD', np.array([100.0, 500.0, 1000.0]))
        assert result.shape == (3,)

    def test_monotone_increasing_with_intensity(self):
        """Higher fire intensity → longer flame (monotone increasing)."""
        result = get_flame_length('Byram_HEAD', np.array([100.0, 500.0, 1000.0]))
        assert result[0] < result[1] < result[2]

    def test_nan_propagates(self):
        result = get_flame_length('Byram_HEAD', np.array([100.0, np.nan]))
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetFlameLengthErrors:
    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError):
            get_flame_length('NotAModel', 100.0)

    def test_non_string_model_raises(self):
        with pytest.raises(TypeError):
            get_flame_length(42, 100.0)

    def test_string_intensity_raises(self):
        with pytest.raises(TypeError):
            get_flame_length('Byram_HEAD', 'high')
