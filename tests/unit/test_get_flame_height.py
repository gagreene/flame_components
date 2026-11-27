# tests/unit/test_get_flame_height.py
"""Unit tests for get_flame_height."""
import numpy as np
import pytest

from flame_components import get_flame_height


class TestGetFlameHeightNelsonModel:
    def test_surface_fire_known_value(self):
        """
        Nelson, fire_type=1 (surface): a=1/360
        height = (1/360) * 1000 / 2 = 1000/720 = 1.3889 m
        """
        result = get_flame_height(
            model='Nelson', flame_length=5.0,
            fire_type=1, fire_intensity=1000.0, midflame_ws=2.0
        )
        assert result == pytest.approx(1000.0 / 720.0, rel=1e-4)

    def test_zero_wind_returns_flame_length(self):
        """When midflame_ws=0, height equals flame_length (vertical flame)."""
        result = get_flame_height(
            model='Nelson', flame_length=5.0,
            fire_type=1, fire_intensity=1000.0, midflame_ws=0.0
        )
        assert result == pytest.approx(5.0)

    def test_height_capped_at_flame_length(self):
        """Computed height > flame_length is capped at flame_length."""
        result = get_flame_height(
            model='Nelson', flame_length=1.0,
            fire_type=1, fire_intensity=10000.0, midflame_ws=1.0
        )
        assert result == pytest.approx(1.0)

    def test_string_surface_equals_integer_1(self):
        """fire_type='surface' and fire_type=1 produce identical results."""
        int_result = get_flame_height(
            'Nelson', 5.0, fire_type=1,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        str_result = get_flame_height(
            'Nelson', 5.0, fire_type='surface',
            fire_intensity=1000.0, midflame_ws=2.0
        )
        assert str_result == pytest.approx(int_result)

    def test_active_crown_uses_larger_a_coefficient(self):
        """
        Active crown (fire_type=3) uses a=0.0175; surface uses a=1/360≈0.00278.
        With long enough flame_length, crown height >> surface height.
        """
        crown = get_flame_height(
            'Nelson', 100.0, fire_type=3,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        surface = get_flame_height(
            'Nelson', 100.0, fire_type=1,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        assert crown > surface

    def test_passive_crown_same_coefficient_as_surface(self):
        """passive crown (fire_type=2) uses a=1/360, same as surface (type 1)."""
        surface = get_flame_height(
            'Nelson', 100.0, fire_type=1,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        passive = get_flame_height(
            'Nelson', 100.0, fire_type='passive crown',
            fire_intensity=1000.0, midflame_ws=2.0
        )
        assert passive == pytest.approx(surface)

    def test_returns_float_not_array(self):
        result = get_flame_height(
            'Nelson', 5.0, fire_type=1,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        assert isinstance(result, (float, np.floating))

    def test_non_negative(self):
        result = get_flame_height(
            'Nelson', 5.0, fire_type=1,
            fire_intensity=1000.0, midflame_ws=2.0
        )
        assert result >= 0.0


class TestGetFlameHeightFinneyModel:
    def test_zero_tilt_vertical_flame_equals_flame_length(self):
        """
        Finney, tilt=0 (vertical flame), flat ground:
        tilt_h = pi/2 - 0 = pi/2, sin(pi/2) = 1 → height = flame_length
        """
        result = get_flame_height(
            'Finney', flame_length=10.0,
            flame_tilt=0.0, slope_angle=0.0, slope_units='degrees'
        )
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_45_degree_tilt_flat_ground(self):
        """
        Finney, tilt=45 deg, flat ground:
        tilt_h = pi/2 - pi/4 = pi/4, sin(pi/4) = sqrt(2)/2 ≈ 0.7071
        height = 10.0 * 0.7071 = 7.071 m
        """
        result = get_flame_height(
            'Finney', flame_length=10.0,
            flame_tilt=45.0, slope_angle=0.0, slope_units='degrees'
        )
        assert result == pytest.approx(10.0 * np.sin(np.pi / 4), rel=1e-4)

    def test_percent_slope_units_accepted(self):
        """slope_units='percent' works without error and returns non-negative."""
        result = get_flame_height(
            'Finney', flame_length=10.0,
            flame_tilt=30.0, slope_angle=20.0, slope_units='percent'
        )
        assert result >= 0.0

    def test_non_negative_on_steep_slope(self):
        result = get_flame_height(
            'Finney', flame_length=5.0,
            flame_tilt=60.0, slope_angle=30.0, slope_units='degrees'
        )
        assert result >= 0.0


class TestGetFlameHeightArray:
    def test_array_output_shape(self):
        result = get_flame_height(
            'Nelson',
            flame_length=np.array([5.0, 10.0]),
            fire_type=1,
            fire_intensity=np.array([1000.0, 2000.0]),
            midflame_ws=np.array([2.0, 3.0])
        )
        assert result.shape == (2,)

    def test_array_all_non_negative(self):
        result = get_flame_height(
            'Nelson',
            flame_length=np.array([5.0, 10.0, 15.0]),
            fire_type=1,
            fire_intensity=np.array([500.0, 1000.0, 2000.0]),
            midflame_ws=np.array([1.0, 2.0, 3.0])
        )
        assert np.all(result >= 0.0)

    def test_nan_propagates(self):
        result = get_flame_height(
            'Nelson',
            flame_length=np.array([5.0, np.nan]),
            fire_type=1,
            fire_intensity=np.array([1000.0, 2000.0]),
            midflame_ws=np.array([2.0, 3.0])
        )
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetFlameHeightErrors:
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            get_flame_height('NotAModel', 5.0)

    def test_nelson_without_fire_type_raises(self):
        with pytest.raises(ValueError):
            get_flame_height('Nelson', 5.0, fire_intensity=1000.0, midflame_ws=2.0)

    def test_finney_without_slope_raises(self):
        with pytest.raises(ValueError):
            get_flame_height('Finney', 5.0, flame_tilt=30.0)

    def test_non_string_model_raises(self):
        with pytest.raises(TypeError):
            get_flame_height(42, 5.0)

    def test_invalid_integer_fire_type_raises(self):
        """
        Regression test: integer fire_type was never checked against the valid
        domain {1, 2, 3} — only string fire_type went through the fire_type_dict
        lookup. A stray value like 99 previously fell through to the crown-fire
        branch silently (since it isn't in [1, 2], `a` became 0.0175) and returned
        a plausible-looking but physically meaningless number instead of erroring.
        """
        with pytest.raises(ValueError):
            get_flame_height('Nelson', 5.0, fire_type=99, fire_intensity=1000.0, midflame_ws=2.0)

    def test_zero_fire_type_raises(self):
        with pytest.raises(ValueError):
            get_flame_height('Nelson', 5.0, fire_type=0, fire_intensity=1000.0, midflame_ws=2.0)
