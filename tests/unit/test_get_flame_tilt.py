# tests/unit/test_get_flame_tilt.py
"""Unit tests for get_flame_tilt."""
import numpy as np
import pytest

from flame_components import get_flame_tilt


class TestGetFlameTiltStandardModel:
    def test_vertical_flame_zero_tilt(self):
        """Vertical flame (height == length) → tilt = 0 degrees."""
        result = get_flame_tilt('Standard', flame_length=10.0, flame_height=10.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_horizontal_flame_90_degrees(self):
        """Fully horizontal flame (height = 0) → tilt = 90 degrees."""
        result = get_flame_tilt('Standard', flame_length=10.0, flame_height=0.0)
        assert result == pytest.approx(90.0, rel=1e-4)

    def test_arccos_half_equals_60_degrees(self):
        """height/length = 0.5 → arccos(0.5) = 60 degrees."""
        result = get_flame_tilt('Standard', flame_length=10.0, flame_height=5.0)
        assert result == pytest.approx(60.0, rel=1e-4)

    def test_returns_float_not_array(self):
        result = get_flame_tilt('Standard', flame_length=10.0, flame_height=7.0)
        assert isinstance(result, (float, np.floating))

    def test_non_negative(self):
        result = get_flame_tilt('Standard', flame_length=10.0, flame_height=7.0)
        assert result >= 0.0

    def test_height_exceeds_length_is_undefined_returns_nan(self):
        """
        tilt = arccos(height/length) requires height/length in [0, 1] — a flame
        can't be taller than its own length. height > length pushes arccos out of
        its domain; documents that this returns NaN (masked-domain arccos) rather
        than raising, consistent with the library's NaN-for-undefined convention.
        """
        result = get_flame_tilt('Standard', flame_length=5.0, flame_height=10.0)
        assert np.isnan(result)


class TestGetFlameTiltFinneyModel:
    def test_flat_ground_returns_non_negative(self):
        result = get_flame_tilt(
            'Finney', flame_length=10.0, flame_height=8.0,
            slope_angle=0.0, slope_units='degrees'
        )
        assert result >= 0.0

    def test_percent_slope_matches_equivalent_degree_slope(self):
        """
        A 20% slope is arctan(0.20) in degrees (~11.31°). Converting through
        either 'percent' or the equivalent 'degrees' value must produce the same
        tilt, confirming the percent-to-radians conversion is correct rather than
        merely non-negative.
        """
        percent = get_flame_tilt(
            'Finney', flame_length=10.0, flame_height=8.0,
            slope_angle=20.0, slope_units='percent'
        )
        degrees = get_flame_tilt(
            'Finney', flame_length=10.0, flame_height=8.0,
            slope_angle=np.degrees(np.arctan(0.20)), slope_units='degrees'
        )
        assert percent == pytest.approx(degrees, rel=1e-4)


class TestGetFlameTiltButlerModel:
    def test_butler_returns_positive_tilt(self):
        """Crown fire with wind → positive tilt angle."""
        result = get_flame_tilt(
            'Butler', wind_speed=50.0, wind_speed_units='kph', canopy_ht=20.0
        )
        assert result > 0.0

    def test_higher_wind_produces_more_tilt(self):
        """Stronger wind → greater tilt angle."""
        tilt_low = get_flame_tilt('Butler', wind_speed=20.0, wind_speed_units='kph', canopy_ht=20.0)
        tilt_high = get_flame_tilt('Butler', wind_speed=80.0, wind_speed_units='kph', canopy_ht=20.0)
        assert tilt_high > tilt_low

    def test_mph_units_matches_equivalent_kph(self):
        """36 km/h == 22.3694 mph; both must convert to the same internal m/s wind speed."""
        kph = get_flame_tilt('Butler', wind_speed=36.0, wind_speed_units='kph', canopy_ht=20.0)
        mph = get_flame_tilt('Butler', wind_speed=22.3694, wind_speed_units='mph', canopy_ht=20.0)
        assert mph == pytest.approx(kph, rel=1e-4)

    def test_mps_units_matches_equivalent_kph(self):
        """36 km/h == 10 m/s; both must convert to the same internal m/s wind speed."""
        kph = get_flame_tilt('Butler', wind_speed=36.0, wind_speed_units='kph', canopy_ht=20.0)
        mps = get_flame_tilt('Butler', wind_speed=10.0, wind_speed_units='mps', canopy_ht=20.0)
        assert mps == pytest.approx(kph, rel=1e-4)

    def test_zero_canopy_ht_raises(self):
        """
        Regression test: canopy_ht previously had NO validation/masked-array
        conversion at all in this function (unlike every other parameter) — a
        value of 0 leaked a raw ZeroDivisionError from `28 / canopy_ht` instead of
        a clean, typed error matching the rest of the library's validation style.
        """
        with pytest.raises(ValueError):
            get_flame_tilt('Butler', wind_speed=50.0, wind_speed_units='kph', canopy_ht=0.0)

    def test_negative_canopy_ht_raises(self):
        with pytest.raises(ValueError):
            get_flame_tilt('Butler', wind_speed=50.0, wind_speed_units='kph', canopy_ht=-5.0)

    def test_string_canopy_ht_raises_type_error(self):
        with pytest.raises(TypeError):
            get_flame_tilt('Butler', wind_speed=50.0, wind_speed_units='kph', canopy_ht='tall')


class TestGetFlameTiltArray:
    def test_array_output_shape(self):
        result = get_flame_tilt(
            'Standard',
            flame_length=np.array([10.0, 20.0]),
            flame_height=np.array([5.0, 10.0])
        )
        assert result.shape == (2,)

    def test_nan_propagates(self):
        result = get_flame_tilt(
            'Standard',
            flame_length=np.array([10.0, np.nan]),
            flame_height=np.array([5.0, 5.0])
        )
        assert np.isnan(result[1])
        assert not np.isnan(result[0])

    def test_nan_propagates_via_canopy_ht_butler_model(self):
        """
        Regression test: canopy_ht is now converted to a masked array like every
        other parameter, so NaN in a canopy_ht array must mask (not corrupt) only
        the affected element — verifies the fix generalizes to array inputs, not
        just the scalar zero/negative cases.
        """
        result = get_flame_tilt(
            'Butler',
            wind_speed=np.array([50.0, 50.0]),
            wind_speed_units='kph',
            canopy_ht=np.array([20.0, np.nan])
        )
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetFlameTiltErrors:
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            get_flame_tilt('NotAModel', flame_length=10.0, flame_height=5.0)

    def test_standard_missing_flame_height_raises(self):
        with pytest.raises(ValueError):
            get_flame_tilt('Standard', flame_length=10.0)

    def test_butler_missing_canopy_ht_raises(self):
        with pytest.raises(ValueError):
            get_flame_tilt('Butler', wind_speed=50.0, wind_speed_units='kph')
