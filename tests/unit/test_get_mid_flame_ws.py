# tests/unit/test_get_mid_flame_ws.py
"""Unit tests for get_mid_flame_ws."""
import numpy as np
import pytest

from flame_components import get_mid_flame_ws


class TestGetMidFlameWSScalar:
    def test_si_returns_positive_float(self):
        """Valid SI inputs produce a positive mid-flame wind speed (m/s)."""
        result = get_mid_flame_ws(
            wind_speed=36.0, canopy_cover=50,
            canopy_ht=20.0, canopy_baseht=5.0, units='SI'
        )
        assert result > 0.0

    def test_imp_returns_positive_float(self):
        """Valid IMP inputs produce a positive mid-flame wind speed (m/s)."""
        result = get_mid_flame_ws(
            wind_speed=20.0, canopy_cover=50,
            canopy_ht=60.0, canopy_baseht=10.0, units='IMP'
        )
        assert result > 0.0

    def test_zero_wind_returns_zero(self):
        """Zero input wind speed → zero mid-flame wind speed."""
        result = get_mid_flame_ws(
            wind_speed=0.0, canopy_cover=50,
            canopy_ht=20.0, canopy_baseht=5.0, units='SI'
        )
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_returns_float_not_array(self):
        result = get_mid_flame_ws(
            wind_speed=36.0, canopy_cover=50,
            canopy_ht=20.0, canopy_baseht=5.0, units='SI'
        )
        assert isinstance(result, (float, np.floating))

    def test_canopy_ht_zero_warns(self):
        """canopy_ht=0 triggers UserWarning (zero-division guard)."""
        with pytest.warns(UserWarning, match='canopy_ht'):
            get_mid_flame_ws(
                wind_speed=36.0, canopy_cover=50,
                canopy_ht=0.0, canopy_baseht=0.0, units='SI'
            )

    def test_canopy_ht_zero_returns_non_negative(self):
        """After the zero-height substitution, result is still non-negative."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = get_mid_flame_ws(
                wind_speed=36.0, canopy_cover=50,
                canopy_ht=0.0, canopy_baseht=0.0, units='SI'
            )
        assert result >= 0.0

    def test_si_and_imp_agree_for_equivalent_inputs(self):
        """
        SI and IMP describe the same physical scenario in different units — but
        NOT the same wind measurement: SI expects 10-m wind speed (km/h), IMP
        expects 20-ft wind speed (mi/h), and the two differ by the documented 1.15
        factor (Lawson and Armitage 2008). Converting 36 km/h (10-m) to its 20-ft
        mph equivalent (36 / 1.15 / 1.60934 mph) — plus straightforward m->ft
        height conversions — must produce the same mid-flame wind speed (m/s)
        regardless of which unit system described the scenario. Previously this
        test only checked `>= 0` on each result independently, which would pass
        even if the two unit conversions silently disagreed.
        """
        si_result = get_mid_flame_ws(36.0, 50, 20.0, 5.0, 'SI')
        imp_result = get_mid_flame_ws(36.0 / 1.15 / 1.60934, 50, 20.0 * 3.28084, 5.0 * 3.28084, 'IMP')
        assert imp_result == pytest.approx(si_result, rel=1e-3)

    def test_negative_wind_speed_clamped_to_zero(self):
        """Negative wind speed is non-physical; result is floored at exactly 0."""
        result = get_mid_flame_ws(
            wind_speed=-20.0, canopy_cover=50,
            canopy_ht=20.0, canopy_baseht=5.0, units='SI'
        )
        assert result == pytest.approx(0.0)


class TestGetMidFlameWSArray:
    def test_array_output_shape(self):
        result = get_mid_flame_ws(
            wind_speed=np.array([20.0, 30.0]),
            canopy_cover=50,
            canopy_ht=np.array([15.0, 25.0]),
            canopy_baseht=np.array([3.0, 5.0]),
            units='SI'
        )
        assert result.shape == (2,)

    def test_array_all_non_negative(self):
        result = get_mid_flame_ws(
            wind_speed=np.array([10.0, 20.0, 30.0]),
            canopy_cover=50,
            canopy_ht=np.array([10.0, 20.0, 30.0]),
            canopy_baseht=np.array([2.0, 4.0, 6.0]),
            units='SI'
        )
        assert np.all(result >= 0.0)

    def test_nan_propagates(self):
        result = get_mid_flame_ws(
            wind_speed=np.array([20.0, np.nan]),
            canopy_cover=50,
            canopy_ht=np.array([20.0, 20.0]),
            canopy_baseht=np.array([5.0, 5.0]),
            units='SI'
        )
        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestGetMidFlameWSErrors:
    def test_invalid_units_raises(self):
        with pytest.raises(ValueError):
            get_mid_flame_ws(20.0, 50, 20.0, 5.0, units='METRIC')

    def test_non_string_units_raises(self):
        with pytest.raises(TypeError):
            get_mid_flame_ws(20.0, 50, 20.0, 5.0, units=42)

    def test_string_wind_speed_raises(self):
        with pytest.raises(TypeError):
            get_mid_flame_ws('fast', 50, 20.0, 5.0)
