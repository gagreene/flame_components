# tests/integration/test_flame_pipeline.py
"""
Integration tests: chains of flame_components functions used as in real workflows.
These tests verify that output from one function feeds correctly into another.
"""
import numpy as np
import pytest

from flame_components import (
    get_flame_depth,
    get_flame_height,
    get_flame_length,
    get_flame_residence_time,
    get_flame_tilt,
    get_mid_flame_ws,
)


class TestResidenceTimeToFlameDepth:
    """Pipeline: get_flame_residence_time → get_flame_depth."""

    def test_scalar_pipeline_produces_positive_depth(self):
        """Residence time output feeds into flame depth without type errors."""
        ros = 2.0           # m/min
        fuel_consumption = 0.8  # kg/m^2
        midflame_ws = 1.5   # m/s

        res_time = get_flame_residence_time(ros, fuel_consumption, midflame_ws, units='min')
        assert res_time > 0.0

        flame_depth = get_flame_depth(ros, res_time)
        assert flame_depth > 0.0

    def test_residence_depth_pipeline_known_value(self):
        """
        ros=1.0, fuel_consumption=1.0, midflame_ws=1.0, units='min':
        res_time = 0.39 * 1^0.25 * 1^1.51 / (1/60) / 60 = 0.39 min
        flame_depth = ros * res_time = 1.0 * 0.39 = 0.39 m
        Pins an exact numeric value through both functions, catching
        unit-conversion drift between the two that a positivity check can't.
        """
        ros = 1.0
        fuel_consumption = 1.0
        midflame_ws = 1.0

        res_time = get_flame_residence_time(ros, fuel_consumption, midflame_ws, units='min')
        flame_depth = get_flame_depth(ros, res_time)

        assert res_time == pytest.approx(0.39, rel=1e-6)
        assert flame_depth == pytest.approx(0.39, rel=1e-6)

    def test_array_pipeline_preserves_shape(self):
        """Vectorized pipeline preserves shape and all values are non-negative."""
        ros = np.array([1.0, 2.0, 3.0])
        fc = np.array([0.5, 0.8, 1.2])
        ws = np.array([1.0, 1.5, 2.0])

        res_time = get_flame_residence_time(ros, fc, ws, units='min')
        assert res_time.shape == (3,)

        flame_depth = get_flame_depth(ros, res_time)
        assert flame_depth.shape == (3,)
        assert np.all(flame_depth >= 0.0)


class TestMidFlameWSToFlameHeight:
    """Pipeline: get_mid_flame_ws → get_flame_length → get_flame_height."""

    def test_scalar_height_bounded_by_flame_length(self):
        """Height must not exceed flame length (Nelson model enforces this cap)."""
        midflame_ws = get_mid_flame_ws(
            wind_speed=20.0, canopy_cover=40,
            canopy_ht=15.0, canopy_baseht=3.0, units='SI'
        )
        assert midflame_ws >= 0.0

        fl = get_flame_length('Byram_HEAD', 2000.0)

        height = get_flame_height(
            'Nelson', fl, fire_type='surface',
            fire_intensity=2000.0, midflame_ws=midflame_ws
        )
        assert 0.0 <= height <= fl + 1e-9

    def test_array_pipeline_shape_and_bounds(self):
        """Vectorized pipeline: shapes match, heights bounded by flame lengths."""
        wind_speed = np.array([10.0, 20.0, 30.0])
        canopy_ht = np.array([10.0, 15.0, 20.0])
        canopy_baseht = np.array([2.0, 3.0, 4.0])

        midflame_ws = get_mid_flame_ws(wind_speed, 50, canopy_ht, canopy_baseht, 'SI')
        assert midflame_ws.shape == (3,)

        fire_intensity = np.array([500.0, 1000.0, 2000.0])
        fl = get_flame_length('Byram_HEAD', fire_intensity)

        height = get_flame_height(
            'Nelson', fl, fire_type='surface',
            fire_intensity=fire_intensity, midflame_ws=midflame_ws
        )
        assert height.shape == (3,)
        assert np.all(height >= 0.0)
        assert np.all(height <= fl + 1e-9)


class TestFlameLengthHeightTiltRoundtrip:
    """
    Roundtrip: height → get_flame_tilt (Standard) → get_flame_height (Finney, flat).
    On flat ground, Finney recovers the original height from Standard tilt.
    """

    def test_roundtrip_recovers_original_height(self):
        """
        Standard tilt from a known height, then Finney height on flat ground:
        height_in → tilt_standard → height_finney ≈ height_in (flat ground identity).

        Derivation:
          tilt = arccos(h/L)
          height_finney = L * sin(pi/2 - tilt) = L * cos(arccos(h/L)) = h  ✓
        """
        flame_length = 8.0
        height_input = 6.0

        tilt = get_flame_tilt('Standard', flame_length=flame_length, flame_height=height_input)
        height_recovered = get_flame_height(
            'Finney', flame_length=flame_length,
            flame_tilt=tilt, slope_angle=0.0, slope_units='degrees'
        )
        assert height_recovered == pytest.approx(height_input, rel=1e-3)


class TestNaNPropagationAcrossPipeline:
    """NaN in any array input propagates through the entire pipeline."""

    def test_nan_in_ros_propagates_to_depth(self):
        """NaN in ros contaminates both residence time and flame depth."""
        ros = np.array([1.0, np.nan, 3.0])
        fc = np.array([0.5, 0.8, 1.0])
        ws = np.array([1.0, 1.5, 2.0])

        res_time = get_flame_residence_time(ros, fc, ws, 'min')
        flame_depth = get_flame_depth(ros, res_time)

        assert np.isnan(flame_depth[1])
        assert not np.isnan(flame_depth[0])
        assert not np.isnan(flame_depth[2])
