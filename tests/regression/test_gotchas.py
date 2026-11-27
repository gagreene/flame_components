# tests/regression/test_gotchas.py
"""
Regression tests for bugs documented in docs/CODEBASE.md and fixed in
docs/GOTCHAS_FIX_PLAN.md. These must never regress.
Updated to use snake_case function names (snake_case since 0.1.0).
"""
import numpy as np
import pytest

from flame_components import (
    flame_component_array_multiprocessing,
    get_flame_height,
    get_flame_residence_time,
)


class TestFireTypeBugs:
    """
    Two bugs in get_flame_height fire_type validation (fixed):
      Bug 1: str not in isinstance guard → TypeError for string fire_type
      Bug 2: isinstance checked midflame_ws not fire_type → AttributeError
             when midflame_ws is ndarray and fire_type is scalar
    """

    def test_string_fire_type_surface_does_not_raise(self):
        """Bug 1: fire_type='surface' must not raise TypeError."""
        result = get_flame_height(
            model='Nelson',
            flame_length=5.0,
            fire_type='surface',
            fire_intensity=1000.0,
            midflame_ws=2.0,
        )
        assert result >= 0

    def test_string_fire_type_active_crown_does_not_raise(self):
        """Bug 1: fire_type='active crown' must not raise TypeError."""
        result = get_flame_height(
            model='Nelson',
            flame_length=10.0,
            fire_type='active crown',
            fire_intensity=5000.0,
            midflame_ws=3.0,
        )
        assert result >= 0

    def test_string_fire_type_with_array_midflame_ws(self):
        """Both bugs: string fire_type + ndarray midflame_ws requires both fixes."""
        result = get_flame_height(
            model='Nelson',
            flame_length=np.array([5.0, 6.0]),
            fire_type='surface',
            fire_intensity=np.array([1000.0, 1500.0]),
            midflame_ws=np.array([2.0, 2.5]),
        )
        assert result.shape == (2,)
        assert np.all(result >= 0)


class TestMultiprocessingBugs:
    """
    Two bugs in flame_component_array_multiprocessing (fixed):
      Bug 3: *kwargs (positional tuple) instead of **kwargs (keyword dict)
      Bug 4: 'getFlameResidence' name string (missing 'Time') resolved to None
    """

    def test_kwargs_signature_accepts_keyword_arguments(self):
        """
        Bug 3: keyword arguments must not raise TypeError. Also asserts the
        dispatcher's result is numerically correct, not just exception-free --
        concatenating the returned blocks must match a direct call.
        """
        ros = np.array([1.0, 1.5, 2.0, 2.5])
        fc_arr = np.array([0.5, 0.6, 0.7, 0.8])
        ws = np.array([1.0, 1.2, 1.4, 1.6])
        try:
            blocks = flame_component_array_multiprocessing(
                'flame_residence', 2, None,
                ros=ros, fuel_consumption=fc_arr, midflame_ws=ws, units='sec'
            )
        except TypeError as e:
            pytest.fail(f"Bug 3 still present — *kwargs rejected keyword args: {e}")
        result = np.concatenate(blocks)
        expected = get_flame_residence_time(ros, fc_arr, ws, units='sec')
        np.testing.assert_allclose(result, expected)

    def test_flame_residence_key_resolves_to_valid_function(self):
        """
        Bug 4: 'flame_residence' key must resolve to get_flame_residence_time.
        Also asserts the resolved function actually computes the correct result,
        not just that it resolves to something callable.
        """
        ros = np.array([1.0, 1.5, 2.0, 2.5])
        fc_arr = np.array([0.5, 0.6, 0.7, 0.8])
        ws = np.array([1.0, 1.2, 1.4, 1.6])
        try:
            blocks = flame_component_array_multiprocessing(
                'flame_residence', 2, None,
                ros=ros, fuel_consumption=fc_arr, midflame_ws=ws, units='sec'
            )
        except ValueError as e:
            assert 'does not exist' not in str(e), (
                f"Bug 4 still present — 'flame_residence' resolved to None: {e}"
            )
            raise
        result = np.concatenate(blocks)
        expected = get_flame_residence_time(ros, fc_arr, ws, units='sec')
        np.testing.assert_allclose(result, expected)
