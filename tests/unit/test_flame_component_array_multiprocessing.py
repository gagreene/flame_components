# tests/unit/test_flame_component_array_multiprocessing.py
"""Unit tests for flame_component_array_multiprocessing."""
import warnings

import numpy as np
import pytest

from flame_components import flame_component_array_multiprocessing, get_flame_depth


class TestFlameComponentArrayMultiprocessingBasic:
    def test_matches_direct_call_when_concatenated(self):
        """
        The function returns a list of per-block arrays (NOT pre-concatenated —
        see the docstring). Concatenating the blocks must reproduce exactly what
        calling the underlying function directly on the full array would give.
        """
        ros = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res_time = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        blocks = flame_component_array_multiprocessing(
            'flame_depth', num_processors=2, block_size=None, ros=ros, res_time=res_time
        )
        assert isinstance(blocks, list)
        result = np.concatenate(blocks)
        expected = get_flame_depth(ros, res_time)
        np.testing.assert_allclose(result, expected)

    def test_explicit_block_size_produces_expected_block_count(self):
        ros = np.array([1.0, 2.0, 3.0, 4.0])
        res_time = np.array([1.0, 1.0, 1.0, 1.0])
        blocks = flame_component_array_multiprocessing(
            'flame_depth', num_processors=2, block_size=2, ros=ros, res_time=res_time
        )
        assert len(blocks) == 2
        assert all(len(b) == 2 for b in blocks)


class TestFlameComponentArrayMultiprocessingEdgeCases:
    def test_num_processors_less_than_two_warns_and_still_runs(self):
        """
        Regression test: num_processors < 2 previously did `raise UserWarning(...)`
        instead of `warnings.warn(...)` — a UserWarning IS an Exception subclass,
        so `raise` terminated the call entirely instead of warning and continuing
        with the documented num_processors=2 fallback. Verifies the call now
        completes successfully and actually emits the warning.
        """
        ros = np.array([1.0, 2.0, 3.0, 4.0])
        res_time = np.array([1.0, 1.0, 1.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            blocks = flame_component_array_multiprocessing(
                'flame_depth', num_processors=1, block_size=None, ros=ros, res_time=res_time
            )
        assert any(issubclass(w.category, UserWarning) for w in caught)
        np.testing.assert_allclose(np.concatenate(blocks), get_flame_depth(ros, res_time))

    def test_num_processors_zero_warns_and_still_runs(self):
        """Same fallback path as num_processors=1, exercised at the num_processors=0 boundary."""
        ros = np.array([1.0, 2.0, 3.0, 4.0])
        res_time = np.array([1.0, 1.0, 1.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            blocks = flame_component_array_multiprocessing(
                'flame_depth', num_processors=0, block_size=None, ros=ros, res_time=res_time
            )
        assert any(issubclass(w.category, UserWarning) for w in caught)
        np.testing.assert_allclose(np.concatenate(blocks), get_flame_depth(ros, res_time))


class TestFlameComponentArrayMultiprocessingErrors:
    def test_unknown_flame_function_raises(self):
        with pytest.raises(ValueError):
            flame_component_array_multiprocessing(
                'not_a_real_function', num_processors=2, block_size=None,
                ros=np.array([1.0, 2.0])
            )

    def test_no_array_inputs_raises(self):
        """Scalar-only kwargs give nothing to split into blocks."""
        with pytest.raises(ValueError):
            flame_component_array_multiprocessing(
                'flame_depth', num_processors=2, block_size=None, ros=1.0, res_time=2.0
            )

    def test_mismatched_array_shapes_raises(self):
        with pytest.raises(ValueError):
            flame_component_array_multiprocessing(
                'flame_depth', num_processors=2, block_size=None,
                ros=np.array([1.0, 2.0, 3.0]), res_time=np.array([1.0, 2.0])
            )
