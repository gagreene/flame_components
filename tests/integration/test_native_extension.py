"""Exercise the compiled extension and precision-sensitive Python calls."""

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import find_spec

import numpy as np
import pytest

from flame_components import (
    get_flame_depth,
    get_flame_height,
    get_flame_tilt,
    get_mid_flame_ws,
)


@pytest.fixture(scope="module")
def native():
    if find_spec("flame_components._native") is None:
        pytest.skip("Rust extension is not installed")

    from flame_components import _native

    return _native


def test_native_kernel_is_compiled_and_keeps_exact_integer_and_mask(native):
    assert any(native.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    data = np.array([2**53 + 1, 7], dtype=np.int64)
    mask = np.array([False, True])
    result, result_mask, substituted = native.flame_depth(
        (data, mask), (np.ones_like(data), np.zeros_like(mask))
    )
    assert result.dtype == data.dtype
    assert result[0] == 2**53 + 1
    np.testing.assert_array_equal(result_mask, mask)
    assert substituted is False


@pytest.mark.parametrize("dtype", ["?", ">u2", "float16"])
def test_depth_accepts_boolean_endian_and_binary16_buffers(native, dtype):
    values = np.array([2, 11, 300], dtype=dtype)[::-1]
    time = np.array([2], dtype=dtype)
    expected = get_flame_depth(values, time, backend="python")
    actual = get_flame_depth(values, time, backend="rust")
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def _compare(function, *args, rtol=5e-7, **kwargs):
    expected = function(*args, **kwargs, backend="python")
    actual = function(*args, **kwargs, backend="rust")
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)


@pytest.mark.parametrize("units", ["degrees", "percent"])
def test_finney_height_uses_numpy_widths(native, units):
    _compare(
        get_flame_height,
        "Finney",
        np.array([8, 7, 6], dtype=np.float32),
        flame_tilt=np.array([25, 14, 22], dtype=np.float32),
        slope_angle=np.array([1, 3, 15], dtype=np.float32),
        slope_units=units,
    )


def test_float32_wind_matches_numpy_branch_at_shelter_threshold(native):
    inputs = [np.array([value], dtype=np.float32) for value in (20, 1600, 8, 0.5)]
    _compare(get_mid_flame_ws, *inputs)


def test_butler_keeps_binary16_intermediates(native):
    _compare(
        get_flame_tilt,
        "Butler",
        wind_speed=np.array([18, 0], dtype=np.float16),
        wind_speed_units="kph",
        canopy_ht=np.array([10, 20], dtype=np.float16),
        rtol=4e-3,
    )


def test_standard_tilt_masks_out_of_domain_binary16_ratio(native):
    _compare(
        get_flame_tilt,
        "Standard",
        np.array([2, 3, 2], dtype=np.float16),
        np.array([1, 1, 3], dtype=np.float16),
        rtol=4e-3,
    )


def test_nelson_preserves_scalar_coefficient_promotion(native):
    _compare(
        get_flame_height,
        "Nelson",
        np.array([100], dtype=np.float16),
        fire_type=np.array(1),
        fire_intensity=np.array([10000], dtype=np.float16),
        midflame_ws=np.array([1], dtype=np.float16),
    )


@pytest.mark.parametrize(("dtype", "large"), [(np.int64, 2**53), (np.uint64, 2**63)])
def test_finney_uses_exact_integer_equality_and_operand_masks(native, dtype, large):
    length = np.array([large + 1, large + 1, large + 1], dtype=dtype)
    height = np.ma.array([large, large + 1, large], mask=[False, False, True], dtype=dtype)

    expected = get_flame_tilt("Finney", length, height, -10.0, "degrees", backend="python")
    actual = get_flame_tilt("Finney", length, height, -10.0, "degrees", backend="rust")

    assert expected[0] > 19
    assert expected[1] == 0
    assert np.isnan(expected[2])
    np.testing.assert_allclose(actual, expected, rtol=2e-14)


def test_standard_tilt_respects_magnitude_dependent_zero_dimensional_dtype(native):
    length = np.array(1e39)
    height = np.array([1e38, 2e38], dtype=np.float32)
    expected = get_flame_tilt("Standard", length, height, backend="python")
    actual = get_flame_tilt("Standard", length, height, backend="rust")

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-14)


@pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8, reason="binary64 longdouble")
def test_extended_precision_requires_explicit_python_backend(native):
    values = np.array([2.0], dtype=np.longdouble)
    with pytest.raises(TypeError, match="extended-precision"):
        get_flame_depth(values, values, backend="rust")
    np.testing.assert_array_equal(get_flame_depth(values, values, backend="python"), [4])


def test_native_boolean_buffers_accept_noncanonical_numpy_true(native):
    data = np.array([0, 2, 3], dtype=np.uint8).view(np.bool_)
    mask = np.array([0, 0, 2], dtype=np.uint8).view(np.bool_)
    result, result_mask, _ = native.flame_depth(
        (data, mask), (np.ones((), dtype=np.bool_), np.zeros((), dtype=np.bool_))
    )
    np.testing.assert_array_equal(result[:2], [False, True])
    np.testing.assert_array_equal(result_mask, [False, False, True])


def test_finney_shape_error_preserves_reference_operand_order(native):
    length = np.ones(3)
    height = np.ones(2)
    slope = np.ones(4)

    with pytest.raises(ValueError, match="broadcast") as python_error:
        get_flame_tilt("Finney", length, height, slope, "degrees", backend="python")
    with pytest.raises(ValueError, match="broadcast") as rust_error:
        get_flame_tilt("Finney", length, height, slope, "degrees", backend="rust")

    assert str(rust_error.value) == str(python_error.value)
