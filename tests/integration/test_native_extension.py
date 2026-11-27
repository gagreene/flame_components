"""Exercise the compiled extension and precision-sensitive Python calls."""

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import find_spec

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


requires_wide = pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8, reason="binary64 longdouble"
)


@requires_wide
def test_extended_depth_preserves_bits_range_masks_and_strides(native):
    one = np.longdouble(1)
    adjacent = np.nextafter(one, np.longdouble(2))
    huge = np.ldexp(one, 1100)
    data = np.array([[adjacent, huge], [one, huge]], dtype=np.longdouble)
    values = np.ma.array(data, mask=[[False, False], [True, False]])[:, ::-1]
    expected = get_flame_depth(values, np.array(one), backend="python")
    actual = get_flame_depth(values, np.array(one), backend="rust")

    assert expected[0, 1] != one
    assert np.isfinite(expected[0, 0])
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@requires_wide
def test_extended_zero_dimensional_empty_and_swapped_buffers(native):
    one = np.longdouble(1)
    adjacent = np.nextafter(one, np.longdouble(2))
    for values in (np.array(adjacent), np.empty((2, 0), dtype=np.longdouble)):
        swapped = values.astype(values.dtype.newbyteorder("S"))
        expected = get_flame_depth(swapped, one, backend="python")
        actual = get_flame_depth(swapped, one, backend="rust")
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


@requires_wide
def test_extended_standard_tilt_preserves_the_domain_boundary(native):
    one = np.longdouble(1)
    lengths = np.array([one, one, one])
    heights = np.array([np.nextafter(one, np.longdouble(2)), one, np.longdouble("0.5")])
    expected = get_flame_tilt("Standard", lengths, heights, backend="python")
    actual = get_flame_tilt("Standard", lengths, heights, backend="rust")
    assert np.isnan(expected[0])
    assert expected[1] == 0
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=32 * np.finfo(np.longdouble).eps, atol=0)


@requires_wide
@pytest.mark.parametrize(
    ("equation", "args", "kwargs"),
    [
        (get_flame_length, ("Byram_HEAD", [100, 600]), {}),
        (get_flame_length, ("Finney_HEAD", [100, 600], [5, 9]), {}),
        (get_flame_height, ("Nelson", [6, 8]), {
            "fire_type": [1, 3], "fire_intensity": [100, 600], "midflame_ws": [0, 3],
        }),
        (get_flame_height, ("Finney", [6, 8]), {
            "flame_tilt": [20, 30], "slope_angle": [0.5, 12], "slope_units": "degrees",
        }),
        (get_flame_tilt, ("Finney", [6, 8], [3, 5], [-20, 10], "percent"), {}),
        (get_flame_tilt, ("Butler",), {
            "wind_speed": [18, 26], "wind_speed_units": "kph", "canopy_ht": [10, 15],
        }),
        (get_mid_flame_ws, ([20, 30], [30, 1900], [10, 8], [1, 0.5]), {}),
        (get_flame_residence_time, ([3, 6], [0.7, 0.8], [1.1, 4], "min"), {}),
    ],
)
def test_extended_kernels_match_numpy_values_and_dtypes(native, equation, args, kwargs):
    def convert(value):
        return np.array(value, dtype=np.longdouble) if isinstance(value, list) else value

    values = tuple(map(convert, args))
    options = {name: convert(value) for name, value in kwargs.items()}
    _compare(equation, *values, **options, rtol=64 * np.finfo(np.longdouble).eps)


@requires_wide
def test_extended_wind_keeps_canopy_substitution_diagnostic(native):
    height = np.array([0.0, 10.0], dtype=np.longdouble)
    with pytest.warns(UserWarning, match="canopy_ht") as reference:
        expected = get_mid_flame_ws(20, 50, height, 1, backend="python")
    with pytest.warns(UserWarning, match="canopy_ht") as measured:
        actual = get_mid_flame_ws(20, 50, height, 1, backend="rust")
    assert len(measured) == len(reference) == 1
    assert str(measured[0].message) == str(reference[0].message)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=64 * np.finfo(np.longdouble).eps, atol=0)


def test_finney_shape_error_preserves_reference_operand_order(native):
    length = np.ones(3)
    height = np.ones(2)
    slope = np.ones(4)

    with pytest.raises(ValueError, match="broadcast") as python_error:
        get_flame_tilt("Finney", length, height, slope, "degrees", backend="python")
    with pytest.raises(ValueError, match="broadcast") as rust_error:
        get_flame_tilt("Finney", length, height, slope, "degrees", backend="rust")

    assert str(rust_error.value) == str(python_error.value)


@pytest.mark.parametrize("ros_dtype", [np.float64, np.longdouble])
def test_division_cutoff_respects_original_numerator_dtype(native, ros_dtype):
    ros = np.array([3e-307], dtype=ros_dtype)
    fuel = np.array([1.0], dtype=np.float32)
    wind = np.array([1.0], dtype=np.float32)
    _compare(get_flame_residence_time, ros, fuel, wind, "sec")


@pytest.mark.parametrize("value", [1e-8, 3e-8])
def test_depth_rounds_legacy_mixed_float16_inputs_before_multiplication(native, value):
    ros = np.array(value, dtype=np.float64)
    residence = np.array([1000], dtype=np.float16)
    expected = get_flame_depth(ros, residence, backend="python")
    actual = get_flame_depth(ros, residence, backend="rust")

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
