"""Public input validation and block partitioning regressions."""
import numpy as np
import pytest

from flame_components import (
    flame_component_array_multiprocessing,
    get_flame_depth,
    get_flame_height,
    get_flame_length,
)


@pytest.mark.parametrize("ros", [np.array(["2"]), np.array([1j])])
def test_depth_rejects_non_real_arrays(ros):
    with pytest.raises(TypeError, match="ros"):
        get_flame_depth(ros, 2.0)


@pytest.mark.parametrize(
    ("options", "error", "name"),
    [
        ({"block_size": 0}, ValueError, "block_size"),
        ({"block_size": -1}, ValueError, "block_size"),
        ({"block_size": 1.5}, TypeError, "block_size"),
        ({"num_processors": 2.5}, TypeError, "num_processors"),
    ],
)
def test_invalid_worker_options(options, error, name):
    with pytest.raises(error, match=name):
        flame_component_array_multiprocessing("flame_depth", ros=np.ones(2), res_time=1, **options)


@pytest.mark.parametrize("array", [np.array(1.0), np.array([])])
def test_workers_require_rows(array):
    with pytest.raises(ValueError, match="ros"):
        flame_component_array_multiprocessing("flame_depth", ros=array, res_time=1)


@pytest.mark.parametrize(("count", "block_size"), [(1, None), (3, 2), (2, 10)])
def test_workers_cover_every_row(count, block_size):
    ros = np.arange(count, dtype=float)
    blocks = flame_component_array_multiprocessing("flame_depth", block_size=block_size, ros=ros, res_time=2)
    np.testing.assert_array_equal(np.concatenate(blocks), ros * 2)


def test_depth_preserves_integer_precision_for_scalars_and_arrays():
    value = 2**53 + 1
    assert int(get_flame_depth(value, 1)) == value
    assert int(get_flame_depth(np.array([value]), np.array([1]))[0]) == value


def test_depth_handles_zero_dimensional_arrays():
    result = get_flame_depth(np.array(2.0), np.array(3.0))
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result == 6.0


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_depth_preserves_floating_dtype(dtype):
    result = get_flame_depth(np.array([2], dtype=dtype), np.array([3], dtype=dtype))
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, [6])


def test_depth_preserves_unsigned_integer_precision():
    value = 2**63 + 1
    result = get_flame_depth(np.array([value], dtype=np.uint64), np.array([1], dtype=np.uint64))
    assert result.dtype == np.uint64
    assert int(result[0]) == value


def test_depth_preserves_computed_nan():
    with np.errstate(invalid="ignore"):
        result = get_flame_depth(np.array([0.0]), np.array([np.inf]))
    assert np.isnan(result[0])


def test_depth_preserves_negative_zero():
    result = get_flame_depth(np.array([-0.0]), np.array([1.0]))
    assert result[0] == 0.0
    assert np.signbit(result[0])


def test_depth_fills_missing_integer_results_with_nan():
    ros = np.ma.array([2, 3], mask=[False, True])
    result = get_flame_depth(ros, 2)
    np.testing.assert_array_equal(result, [4.0, np.nan])


@pytest.mark.parametrize(
    ("dtype", "ros", "expected"),
    [(np.int8, 100, 0), (np.uint8, 200, 144)],
)
def test_depth_uses_numpy_integer_arithmetic(dtype, ros, expected):
    result = get_flame_depth(np.array([ros], dtype=dtype), np.array([2], dtype=dtype))
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, [expected])


def test_depth_rejects_timedelta_scalar():
    with pytest.raises(TypeError, match="ros must be a real number"):
        get_flame_depth(np.timedelta64(2, "s"), 2)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int8])
def test_finney_height_uses_double_precision_for_scalar_flame_tilt(dtype):
    options = {"slope_angle": 12.0, "slope_units": "degrees"}
    result = get_flame_height("Finney", 8.0, flame_tilt=dtype(37), **options)
    expected = get_flame_height("Finney", 8.0, flame_tilt=37.0, **options)
    assert result == expected


def test_finney_height_ignores_nelson_fire_type():
    result = get_flame_height(
        "Finney", 8.0, fire_type="unused", flame_tilt=0, slope_angle=0, slope_units="degrees"
    )
    assert result == 8.0


@pytest.mark.parametrize("params_only", [False, True])
def test_barbon_length_accepts_legacy_model_name(params_only):
    assert get_flame_length("Barbon_iNOWIND", 1000.0, params_only=params_only) == get_flame_length(
        "Barbon_NOWIND", 1000.0, params_only=params_only
    )
