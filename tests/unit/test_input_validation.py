"""Public input validation and block partitioning regressions."""
import numpy as np
import pytest

from flame_components import flame_component_array_multiprocessing, get_flame_depth


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
