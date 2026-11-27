"""Check the native byte adapter on both extended and binary64 platforms."""

from importlib.util import find_spec

import numpy as np
import pytest

from flame_components import get_flame_depth


@pytest.fixture(scope="module")
def native_format():
    if find_spec("flame_components._native") is None:
        pytest.skip("Rust extension is not installed")

    from flame_components import _native

    dtype = np.dtype(np.longdouble)
    info = np.finfo(dtype)
    expected = (dtype.itemsize, info.nmant + 1, info.minexp, info.maxexp)
    if _native.extended_format() != expected:
        pytest.skip("Rust extension and NumPy use different long double layouts")
    return _native, dtype


def _pack(data, mask=None):
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)
    return data.reshape(-1, 1).view(np.uint8), mask.reshape(-1), data.shape


def test_extended_adapter_preserves_strides_scalar_broadcast_and_mask(native_format):
    native, dtype = native_format
    data, mask, shape = _pack(np.array([2, 4, 8], dtype=dtype), np.array([False, True, False]))
    storage = np.empty((3, 2 * dtype.itemsize), dtype=np.uint8)
    strided = storage[::-1, ::2]
    strided[...] = data
    time = _pack(np.array(3, dtype=dtype))

    result, measured_mask, measured_shape, substituted = native.extended(
        "flame_depth", ((strided, mask, shape), time)
    )

    assert result.shape == (3, dtype.itemsize)
    assert tuple(measured_shape) == shape
    np.testing.assert_array_equal(result.view(dtype).reshape(3)[~measured_mask], [6, 24])
    np.testing.assert_array_equal(measured_mask, mask)
    assert substituted is False


@pytest.mark.parametrize("wrong_axis", [False, True])
def test_extended_adapter_rejects_invalid_byte_shape(native_format, wrong_axis):
    native, dtype = native_format
    data, mask, shape = _pack(np.array([2, 4], dtype=dtype))
    if wrong_axis:
        invalid = (data[:, :-1], mask, shape)
    else:
        invalid = (data, mask[:1], shape)

    with pytest.raises(ValueError, match="extended data shape"):
        native.extended("flame_depth", (invalid, (data, mask, shape)))


def test_extended_adapter_rejects_non_byte_data(native_format):
    native, dtype = native_format
    data, mask, shape = _pack(np.array([2, 4], dtype=dtype))

    with pytest.raises(TypeError):
        native.extended(
            "flame_depth", ((data.astype(np.int8), mask, shape), (data, mask, shape))
        )


def test_extended_adapter_rejects_inconsistent_logical_shape(native_format):
    native, dtype = native_format
    data, mask, shape = _pack(np.array([2, 4], dtype=dtype))

    with pytest.raises(ValueError, match="shape"):
        native.extended("flame_depth", ((data, mask, (3,)), (data, mask, shape)))


def _max_rank():
    return 32 if np.lib.NumpyVersion(np.__version__) < "2.0.0" else 64


def test_extended_buffer_packer_accepts_maximum_rank():
    from flame_components._native_bridge import _extended_buffers

    dtype = np.dtype(np.longdouble)
    shape = (1,) * _max_rank()
    values = np.ones(shape, dtype=dtype)
    packet, mask, logical_shape = _extended_buffers(values, dtype)

    assert packet.shape == (1, dtype.itemsize)
    assert mask.shape == (1,)
    assert logical_shape == shape
    np.testing.assert_array_equal(packet.view(dtype).reshape(shape), values)


def test_native_extended_depth_preserves_maximum_rank(native_format):
    native, dtype = native_format
    shape = (1,) * _max_rank()
    data = np.full(shape, 2, dtype=dtype)
    scalar = np.array(3, dtype=dtype)

    result, measured_mask, logical_shape, _ = native.extended(
        "flame_depth", (_pack(data), _pack(scalar))
    )

    assert result.shape == (1, dtype.itemsize)
    assert measured_mask.shape == (1,)
    assert tuple(logical_shape) == shape
    np.testing.assert_array_equal(result.view(dtype).reshape(shape), np.full(shape, 6))


@pytest.mark.parametrize("kind", ["int64", "float64", "longdouble"])
def test_public_native_depth_preserves_maximum_rank(native_format, kind):
    _, _ = native_format
    shape = (1,) * (_max_rank() - 1) + (2,)
    ros = np.array([2, 5], dtype=kind).reshape(shape)
    time = np.array(3, dtype=kind)
    expected = get_flame_depth(ros, time, backend="python")
    actual = get_flame_depth(ros, time, backend="rust")

    assert actual.shape == expected.shape == shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
