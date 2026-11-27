"""Check the native byte adapter on both extended and binary64 platforms."""

from importlib.util import find_spec

import numpy as np
import pytest


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


def _pack(data):
    return data.reshape(-1).view(np.uint8).reshape((*data.shape, data.itemsize))


def test_extended_adapter_preserves_strides_scalar_broadcast_and_mask(native_format):
    native, dtype = native_format
    data = _pack(np.array([2, 4, 8], dtype=dtype))
    storage = np.empty((3, 2 * dtype.itemsize), dtype=np.uint8)
    strided = storage[::-1, ::2]
    strided[...] = data
    time = _pack(np.array(3, dtype=dtype))

    result, mask, substituted = native.extended(
        "flame_depth",
        ((strided, np.array([False, True, False])), (time, np.array(False))),
    )

    assert result.shape == (3, dtype.itemsize)
    np.testing.assert_array_equal(result.view(dtype).reshape(3)[~mask], [6, 24])
    np.testing.assert_array_equal(mask, [False, True, False])
    assert substituted is False


@pytest.mark.parametrize("wrong_axis", [False, True])
def test_extended_adapter_rejects_invalid_byte_shape(native_format, wrong_axis):
    native, dtype = native_format
    data = _pack(np.array([2, 4], dtype=dtype))
    mask = np.zeros(2, dtype=bool)
    if wrong_axis:
        invalid = (data[:, :-1], mask)
    else:
        invalid = (data, mask[:1])

    with pytest.raises(ValueError, match="extended data shape"):
        native.extended("flame_depth", (invalid, (data, mask)))


def test_extended_adapter_rejects_non_byte_data(native_format):
    native, dtype = native_format
    data = _pack(np.array([2, 4], dtype=dtype))
    mask = np.zeros(2, dtype=bool)

    with pytest.raises(TypeError):
        native.extended(
            "flame_depth", ((data.astype(np.int8), mask), (data, mask))
        )
