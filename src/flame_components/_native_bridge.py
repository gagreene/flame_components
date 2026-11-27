"""Convert normalized NumPy buffers to and from the Rust extension."""

import numpy as np
from numpy import ma

from . import _backend, _equations, _precision


def _high_rank_shape(args):
    arrays = [arg for arg in args if isinstance(arg, np.ndarray)]
    # The native array view supports at most 32 dimensions. NumPy 2 allows 64.
    if any(array.ndim > 32 for array in arrays):
        return np.broadcast_shapes(*(array.shape for array in arrays))
    return None


def _buffers(array, dtype, *, round_to=None, shape=None):
    data = ma.getdata(array)
    mask = ma.getmaskarray(array)
    if round_to is not None:
        with np.errstate(all="ignore"):
            data = data.astype(round_to, copy=False)
    if shape is not None:
        data = np.broadcast_to(data, shape).reshape(-1)
        mask = np.broadcast_to(mask, shape).reshape(-1)
    return np.require(data, dtype=dtype, requirements="A"), np.require(mask, requirements="A")


def _restore_shape(result, shape):
    if shape is None:
        return result
    data, mask, substituted = result
    return data.reshape(shape), mask.reshape(shape), substituted


def _return_array(data, mask, dtype):
    if data.ndim == 0 and bool(mask):
        return ma.masked
    with np.errstate(all="ignore"):
        return ma.array(data.astype(dtype, copy=False), mask=mask, copy=False)


def _is_extended(dtype):
    return dtype.kind == "f" and dtype.itemsize > 8


def _extended_buffers(array, dtype):
    data = np.require(ma.getdata(array), dtype=dtype, requirements=("A", "C"))
    # Keep data and mask ranks fixed even at NumPy's maximum logical rank.
    packet = data.reshape(-1, 1).view(np.uint8)
    mask = np.require(ma.getmaskarray(array).reshape(-1), requirements="A")
    return packet, mask, array.shape


def _extended(native, equation, args, precision, equality=None):
    dtype = np.dtype(np.longdouble)
    info = np.finfo(dtype)
    expected = (dtype.itemsize, info.nmant + 1, info.minexp, info.maxexp)
    if native.extended_format() != expected:
        raise TypeError("The Rust backend does not support this platform's NumPy longdouble format")
    buffers = tuple(
        _extended_buffers(arg, dtype) if isinstance(arg, np.ndarray) else arg
        for arg in args
    )
    if equality is not None:
        equality = (*_buffers(equality.reshape(-1), np.dtype("bool")), equality.shape)
    data, mask, shape, substituted = native.extended(
        equation, buffers, precision, equality=equality
    )
    return data.view(dtype).reshape(shape), mask.reshape(shape), substituted


def _depth(native, args):
    dtype = np.result_type(*(ma.getdata(arg) for arg in args))
    if _is_extended(dtype):
        return _extended(native, "flame_depth", args, None), dtype
    shape = _high_rank_shape(args)
    if dtype == np.dtype("float16"):
        # NumPy rounds the inputs before multiplying. Their product then fits
        # exactly in binary32 until it is rounded back to binary16.
        buffers = (_buffers(arg, np.dtype("float32"), round_to=dtype, shape=shape) for arg in args)
    else:
        buffers = (_buffers(arg, dtype.newbyteorder("="), shape=shape) for arg in args)
    return _restore_shape(native.flame_depth(*buffers), shape), dtype


def _active_inputs(equation, args):
    # Caller validation has already checked supplied optional arguments.
    if equation == "flame_length" and args[0] != "Finney_HEAD":
        return (*args[:2], None)
    if equation == "flame_height":
        if args[0] == "Nelson":
            return (*args[:5], None, None, None)
        return (*args[:2], None, None, None, *args[5:])
    if equation == "flame_tilt":
        if args[0] == "Standard":
            return (*args[:3], None, None, None, None, None)
        if args[0] == "Finney":
            return (*args[:5], None, None, None)
        return (args[0], None, None, None, None, *args[5:])
    return args


def _finney_equality(length, height):
    # Equality must use the original NumPy dtypes before Rust receives f64 inputs.
    try:
        np.broadcast_shapes(length.shape, height.shape)
    except ValueError:
        # Let the Rust kernel report the incompatible operands in equation order.
        return None
    return height == length


def run(equation, *args):
    native = _backend.load_native()
    if equation == "flame_length_parameters":
        return native.flame_length_parameters(*args), False

    if equation == "flame_depth":
        result, dtype = _depth(native, args)
    else:
        args = _active_inputs(equation, args)
        parameters = _equations.LENGTH_MODELS[args[0]] if equation == "flame_length" else None
        precision, dtype = _precision.metadata(equation, args, parameters)
        equality = (
            _finney_equality(args[1], args[2])
            if equation == "flame_tilt" and args[0] == "Finney" else None
        )
        wide_inputs = any(
            isinstance(arg, np.ndarray) and _is_extended(arg.dtype) for arg in args
        )
        if wide_inputs:
            result = _extended(native, equation, args, precision, equality)
        else:
            shape = _high_rank_shape(args)
            inputs = tuple(
                _buffers(arg, np.dtype("float64"), shape=shape) if isinstance(arg, np.ndarray) else arg
                for arg in args
            )
            options = {} if precision is None else {"precision": precision}
            if equality is not None:
                options["equality"] = _buffers(equality, np.dtype("bool"), shape=shape)
            result = _restore_shape(getattr(native, equation)(*inputs, **options), shape)
            if shape is not None and 0 in shape and equation == "mid_flame_ws":
                # Substitution is reported even if later broadcasting yields no cells.
                data, mask, substituted = result
                result = data, mask, substituted or bool(ma.any(args[2] == 0))

    data, mask, substituted = result
    return _return_array(data, mask, dtype), substituted
