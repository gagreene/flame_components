"""Convert normalized NumPy buffers to and from the Rust extension."""

import numpy as np
from numpy import ma

from . import _backend, _equations, _precision


def _buffers(array, dtype):
    return (
        np.require(ma.getdata(array), dtype=dtype, requirements="A"),
        np.require(ma.getmaskarray(array), requirements="A"),
    )


def _return_array(data, mask, dtype):
    if data.ndim == 0 and bool(mask):
        return ma.masked
    with np.errstate(all="ignore"):
        return ma.array(data.astype(dtype, copy=False), mask=mask, copy=False)


def _is_extended(dtype):
    return dtype.kind == "f" and dtype.itemsize > 8


def _extended_buffers(array, dtype):
    data = np.require(ma.getdata(array), dtype=dtype, requirements=("A", "C"))
    # A trailing byte axis works for zero-dimensional and empty arrays as well.
    packet = data.reshape((*data.shape, 1)).view(np.uint8)
    return packet, np.require(ma.getmaskarray(array), requirements="A")


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
    data, mask, substituted = native.extended(equation, buffers, precision, equality=equality)
    return data.view(dtype).reshape(mask.shape), mask, substituted


def _depth(native, args):
    dtype = np.result_type(*(ma.getdata(arg) for arg in args))
    if _is_extended(dtype):
        return _extended(native, "flame_depth", args, None), dtype
    # A binary16 product fits exactly in binary32 before the final rounding.
    storage = np.dtype("float32") if dtype == np.dtype("float16") else dtype.newbyteorder("=")
    return native.flame_depth(*(_buffers(arg, storage) for arg in args)), dtype


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
    return _buffers(height == length, np.dtype("bool"))


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
            if equation == "flame_tilt" and args[0] == "Standard":
                precision = (dtype.itemsize,) * 7
            result = _extended(native, equation, args, precision, equality)
        else:
            storage = np.dtype("float64")
            if equation == "flame_tilt" and args[0] == "Standard":
                if dtype == np.dtype("float16"):
                    precision = (2,) * 7
                else:
                    storage = dtype.newbyteorder("=")
            inputs = tuple(
                _buffers(arg, storage) if isinstance(arg, np.ndarray) else arg
                for arg in args
            )
            options = {} if precision is None else {"precision": precision}
            if equality is not None:
                options["equality"] = equality
            result = getattr(native, equation)(*inputs, **options)

    data, mask, substituted = result
    return _return_array(data, mask, dtype), substituted
