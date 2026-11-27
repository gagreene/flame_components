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


def _depth(native, args):
    dtype = np.result_type(*(ma.getdata(arg) for arg in args))
    if dtype.kind == "f" and dtype.itemsize > 8:
        raise TypeError("The Rust backend does not support extended-precision flame depth")
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


def _check_finney_lengths(args):
    for arg in args[1:3]:
        if (
            arg.dtype.kind in "iu"
            and arg.dtype.itemsize == 8
            and (ma.any(arg > 2**53) or (arg.dtype.kind == "i" and ma.any(arg < -(2**53))))
        ):
            raise TypeError("The Rust backend requires Finney tilt integer magnitudes at most 2**53")



def run(equation, *args):
    native = _backend.load_native()
    if equation == "flame_length_parameters":
        return native.flame_length_parameters(*args), False

    if equation == "flame_depth":
        result, dtype = _depth(native, args)
    else:
        args = _active_inputs(equation, args)
        if equation == "flame_tilt" and args[0] == "Finney":
            _check_finney_lengths(args)
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.dtype.kind == "f" and arg.dtype.itemsize > 8:
                raise TypeError("The Rust backend does not support extended-precision inputs")
        parameters = _equations.LENGTH_MODELS[args[0]] if equation == "flame_length" else None
        precision, dtype = _precision.metadata(equation, args, parameters)
        storage = np.dtype("float64")
        if equation == "flame_tilt" and args[0] == "Standard":
            if dtype == np.dtype("float16"):
                precision = (2,) * 7
            else:
                storage = dtype.newbyteorder("=")
        inputs = tuple(_buffers(arg, storage) if isinstance(arg, np.ndarray) else arg for arg in args)
        kernel = getattr(native, equation)
        result = kernel(*inputs) if precision is None else kernel(*inputs, precision=precision)

    data, mask, substituted = result
    return _return_array(data, mask, dtype), substituted
