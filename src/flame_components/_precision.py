"""Resolve NumPy operation dtypes with scalar and one-cell probes."""

from functools import lru_cache

import numpy as np
from numpy import ma

_VALUE_DEPENDENT_SCALARS = np.lib.NumpyVersion(np.__version__) < "2.0.0"


def _spec(array):
    # NumPy 1.x can choose a dtype based on the value of a zero-dimensional operand.
    if _VALUE_DEPENDENT_SCALARS and array.ndim == 0:
        return array.dtype, 0, ma.getdata(array).tobytes(), bool(ma.getmaskarray(array))
    return array.dtype, min(array.ndim, 1), None, False


def _sample(spec):
    dtype, rank, scalar, masked = spec
    if scalar is not None:
        return ma.array(np.frombuffer(scalar, dtype=dtype).reshape(()), mask=masked)
    return ma.array(np.ones((1,) if rank else (), dtype=dtype))


def _widths(*values, divisions=()):
    widths = [value.dtype.itemsize if value.dtype.kind == "f" else 8 for value in values]
    for index, numerator in divisions:
        # NumPy resolves the mask cutoff using the original numerator dtype.
        cutoff = np.abs(np.asarray(numerator)) * np.finfo(float).tiny
        width = cutoff.dtype.itemsize
        mode = 1 if width < 8 else 2 if width == 8 else 3
        widths[index] |= mode << 5
    return tuple(widths)


def _slope_radians(slope, units):
    return ma.arctan(slope / 100) if units == "percent" else np.radians(slope)


@lru_cache(maxsize=128)
def _height(length, tilt, slope, units):
    length, tilt, slope = map(_sample, (length, tilt, slope))
    tilt_radians = np.radians(tilt)
    tilt_horizontal = np.pi / 2 - tilt_radians
    slope_radians = _slope_radians(slope, units)
    flat_height = length * ma.sin(tilt_horizontal)
    relative_tilt = tilt_horizontal - slope_radians
    numerator = length * ma.sin(relative_tilt)
    complement = np.radians(90) - slope_radians
    sloped_height = numerator / ma.sin(complement)
    output = ma.where(slope <= 1, flat_height, sloped_height)
    divisions = [(7, numerator)]
    if units == "percent":
        divisions.append((2, slope))
    return _widths(
        tilt_radians, tilt_horizontal, slope_radians, flat_height, relative_tilt,
        numerator, complement, sloped_height, divisions=divisions,
    ), output.dtype


@lru_cache(maxsize=128)
def _nelson(length, fire_type, intensity, wind):
    length, fire_type, intensity, wind = map(_sample, (length, fire_type, intensity, wind))
    coefficient = ma.where(ma.isin(fire_type, [1, 2]), 1 / 360, 0.0175)
    numerator = coefficient * intensity
    ratio = numerator / wind
    height = ma.where(wind == 0, length, ratio)
    output = ma.where(height > length, length, height)
    return _widths(numerator, ratio, height, divisions=[(1, numerator)]), output.dtype


@lru_cache(maxsize=128)
def _tilt(length, height, slope, units):
    length, height, slope = map(_sample, (length, height, slope))
    slope_radians = _slope_radians(slope, units)
    complement = np.radians(90) - slope_radians
    numerator = height * np.degrees(ma.sin(complement))
    ratio = numerator / length
    tilt_horizontal = ma.arcsin(np.radians(ratio)) + slope_radians
    tilt_vertical = np.pi / 2 - tilt_horizontal
    output = np.degrees(ma.where(height == length, 0, tilt_vertical))
    divisions = [(3, numerator)]
    if units == "percent":
        divisions.append((0, slope))
    return _widths(
        slope_radians, complement, numerator, ratio, tilt_horizontal, tilt_vertical,
        output, divisions=divisions,
    ), output.dtype


@lru_cache(maxsize=128)
def _length(intensity, depth, parameters):
    intensity = _sample(intensity)
    intensity_power = ma.power(intensity, parameters[1])
    numerator = parameters[0] * intensity_power
    if len(parameters) == 2:
        return _widths(intensity_power, numerator, numerator, numerator), numerator.dtype
    depth_power = ma.power(_sample(depth), parameters[2])
    output = numerator / depth_power
    return _widths(
        intensity_power, numerator, depth_power, output, divisions=[(3, numerator)]
    ), output.dtype


@lru_cache(maxsize=128)
def _residence(ros, fuel, wind, units):
    ros, fuel, wind = map(_sample, (ros, fuel, wind))
    fuel_power = ma.power(fuel, 0.25)
    scaled_fuel = 0.39 * fuel_power
    wind_power = ma.power(wind, 1.51)
    numerator = scaled_fuel * wind_power
    denominator = ros / 60
    seconds = numerator / denominator
    output = seconds / 60 if units == "min" else seconds
    divisions = [(4, ros), (5, numerator)]
    if units == "min":
        divisions.append((6, seconds))
    return _widths(
        fuel_power, scaled_fuel, wind_power, numerator, denominator, seconds, output,
        divisions=divisions,
    ), output.dtype


@lru_cache(maxsize=128)
def _wind(wind, cover, canopy, base, units):
    wind, cover, canopy, base = map(_sample, (wind, cover, canopy, base))
    input_wind = wind
    if units == "SI":
        wind = wind / (3.6 * 1.15)
        canopy = canopy * 3.28084
        base = base * 3.28084
    else:
        wind = wind / 2.23694
    ht = ma.where(canopy == 0, 0.5 * 3.28084, canopy)
    difference = ht - base
    ratio = difference / ht
    covered_ratio = ratio * cover
    shelter = covered_ratio / 300
    scaled_ht = 0.36 * ht
    log_numerator = 20 + scaled_ht
    log_denominator = 0.13 * ht
    log_argument = log_numerator / log_denominator
    logarithm = ma.log(log_argument)
    unsheltered_numerator = wind * 1.83
    unsheltered = unsheltered_numerator / logarithm
    sheltered_numerator = wind * 0.555
    sheltered_product = shelter * ht
    sheltered_denominator = ma.sqrt(sheltered_product) * logarithm
    sheltered = sheltered_numerator / sheltered_denominator
    output = ma.where(shelter <= 5, unsheltered, sheltered)
    return _widths(
        wind, canopy, base, ht, difference, ratio, covered_ratio, shelter, scaled_ht,
        log_numerator, log_denominator, log_argument, unsheltered_numerator, unsheltered,
        sheltered_numerator, sheltered_product, sheltered_denominator, sheltered,
        divisions=[(0, input_wind), (5, difference), (7, covered_ratio),
                   (11, log_numerator), (13, unsheltered_numerator), (17, sheltered_numerator)],
    ), output.dtype


@lru_cache(maxsize=128)
def _butler(wind, canopy, units):
    wind, canopy = map(_sample, (wind, canopy))
    input_wind = wind
    if units == "kph":
        wind = wind / 3.6
    elif units == "mph":
        wind = wind / 2.23694
    canopy_ratio = 28 / canopy
    log_argument = 1 + canopy_ratio
    logarithm = 1 + ma.log(log_argument)
    wind_denominator = 3.6 * logarithm
    canopy_wind = wind / wind_denominator
    cube = ma.power(canopy_wind, 3)
    numerator = 3 * cube
    fraction = numerator / (2 * 9.81 * 10)
    output = np.degrees(ma.arctan(ma.sqrt(fraction)))
    return _widths(
        wind, canopy_ratio, log_argument, logarithm, wind_denominator, canopy_wind,
        cube, numerator, fraction, output,
        divisions=[(0, input_wind), (1, 28), (5, wind), (8, numerator)],
    ), output.dtype


@lru_cache(maxsize=128)
def _standard(length, height):
    length, height = map(_sample, (length, height))
    ratio = height / length
    return _widths(ratio, divisions=[(0, height)]) * 7, ratio.dtype


def metadata(equation, args, parameters=None):
    """Return native arithmetic widths and the reference's raw output dtype."""
    arrays = [arg for arg in args if isinstance(arg, np.ndarray)]
    if all(array.dtype == np.dtype("float64") for array in arrays):
        return None, np.dtype("float64")

    with np.errstate(all="ignore"):
        if equation == "mid_flame_ws":
            wind, cover, canopy, base, units = args
            return _wind(_spec(wind), _spec(cover), _spec(canopy), _spec(base), units)
        if equation == "flame_tilt" and args[0] == "Butler":
            _, _, _, _, _, wind, units, canopy = args
            return _butler(_spec(wind), _spec(canopy), units)
        if equation == "flame_height" and args[0] == "Nelson":
            return _nelson(*map(_spec, args[1:5]))
        if equation == "flame_height" and args[0] == "Finney":
            _, length, _, _, _, tilt, slope, units = args
            return _height(_spec(length), _spec(tilt), _spec(slope), units)
        if equation == "flame_tilt" and args[0] == "Finney":
            _, length, height, slope, units, *_ = args
            return _tilt(_spec(length), _spec(height), _spec(slope), units)
        if equation == "flame_tilt" and args[0] == "Standard":
            return _standard(_spec(args[1]), _spec(args[2]))
        if equation == "flame_length":
            _, intensity, depth = args
            return _length(_spec(intensity), _spec(depth) if depth is not None else None, parameters)
        if equation == "flame_residence_time":
            ros, fuel, wind, units = args
            return _residence(_spec(ros), _spec(fuel), _spec(wind), units)
    return None, np.dtype("float64")
