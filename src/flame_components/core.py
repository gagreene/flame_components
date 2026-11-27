"""
Created on Wed Oct  11 13:25:52 2023

@author: Gregory A. Greene

Flame Components — fire behavior calculations for surface and crown fires.

Input NaNs and explicit NumPy masks are handled per cell. The selected branch
determines which masks propagate. Masked results are returned as NaN. The
Python backend preserves explicit NumPy error modes such as over="raise".
"""

import warnings

from numpy import errstate, geterr, ndarray

from . import _backend, _blocks, _equations, _native_bridge, _utils

_CANOPY_WARNING = (
    'One or more canopy_ht values are 0. These will be replaced with 1.6404 ft (0.5 m) '
    'to avoid division-by-zero in the Albini & Baughman (1979) equation. '
    'Note: by this point canopy_ht has been converted to feet internally.'
)


def _python_arithmetic():
    ignore_warnings = {event: "ignore" for event, handling in geterr().items() if handling == "warn"}
    return errstate(**ignore_warnings)


def _run_equation(backend, equation, *args):
    if backend == "rust":
        result, _ = _native_bridge.run(equation.__name__, *args)
        return result
    with _python_arithmetic():
        return equation(*args)


# FUNCTION TO CALCULATE MID-FLAME WIND SPEED
def get_mid_flame_ws(wind_speed: float | ndarray,
                  canopy_cover: float | ndarray,
                  canopy_ht: float | ndarray,
                  canopy_baseht: float | ndarray,
                  units: str = 'SI', *, backend: str = "auto") -> int | float | ndarray:
    """
    Function to calculate mid-flame wind speed.
    :param wind_speed: reference-height wind speed.
        SI mode  → 10-m wind speed (km/h) measured above open ground or canopy top.
        IMP mode → 20-ft wind speed (mi/h).
        WARNING: The 10-m reference height used here differs from the canopy-top wind
        speed formula used in getFlameTilt (Butler model). Do not use the output of
        this function as input to the Butler tilt model.
    :param canopy_cover: canopy cover (percent)
    :param canopy_ht: stand height (m for SI, ft for IMP). Values of 0 will be replaced
        with 0.5 m equivalent (1.6404 ft) and a UserWarning will be issued.
    :param canopy_baseht: canopy base height (m for SI, ft for IMP)
    :param units: units of input data ("SI" or "IMP")
        SI = metric (10-m ws in km/h, height in m, cbh in m)
        IMP = imperial (20-ft ws in mi/h, height in ft, cbh in ft)
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: mid-flame wind speed (m/s). Always metres per second regardless of input units.
        For SI inputs, heights are converted to feet internally before applying the
        Albini and Baughman (1979) equation (defined in US customary units).

    Divide by 3.6 to convert from km/h to m/s\n
    Divide windspeed by 1.15 to convert from 10-m to 20-ft equivalent (Lawson and Armitage 2008)\n
    Calculate mid-flame windspeed (m/s) using under canopy equations (Albini and Baughman 1979)
          Uc/Uh = 0.555 / (sqrt(f*H) * ln((20 + 0.36*H) / (0.13*H)))\n
          where f = crown ratio * canopy cover (proportion) / 3, H = stand height (ft)\n
    Equations explained well in Andrews (2012) - Modeling wind adjustment factor and
    midflame wind speed for Rothermel's surface fire spread model\n
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [wind_speed, canopy_cover, canopy_ht, canopy_baseht])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    wind_speed = _utils.numeric_array("wind_speed", wind_speed)
    canopy_cover = _utils.numeric_array("canopy_cover", canopy_cover)
    canopy_ht = _utils.numeric_array("canopy_ht", canopy_ht)
    canopy_baseht = _utils.numeric_array("canopy_baseht", canopy_baseht)
    _utils.check_choice("units", units, ("SI", "IMP"))

    if backend == "rust":
        try:
            result, substituted = _native_bridge.run(
                "mid_flame_ws", wind_speed, canopy_cover, canopy_ht, canopy_baseht, units
            )
        except ValueError as error:
            if getattr(error, "substituted_canopy_height", False):
                warnings.warn(_CANOPY_WARNING, UserWarning, stacklevel=2)
            raise
        if substituted:
            warnings.warn(_CANOPY_WARNING, UserWarning, stacklevel=2)
    else:
        with _python_arithmetic():
            wind_speed, canopy_ht, canopy_baseht, substituted = _equations.prepare_mid_flame_ws(
                wind_speed, canopy_ht, canopy_baseht, units
            )
            if substituted:
                warnings.warn(_CANOPY_WARNING, UserWarning, stacklevel=2)
            result = _equations.mid_flame_ws(wind_speed, canopy_cover, canopy_ht, canopy_baseht)
    return _utils.nonnegative_result(result, return_array)


# FUNCTION TO CALCULATE FLAME LENGTH
def get_flame_length(model: str,
                   fire_intensity: float | ndarray,
                   flame_depth: float | ndarray | None = None,
                   params_only: bool = False, *, backend: str = "auto") -> int | float | ndarray:
    """
    Function to estimate flame length from a variety of published models.
    Equation from Nelson and Adkins (1986) - referenced in Cruz and Alexander (2018)
    :param model: published flame length model selector
        All models come from Finney and Grumstrup (2023), including their own 2023 model.
    :param fire_intensity: head fire intensity (kW/m)
    :param flame_depth: head fire flame depth (m) [OPTIONAL]
        Only required for "Finney_HEAD" model
    :param params_only: return only the model parameters ("True" or "False"; Default = False)
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: flame length (m) or model parameters
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [fire_intensity, flame_depth])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    _utils.check_choice("model", model, _equations.LENGTH_MODELS)
    if model == "Finney_HEAD":
        _utils.require_inputs(model, fire_intensity=fire_intensity, flame_depth=flame_depth)
    fire_intensity = _utils.numeric_array("fire_intensity", fire_intensity)
    flame_depth = _utils.numeric_array("flame_depth", flame_depth, optional=True)
    _utils.check_bool("params_only", params_only)

    if params_only:
        return _run_equation(backend, _equations.flame_length_parameters, model)

    result = _run_equation(backend, _equations.flame_length, model, fire_intensity, flame_depth)
    return _utils.nonnegative_result(result, return_array)


# FUNCTION TO CALCULATE FLAME HEIGHT
def get_flame_height(model: str,
                   flame_length: float | ndarray,  # For all models
                   fire_type: str | int | ndarray | None = None,  # For Nelson model
                   fire_intensity: float | ndarray | None = None,  # For Nelson model
                   midflame_ws: float | ndarray | None = None,  # For Nelson model
                   flame_tilt: float | ndarray | None = None,  # For Finney model
                   slope_angle: float | ndarray | None = None,  # For Finney model
                   slope_units: str | None = None,  # For Finney model
                   *, backend: str = "auto") -> int | float | ndarray:
    """
    Equations from Nelson and Adkins (1986) or Finney and Martin (1992) - referenced in Cruz and Alexander (2018)
    :param model: model used to estimate flame height ("Nelson", "Finney")
        The Finney (Simard) model is suggested if you already know tilt angle
    :param flame_length: [Both models] head fire flame length (m)
    :param fire_type: [Nelson model only] type of fire (1 or "surface", 2 or "passive crown", 3 or "active crown")
    :param fire_intensity: [Nelson model only] head fire intensity (kW/m)
    :param midflame_ws: [Nelson model only] mid-flame wind speed (m/s)
    :param flame_tilt: [Finney model only] head fire flame tilt relative to vertical (degrees)
    :param slope_angle: [Finney model only] Slope angle of ground (degrees or percent)
    :param slope_units: [Finney model only] Units of slope-angle ("degrees" or "percent")
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: head fire flame height (m)
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [flame_length, fire_type, fire_intensity,
                                                  midflame_ws, flame_tilt, slope_angle, slope_units])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    _utils.check_choice("model", model, ("Nelson", "Finney"))
    if model == "Nelson":
        _utils.require_inputs(model, fire_type=fire_type, fire_intensity=fire_intensity, midflame_ws=midflame_ws)
    else:
        _utils.require_inputs(model, flame_tilt=flame_tilt, slope_angle=slope_angle, slope_units=slope_units)
    flame_length = _utils.numeric_array("flame_length", flame_length)
    if model == "Nelson":
        fire_type = _utils.fire_type_array(fire_type)
    fire_intensity = _utils.numeric_array("fire_intensity", fire_intensity, optional=True)
    midflame_ws = _utils.numeric_array("midflame_ws", midflame_ws, optional=True)
    flame_tilt = _utils.numeric_array("flame_tilt", flame_tilt, optional=True, scalar_as_float=True)
    slope_angle = _utils.numeric_array("slope_angle", slope_angle, optional=True)
    _utils.check_choice("slope_units", slope_units, ("degrees", "percent"), optional=True)

    result = _run_equation(
        backend, _equations.flame_height, model, flame_length, fire_type, fire_intensity,
        midflame_ws, flame_tilt, slope_angle, slope_units,
    )
    return _utils.nonnegative_result(result, return_array)


# FUNCTION TO CALCULATE FLAME TILT
def get_flame_tilt(model: str,
                 flame_length: float | ndarray | None = None,
                 flame_height: float | ndarray | None = None,
                 slope_angle: float | ndarray | None = None,
                 slope_units: str | None = None,
                 wind_speed: float | ndarray | None = None,
                 wind_speed_units: str | None = None,
                 canopy_ht: float | ndarray | None = None, *, backend: str = "auto") -> int | float | ndarray:
    """
    Function calculates flame tilt using Finney and Martin (1992) and Butler et al. (2004) equations
    :param model: The flame tilt model to use ("Standard", "Finney", "Butler")
        Standard = Use standard geometry calculations (for flat ground)
        Finney = Use Finney and Martin (1992) (aka "Simard" model) model (for sloped ground)
        Butler = Use Butler et al. (2004) model (for crown fires only)
    :param flame_length: [Standard & Finney models only]
        Head fire flame length (m)
    :param flame_height: [Standard & Finney models only]
        Head fire flame height (m)
    :param slope_angle: [Finney model only]
        Slope angle of ground (degrees or percent)
    :param slope_units: [Finney model only]
        Units of slope-angle ("degrees" or "percent")
    :param wind_speed: [Butler model only]
        10m wind speed (i.e., measured 10m above open ground or forest canopy)
    :param wind_speed_units: [Butler model only]
        Units of "wind_speed" parameter ("kph", "mps", "mph")
            kph = kilometers per hour
            mps = meters per second
            mph = miles per hour
    :param canopy_ht: [Butler model only]
        Height of the canopy above the ground (m)
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: angle of head fire flame tilt (degrees)
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [flame_length, flame_height, slope_angle, wind_speed, canopy_ht])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    _utils.check_choice("model", model, ("Standard", "Finney", "Butler"))
    if model == "Standard":
        _utils.require_inputs(model, flame_length=flame_length, flame_height=flame_height)
    elif model == "Finney":
        _utils.require_inputs(model, flame_length=flame_length, flame_height=flame_height,
                              slope_angle=slope_angle, slope_units=slope_units)
    else:
        _utils.require_inputs(model, wind_speed=wind_speed, wind_speed_units=wind_speed_units, canopy_ht=canopy_ht)
    flame_length = _utils.numeric_array("flame_length", flame_length, optional=True)
    flame_height = _utils.numeric_array("flame_height", flame_height, optional=True)
    slope_angle = _utils.numeric_array("slope_angle", slope_angle, optional=True)
    _utils.check_choice("slope_units", slope_units, ("degrees", "percent"), optional=True)
    wind_speed = _utils.numeric_array("wind_speed", wind_speed, optional=True)
    _utils.check_choice("wind_speed_units", wind_speed_units, ("kph", "mps", "mph"), optional=True)
    canopy_ht = _utils.numeric_array("canopy_ht", canopy_ht, optional=True)
    if model == "Butler":
        _utils.check_positive("canopy_ht", canopy_ht)

    result = _run_equation(
        backend, _equations.flame_tilt, model, flame_length, flame_height, slope_angle,
        slope_units, wind_speed, wind_speed_units, canopy_ht,
    )
    return _utils.nonnegative_result(result, return_array)


# FUNCTION TO CALCULATE FLAME RESIDENCE TIME
def get_flame_residence_time(ros: float | ndarray,
                          fuel_consumption: float | ndarray,
                          midflame_ws: float | ndarray,
                          units: str, *, backend: str = "auto") -> int | float | ndarray:
    """
    Function to calculate flame residence time using equation from Nelson and Adkins (1988).
    :param ros: Fire rate of spread (m/min). NOTE: input units are always m/min.
        The "units" parameter below controls the OUTPUT unit only.
    :param fuel_consumption: Amount of fuel consumed by fire front (kg/m^2)
    :param midflame_ws: Mid-flame wind speed (m/s)
    :param units: Output unit for flame residence time ("sec" or "min").
        This parameter does NOT affect the expected input unit for ros — ros must
        always be provided in m/min.
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: Flame residence time in seconds ("sec") or minutes ("min").
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [ros, fuel_consumption, midflame_ws])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    ros = _utils.numeric_array("ros", ros)
    fuel_consumption = _utils.numeric_array("fuel_consumption", fuel_consumption)
    midflame_ws = _utils.numeric_array("midflame_ws", midflame_ws)
    _utils.check_choice("units", units, ("sec", "min"))

    result = _run_equation(backend, _equations.flame_residence_time, ros, fuel_consumption, midflame_ws, units)
    return _utils.nonnegative_result(result, return_array)


# FUNCTION TO CALCULATE FLAME DEPTH
def get_flame_depth(ros: float | ndarray,
                  res_time: float | ndarray, *, backend: str = "auto") -> int | float | ndarray:
    """
    Calculate flame depth using equation from Fons et al. (1963)
    :param ros: Fire rate of spread (m/min)
    :param res_time: Time from initial temp rise to the time of definite drop after reaching peak temp (min).
        Definition per Rothermel and Deeming (1980)
    :param backend: Calculation backend ("auto", "rust", or "python").
    :return: flame depth (m)
    """
    backend = _backend.resolve_backend(backend)

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [ros, res_time])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    ros = _utils.numeric_array("ros", ros)
    res_time = _utils.numeric_array("res_time", res_time)

    result = _run_equation(backend, _equations.flame_depth, ros, res_time)
    return _utils.nonnegative_result(result, return_array)


def flame_component_array_multiprocessing(flame_function: str,
                                        num_processors: int = 2,
                                        block_size: int | None = None,
                                        *, backend: str = "auto", **kwargs) -> list:
    """
    Break input arrays into blocks and process them with a pool of worker processes.
    Each block uses the function requested in the "flame_function" parameter.

    **flame_function options**
        "midflame_ws", "flame_length", "flame_height",
        "flame_tilt", "flame_residence" (→ get_flame_residence_time), "flame_depth"

    :param flame_function: The flame components function to implement.
    :param num_processors: Number of cores for multiprocessing
    :param block_size: Maximum rows per block (positive integer).
        If None, divide the row count by num_processors, round down, and use at least one row.
    :param backend: Calculation backend ("auto", "rust", or "python").
    :param kwargs: A dictionary of inputs for the requested flame_components function.
        The dictionary keys must match the required input parameters for the requested function.
        Refer to the function docstring for parameter requirements.
    :return: One result per block in input order, including any final partial block.
        Use ``numpy.concatenate()`` to join numeric result arrays. The flame length
        calculation returns a coefficient tuple per block when ``params_only=True``.
    """
    backend = _backend.resolve_backend(backend)
    function_name, row_count, num_processors = _blocks.prepare_blocks(flame_function, num_processors, kwargs)
    if num_processors < 2:
        num_processors = 2
        warnings.warn('Multiprocessing requires at least two cores.\n'
                      'Defaulting num_processors to 2 for this run',
                      UserWarning,
                      stacklevel=2)

    return _blocks.calculate_blocks(function_name, row_count, num_processors, block_size, backend, kwargs)
