"""
Created on Wed Oct  11 13:25:52 2023

@author: Gregory A. Greene

Flame Components — fire behavior calculations for surface and crown fires.

NaN handling: All functions accept numpy arrays containing NaN values. NaNs are
automatically masked via numpy.ma and propagate silently through calculations —
masked input cells produce masked (NaN) output cells without warning. To inspect
how many cells were masked, check ``result_array[numpy.isnan(result_array)].size``
on the returned array.
"""

import multiprocessing as mp
import warnings

from numpy import degrees, ma, ndarray, pi, radians
from numpy.ma import arccos, arcsin, arctan, log, power, sin, sqrt

from . import _utils


# FUNCTION TO CALCULATE MID-FLAME WIND SPEED
def get_mid_flame_ws(wind_speed: float | ndarray,
                  canopy_cover: float | ndarray,
                  canopy_ht: float | ndarray,
                  canopy_baseht: float | ndarray,
                  units: str = 'SI') -> int | float | ndarray:
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
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [wind_speed, canopy_cover, canopy_ht, canopy_baseht])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    wind_speed = _utils.numeric_array("wind_speed", wind_speed)
    canopy_cover = _utils.numeric_array("canopy_cover", canopy_cover)
    canopy_ht = _utils.numeric_array("canopy_ht", canopy_ht)
    canopy_baseht = _utils.numeric_array("canopy_baseht", canopy_baseht)
    _utils.check_choice("units", units, ("SI", "IMP"))

    # Convert input units
    if units == 'SI':
        wind_speed = wind_speed / (3.6 * 1.15)  # convert 10m (km/hr) windspeed to 20ft equivalent (1.15) and m/s (3.6)
        canopy_ht = canopy_ht * 3.28084  # convert height in meters to feet
        canopy_baseht = canopy_baseht * 3.28084  # convert cbh in meters to feet
    elif units == 'IMP':
        wind_speed = wind_speed / 2.23694  # convert mi/h to m/s

    if ma.any(canopy_ht == 0):
        warnings.warn(
            'One or more canopy_ht values are 0. These will be replaced with 1.6404 ft (0.5 m) '
            'to avoid division-by-zero in the Albini & Baughman (1979) equation. '
            'Note: by this point canopy_ht has been converted to feet internally.',
            UserWarning,
            stacklevel=2,
        )
    canopy_ht = ma.where(canopy_ht == 0,
                         0.5 * 3.28084,
                         canopy_ht)

    crown_ratio = (canopy_ht - canopy_baseht) / canopy_ht  # calculate crown ratio
    f = crown_ratio * canopy_cover / 300

    # Calculate the mid-flame wind speed
    midflame_ws = ma.where(f <= 5,
                           # Calculate unsheltered midflame windspeed
                           # ws * 0.4
                           wind_speed * 1.83 / log((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht)),
                           # Calculate sheltered midflame windspeed
                           wind_speed * 0.555 / (
                                   sqrt(f * canopy_ht) * log((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht))))

    return _utils.nonnegative_result(midflame_ws, return_array)


# FUNCTION TO CALCULATE FLAME LENGTH
def get_flame_length(model: str,
                   fire_intensity: float | ndarray,
                   flame_depth: float | ndarray | None = None,
                   params_only: bool = False) -> int | float | ndarray:
    """
    Function to estimate flame length from a variety of published models.
    Equation from Nelson and Adkins (1986) - referenced in Cruz and Alexander (2018)
    :param model: flame length model (refer to "model_dict" for list of options)
        All models come from Finney and Grumstrup (2023), including their own 2023 model.
    :param fire_intensity: head fire intensity (kW/m)
    :param flame_depth: head fire flame depth (m) [OPTIONAL]
        Only required for "Finney_HEAD" model
    :param params_only: return only the model parameters ("True" or "False"; Default = False)
    :return: flame length (m) or model parameters
    """
    #  Published correlations of flame length with fire intensity (from Finney and Grumstrup 2023)
    model_dict = {
        # Fire = No wind, flat
        'Fons_NOWIND': (0.024018, 2 / 3),  # Fons et al. (1963); Source = Cribs; Lab
        'Thomas_NOWIND': (0.026700, 2 / 3),  # Thomas (1963); Source = Cribs; Lab + Field
        'Yuana_NOWIND': (0.034000, 2 / 3),  # Yuana and Cox (1996); Source = Gas slot burner; Lab
        'Barbon_NOWIND': (0.062000, 0.5336),  # Yuana and Cox (1996); Source = Pine needles; Lab + Field
        # Fire = Backing
        'Nelson_BACK': (0.027973, 2 / 3),  # Nelson (1980); Source = Needles; Lab + Field
        'Fernandes_BACK': (0.029000, 0.7240),  # Fernandes et al. (2009); Source = Pine Needles; Field
        'Clark_BACK': (0.001600, 1.7450),  # Clark (1983); Source = Grass; Field
        'Vega_BACK': (0.087000, 0.4930),  # Vega et al. (1998); Source = Shrubs; Field
        # Fire = Heading
        'Byram_HEAD': (0.0775, 0.4600),  # Byram (1959); Source = Needles; Field
        'Anderson1_HEAD': (0.013876, 0.6510),  # Anderson et al. (1966); Source = Lodgepole pine slash; Field
        'Anderson2_HEAD': (0.008800, 0.6700),  # Anderson et al. (1996); Source = Douglas-fir slash; Field
        'Newman_HEAD': (0.05770, 0.5000),  # Newman (1974); Source = Unkn; Field
        'Sneewujagt_HEAD': (0.037680, 0.5000),  # Sneewujagt and Frandsen (1977); Source = Needles; Field
        'Nelson1_HEAD': (0.044230, 0.5000),  # Nelson (1980); Source = Needles; Field
        'Clark_HEAD': (0.000722, 0.9934),  # Clark (1983); Source = Grass; Field
        'Nelson2_HEAD': (0.047500, 0.4930),  # Nelson and Adkins (1986); Source = Needles/Palmetto; Lab + Field
        'VanWilgen_HEAD': (0.046000, 0.4128),  # Van Wilgen (1986); Source = Grass; Field
        'Burrows_HEAD': (0.040480, 0.5740),  # Burrows (1994, p. 102); Source = Needles; Field
        'MarsdenSmedley_HEAD': (0.148, 0.403),  # Marsden-Smedley and Catchpole (1995); Source = Button grass; Field
        'Weise1_HEAD': (0.016000, 0.7000),  # Weise and Biging (1996); Source = Excelsior & birch stir sticks; Lab
        'Catchpole_HEAD': (0.032500, 0.5600),  # Catchpole et al. (1998); Source = Heath; Field
        'Fernandes1_HEAD': (0.051600, 0.4530),  # Fernandes et al. (2000); Source = Shrubs; Field
        'Butler_HEAD': (0.017500, 2 / 3),  # Butler et al. (2004); Source = Crownfire (add to avg. stand ht); Unkn
        'Fernandes_HEAD': (0.045000, 0.5430),  # Fernandes et al. (2009); Source = Needles; Field
        'Nelson3_HEAD': (0.014200, 2 / 3),  # Nelson et al. (2012); Source = Needles/southern Fuel; Lab
        'Nelson4_HEAD': (0.015500, 2 / 3),  # Nelson et al. (2012); Source = Needles/southern Fuel; Field
        'Weise2_HEAD': (0.2000000, 0.3400),  # Weise et al. (2016); Source = Chaparral; Lab
        'Davies_HEAD': (0.220000, 0.2900),  # Davies et al. (2019); Source = Heathlands; Field
        'Finney_HEAD': (0.01051, 0.774, 0.161)  # Finney and Grumstrup (2023); Source = Gas slot burner; Lab
    }

    # Preserve the original selector for existing callers.
    model_dict['Barbon_iNOWIND'] = model_dict['Barbon_NOWIND']

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [fire_intensity, flame_depth])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    _utils.check_choice("model", model, model_dict)
    if model == "Finney_HEAD":
        _utils.require_inputs(model, fire_intensity=fire_intensity, flame_depth=flame_depth)
    fire_intensity = _utils.numeric_array("fire_intensity", fire_intensity)
    flame_depth = _utils.numeric_array("flame_depth", flame_depth, optional=True)
    _utils.check_bool("params_only", params_only)

    # Get model parameters
    model_params = model_dict.get(model)

    if params_only:
        return model_params

    if model == 'Finney_HEAD':
        fl = (model_params[0] *
              power(fire_intensity, model_params[1]) /
              power(flame_depth, model_params[2]))
    else:
        fl = model_params[0] * power(fire_intensity, model_params[1])

    return _utils.nonnegative_result(fl, return_array)


# FUNCTION TO CALCULATE FLAME HEIGHT
def get_flame_height(model: str,
                   flame_length: float | ndarray,  # For all models
                   fire_type: str | int | ndarray | None = None,  # For Nelson model
                   fire_intensity: float | ndarray | None = None,  # For Nelson model
                   midflame_ws: float | ndarray | None = None,  # For Nelson model
                   flame_tilt: float | ndarray | None = None,  # For Finney model
                   slope_angle: float | ndarray | None = None,  # For Finney model
                   slope_units: str | None = None  # For Finney model
                   ) -> int | float | ndarray:
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
    :return: head fire flame height (m)
    """
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

    if model == 'Nelson':
        # Calculate fire parameter (a)
        a = ma.where(ma.isin(fire_type, [1, 2]),
                     # parameter for experimental lab and field fires (Nelson and Adkins 1986; Nelson et al. 2012)
                     1 / 360,
                     # parameter for crown fires (Butler et al. 2004)
                     0.0175)

        # Calculate height
        height = ma.where(midflame_ws == 0,
                          flame_length,
                          a * fire_intensity / midflame_ws)

        # Rescale height to match flame length if it is predicted to exceed flame length
        height = ma.where(height > flame_length,
                          flame_length,
                          height)

    else:  # model == 'Finney':
        # Convert slope to radians
        if slope_units == 'percent':
            slope_rad = arctan(slope_angle / 100)
        elif slope_units == 'degrees':
            slope_rad = radians(slope_angle)
        else:
            raise ValueError('Unable to calculate flame height - Slope tilt')

        # Convert flame tilt so it is relative to horizontal
        tilt_h = pi / 2 - radians(flame_tilt)

        height = ma.where(slope_angle <= 1,
                          flame_length * sin(tilt_h),
                          # Calculate Finney and Martin (1992) flame height
                          flame_length * sin(tilt_h - slope_rad) / sin(radians(90) - slope_rad))

    return _utils.nonnegative_result(height, return_array)


# FUNCTION TO CALCULATE FLAME TILT
def get_flame_tilt(model: str,
                 flame_length: float | ndarray | None = None,
                 flame_height: float | ndarray | None = None,
                 slope_angle: float | ndarray | None = None,
                 slope_units: str | None = None,
                 wind_speed: float | ndarray | None = None,
                 wind_speed_units: str | None = None,
                 canopy_ht: float | ndarray | None = None) -> int | float | ndarray:
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
    :return: angle of head fire flame tilt (degrees)
    """
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

    # Calculate flame tilt angle (radians)
    if model == 'Standard':
        tilt_v = arccos(flame_height / flame_length)
    elif model == 'Finney':
        # Convert slope to radians
        if slope_units == 'percent':
            slope_rad = arctan(slope_angle / 100)
        elif slope_units == 'degrees':
            slope_rad = radians(slope_angle)
        else:
            raise ValueError('Unable to calculate flame tilt - Invalid slope units provided')

        # Calculate Finney and Martin (1992) flame tilt angle
        # This equation calculates tilt relative to horizontal (tilting up from horizontal flat ground)
        tilt_h = arcsin(radians(flame_height * degrees(sin(radians(90) - slope_rad)) / flame_length)) + slope_rad
        tilt_v = ma.where(flame_height == flame_length,
                          0,
                          # Get equivalent tilt relative to vertical rather than horizontal (tilting down from vertical)
                          pi / 2 - tilt_h)

    else:  # model == 'Butler':
        # THIS MODEL (Butler et al. 2004) IS MADE FOR TILT OF CROWN FIRES
        # ONLY REQUIRES 10m WIND SPEED AS AN INPUT
        # ONLY USE FOR CROWN FIRES, OTHERWISE TILT WILL BE TOO LOW

        # Convert wind speed units if necessary
        if wind_speed_units == 'kph':
            wind_speed = wind_speed / 3.6  # convert kilometers/hour to meters/second
        elif wind_speed_units == 'mph':
            wind_speed = wind_speed / 2.23694  # convert miles/hour to meters/second

        # Wind speed at the top of the forest canopy (Albini and Baughman 1979; Butler et al. 2004)
        uc = wind_speed / (3.6 * (1 + log(1 + (28 / canopy_ht))))

        # acceleration of gravity (m/s^2)
        g = 9.81

        # Calculate Butler et al. (2004) flame tilt angle
        # This equation already calculates tilt relative to vertical rather than horizontal
        tilt_v = arctan(sqrt((3 * power(uc, 3)) / (2 * g * 10)))

    return _utils.nonnegative_result(degrees(tilt_v), return_array)


# FUNCTION TO CALCULATE FLAME RESIDENCE TIME
def get_flame_residence_time(ros: float | ndarray,
                          fuel_consumption: float | ndarray,
                          midflame_ws: float | ndarray,
                          units: str) -> int | float | ndarray:
    """
    Function to calculate flame residence time using equation from Nelson and Adkins (1988).
    :param ros: Fire rate of spread (m/min). NOTE: input units are always m/min.
        The "units" parameter below controls the OUTPUT unit only.
    :param fuel_consumption: Amount of fuel consumed by fire front (kg/m^2)
    :param midflame_ws: Mid-flame wind speed (m/s)
    :param units: Output unit for flame residence time ("sec" or "min").
        This parameter does NOT affect the expected input unit for ros — ros must
        always be provided in m/min.
    :return: Flame residence time in seconds ("sec") or minutes ("min").
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [ros, fuel_consumption, midflame_ws])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    ros = _utils.numeric_array("ros", ros)
    fuel_consumption = _utils.numeric_array("fuel_consumption", fuel_consumption)
    midflame_ws = _utils.numeric_array("midflame_ws", midflame_ws)
    _utils.check_choice("units", units, ("sec", "min"))

    # Calculate flame residence time
    res_time = (0.39 * power(fuel_consumption, 0.25) * power(midflame_ws, 1.51)) / (ros / 60)
    if units == 'min':
        res_time = res_time / 60

    return _utils.nonnegative_result(res_time, return_array)


# FUNCTION TO CALCULATE FLAME DEPTH
def get_flame_depth(ros: float | ndarray,
                  res_time: float | ndarray) -> int | float | ndarray:
    """
    Calculate flame depth using equation from Fons et al. (1963)
    :param ros: Fire rate of spread (m/min)
    :param res_time: Time from initial temp rise to the time of definite drop after reaching peak temp (min).
        Definition per Rothermel and Deeming (1980)
    :return: flame depth (m)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    return_array = any(isinstance(data, ndarray) for data in [ros, res_time])

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    ros = _utils.numeric_array("ros", ros)
    res_time = _utils.numeric_array("res_time", res_time)

    # Calculate flame depth
    fd = ros * res_time

    return _utils.nonnegative_result(fd, return_array)


def _multiprocessing_worker(func_name: str, kwargs_dict: dict):
    """
    Module-level worker for flameComponent_ArrayMultiprocessing.
    Must be at module level (not a closure) so multiprocessing can pickle it.
    :param func_name: name of the flame_components function to call
    :param kwargs_dict: keyword arguments to pass to that function
    """
    func = globals().get(func_name)
    return func(**kwargs_dict)




def flame_component_array_multiprocessing(flame_function: str,
                                        num_processors: int = 2,
                                        block_size: int | None = None,
                                        **kwargs) -> list:
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
    :param kwargs: A dictionary of inputs for the requested flame_components function.
        The dictionary keys must match the required input parameters for the requested function.
        Refer to the function docstring for parameter requirements.
    :return: One result per block in input order, including any final partial block.
        Use ``numpy.concatenate()`` to join numeric result arrays. The flame length
        calculation returns a coefficient tuple per block when ``params_only=True``.
    """
    flame_func_dict = {
        'midflame_ws': 'get_mid_flame_ws',
        'flame_length': 'get_flame_length',
        'flame_height': 'get_flame_height',
        'flame_tilt': 'get_flame_tilt',
        'flame_residence': 'get_flame_residence_time',
        'flame_depth': 'get_flame_depth'
    }
    _utils.check_choice("flame_function", flame_function, flame_func_dict)
    row_count = _utils.common_row_count(kwargs)
    num_processors = _utils.check_int("num_processors", num_processors)

    # Verify num_processors is greater than 1
    if num_processors < 2:
        num_processors = 2
        warnings.warn('Multiprocessing requires at least two cores.\n'
                      'Defaulting num_processors to 2 for this run',
                      UserWarning,
                      stacklevel=2)

    if block_size is None:
        block_size = max(1, row_count // num_processors)
    else:
        block_size = _utils.check_int("block_size", block_size, minimum=1)

    fn_name = flame_func_dict[flame_function]
    tasks = []
    for start in range(0, row_count, block_size):
        rows = slice(start, start + block_size)
        block = {key: value[rows] if isinstance(value, ndarray) else value
                 for key, value in kwargs.items()}
        tasks.append((fn_name, block))

    with mp.Pool(processes=min(num_processors, len(tasks))) as pool:
        return pool.starmap(_multiprocessing_worker, tasks)
