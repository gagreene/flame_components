# -*- coding: utf-8 -*-
"""
Created on Wed Oct  11 13:25:52 2023

@author: Gregory A. Greene
"""

import numbers
from numpy import ma, ndarray, nan, isnan
from numpy.ma import sin, arccos, arcsin, arctan, sqrt, log, power
from numpy import pi, degrees, radians
from typing import Union, Optional
# import inspect
import multiprocessing as mp
# from multiprocessing import current_process


# FUNCTION TO CALCULATE MID-FLAME WIND SPEED
def getMidFlameWS(wind_speed: Union[int, float, ndarray],
                  canopy_cover: Union[int, float, ndarray],
                  canopy_ht: Union[int, float, ndarray],
                  canopy_baseht: Union[int, float, ndarray],
                  units: str = 'SI') -> Union[int, float, ndarray]:
    """
    Function to calculate mid-flame wind speed
    :param wind_speed: wind speed; if units == "SI": 10m wind speed (km/h); if units == "IMP": 20ft wind speed (mi/h)
    :param canopy_cover: canopy cover (percent)
    :param canopy_ht: stand ht (m or ft)
    :param canopy_baseht: canopy base height (m or ft)
    :param units: units of input data ("SI" or "IMP")
        SI = metric (10-m ws in km/h, ht in m, cbh in m)
        IMP = imperial (20-ft ws in mi/h, ht in ft, cbh in ft)
    :return: mid-flame windspeed (m/s)

    Divide by 3.6 to convert from km/h to m/s\n
    Divide windspeed by 1.15 to convert from 10-m to 20-ft equivalent (Lawson and Armitage 2008)\n
    Calculate mid-flame windspeed (m/s) using under canopy equations (Albini and Baughman 1979)
          Uc/Uh = 0.555 / (sqrt(f*H) * ln((20 + 0.36*H) / (0.13*H)))\n
          where f = crown ratio * canopy cover (proportion) / 3, H = stand height (ft)\n
    Equations explained well in Andrews (2012) - Modeling wind adjustement factor and
    midflame wind speed for Rothermel's surface fire spread model\n
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, ndarray) for data in [wind_speed, canopy_cover, canopy_ht, canopy_baseht]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify windspeed
    if not isinstance(wind_speed, (numbers.Real, ndarray)):
        raise TypeError('windspeed must be either int, float or numpy ndarray data types')
    elif isinstance(wind_speed, ndarray):
        wind_speed = ma.array(wind_speed, mask=isnan(wind_speed))
    elif isinstance(wind_speed, numbers.Real):
        wind_speed = ma.array([float(wind_speed)], mask=isnan([wind_speed]))

    # Verify canopy_cover
    if not isinstance(canopy_cover, (numbers.Real, ndarray)):
        raise TypeError('canopy_cover must be either int, float or numpy ndarray data types')
    elif isinstance(canopy_cover, ndarray):
        canopy_cover = ma.array(canopy_cover, mask=isnan(canopy_cover))
    elif isinstance(canopy_cover, numbers.Real):
        canopy_cover = ma.array([float(canopy_cover)], mask=isnan([canopy_cover]))

    # Verify canopy_ht
    if not isinstance(canopy_ht, (numbers.Real, ndarray)):
        raise TypeError('canopy_ht must be either int, float or numpy ndarray data types')
    elif isinstance(canopy_ht, ndarray):
        canopy_ht = ma.array(canopy_ht, mask=isnan(canopy_ht))
    elif isinstance(canopy_ht, numbers.Real):
        canopy_ht = ma.array([float(canopy_ht)], mask=isnan([canopy_ht]))

    # Verify canopy_baseht
    if not isinstance(canopy_baseht, (numbers.Real, ndarray)):
        raise TypeError('canopy_baseht must be either int, float or numpy ndarray data types')
    elif isinstance(canopy_baseht, ndarray):
        canopy_baseht = ma.array(canopy_baseht, mask=isnan(canopy_baseht))
    elif isinstance(canopy_baseht, numbers.Real):
        canopy_baseht = ma.array([float(canopy_baseht)], mask=isnan([canopy_baseht]))

    # Verify units
    if not isinstance(units, str):
        raise TypeError('The "units" parameter must be a str data type')
    elif units not in ['SI', 'IMP']:
        raise ValueError('The "units" parameter must be either "SI" or "IMP"')

    # Convert input units
    if units == 'SI':
        wind_speed = wind_speed / (3.6 * 1.15)  # convert 10m (km/hr) windspeed to 20ft equivalent (1.15) and m/s (3.6)
        canopy_ht = canopy_ht * 3.28084  # convert height in meters to feet
        canopy_baseht = canopy_baseht * 3.28084  # convert cbh in meters to feet
    elif units == 'IMP':
        wind_speed = wind_speed / 2.23694  # convert mi/h to m/s
    crown_ratio = (canopy_ht - canopy_baseht) / canopy_ht  # calculate crown ratio
    f = crown_ratio * canopy_cover / 300

    canopy_ht = ma.where(canopy_ht == 0,
                         0.5 * 3.28084,
                         canopy_ht)

    # Calculate the mid-flame wind speed
    midflame_ws = ma.where(f <= 5,
                           # Calculate unsheltered midflame windspeed
                           # ws * 0.4
                           wind_speed * 1.83 / log((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht)),
                           # Calculate sheltered midflame windspeed
                           wind_speed * 0.555 / (
                                   sqrt(f * canopy_ht) * log((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht))))

    # Ensure midflame_ws >= 0
    midflame_ws[midflame_ws < 0] = 0

    if return_array:
        return midflame_ws.data
    else:
        return midflame_ws.data[0]


# FUNCTION TO CALCULATE FLAME LENGTH
def getFlameLength(model: str,
                   fire_intensity: Union[int, float, ndarray],
                   flame_depth: Optional[Union[int, float, ndarray]] = None,
                   params_only: bool = False) -> Union[int, float, ndarray]:
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
        'Barbon_iNOWIND': (0.062000, 0.5336),  # Yuana and Cox (1996); Source = Pine needles; Lab + Field
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

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, ndarray) for data in [fire_intensity, flame_depth]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify model
    if not isinstance(model, str):
        raise TypeError('The "model" parameter must be a str data type')
    elif model not in list(model_dict.keys()):
        raise ValueError(f'The "model" parameter must be one of the following:\n'
                         f'{list(model_dict.keys())}')

    # Verify inputs for the selected model are valid
    if model == 'Finney_HEAD':
        if any(isinstance(data, type(None)) for data in [fire_intensity, flame_depth]):
            raise ValueError('The "Finney_HEAD" model requires "fire_intensity" and "flame_depth" as inputs')

    # Verify fire_intensity
    if not isinstance(fire_intensity, (numbers.Real, ndarray)):
        raise TypeError('fire_intensity must be either int, float or numpy ndarray data types')
    elif isinstance(fire_intensity, ndarray):
        fire_intensity = ma.array(fire_intensity, mask=isnan(fire_intensity))
    else:  # isinstance(fire_intensity, numbers.Real):
        fire_intensity = ma.array([float(fire_intensity)], mask=isnan([fire_intensity]))

    # Verify flame_depth
    if not isinstance(flame_depth, (numbers.Real, ndarray, type(None))):
        raise TypeError('flame_depth must be either int, float or numpy ndarray data types')
    elif isinstance(flame_depth, ndarray):
        flame_depth = ma.array(flame_depth, mask=isnan(flame_depth))
    elif isinstance(flame_depth, numbers.Real):
        flame_depth = ma.array([float(flame_depth)], mask=isnan([flame_depth]))

    # Verify params_only
    if not isinstance(params_only, bool):
        raise TypeError('The "params_only" parameter must be bool data type')

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

    # Ensure fl >= 0
    fl[fl < 0] = 0

    if return_array:
        return fl.data
    else:
        return fl.data[0]


# FUNCTION TO CALCULATE FLAME HEIGHT
def getFlameHeight(model: str,
                   flame_length: Union[int, float, ndarray],  # For all models
                   fire_type: Optional[Union[str, int]] = None,  # For Nelson model
                   fire_intensity: Optional[Union[int, float, ndarray]] = None,  # For Nelson model
                   midflame_ws: Optional[Union[int, float, ndarray]] = None,  # For Nelson model
                   flame_tilt: Optional[Union[int, float, ndarray]] = None,  # For Finney model
                   slope_angle: Optional[Union[int, float, ndarray]] = None,  # For Finney model
                   slope_units: Optional[str] = None  # For Finney model
                   ) -> Union[int, float, ndarray]:
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
    # Create the fire type dictionary to convert string inputs to integer
    fire_type_dict = {
        'surface': 1,
        'passive crown': 2,
        'active crown': 3
    }

    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, ndarray) for data in [flame_length, fire_type, fire_intensity,
                                                  midflame_ws, flame_tilt, slope_angle, slope_units]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify model
    if not isinstance(model, str):
        raise TypeError('The "model" parameter must be a str data type')
    elif model not in ['Nelson', 'Finney']:
        raise ValueError(f'The "model" parameter must be one of the following: "Nelson", "Finney"')

    # Verify inputs for the selected model are valid
    if model == 'Nelson':
        if any(isinstance(data, type(None)) for data in [fire_type, fire_intensity, midflame_ws]):
            raise ValueError('The "Nelson" model requires "fire_type", "fire_intensity", and "midflame_ws" as inputs')
    elif model == 'Finney':
        if any(isinstance(data, type(None)) for data in [flame_tilt, slope_angle, slope_units]):
            raise ValueError('The "Finney" model requires "flame_tilt", "slope_angle", and "slope_units" as inputs')

    # Verify flame_length
    if not isinstance(flame_length, (numbers.Real, ndarray)):
        raise TypeError('flame_length must be either int, float or numpy ndarray data types')
    elif isinstance(flame_length, ndarray):
        flame_length = ma.array(flame_length, mask=isnan(flame_length))
    elif isinstance(flame_length, numbers.Real):
        flame_length = ma.array([float(flame_length)], mask=isnan([flame_length]))

    # Verify fire_type
    if model == 'Nelson':
        if not isinstance(fire_type, (numbers.Real, float, ndarray, type(None))):
            raise TypeError('fire_type must be either None, str, int or numpy ndarray data types')
        elif isinstance(fire_type, (str, type(None))):
            if fire_type not in ['surface', 'passive crown', 'active crown']:
                raise ValueError(f'The "fire_type" parameter must be one of the following: '
                                 f'"surface", "passive crown", "active crown"')
            else:
                fire_type = fire_type_dict.get(fire_type, nan)  # Convert fire_type to integer value
        if isinstance(fire_type, ndarray):
            fire_type = ma.array(fire_type, mask=isnan(fire_type))
        elif isinstance(midflame_ws, numbers.Real):
            fire_type = ma.array([float(fire_type)], mask=isnan([fire_type]))

    # Verify fire_intensity
    if not isinstance(fire_intensity, (numbers.Real, ndarray, type(None))):
        raise TypeError('fire_intensity must be either iNone, nt, float or numpy ndarray data types')
    elif isinstance(fire_intensity, ndarray):
        fire_intensity = ma.array(fire_intensity, mask=isnan(fire_intensity))
    elif isinstance(fire_intensity, numbers.Real):
        fire_intensity = ma.array([float(fire_intensity)], mask=isnan([fire_intensity]))

    # Verify midflame_ws
    if not isinstance(midflame_ws, (numbers.Real, ndarray, type(None))):
        raise TypeError('midflame_ws must be either None, int, float or numpy ndarray data types')
    elif isinstance(midflame_ws, ndarray):
        midflame_ws = ma.array(midflame_ws, mask=isnan(midflame_ws))
    elif isinstance(midflame_ws, numbers.Real):
        midflame_ws = ma.array([float(midflame_ws)], mask=isnan([midflame_ws]))

    # Verify flame_tilt
    if not isinstance(flame_tilt, (numbers.Real, ndarray, type(None))):
        raise TypeError('flame_tilt must be either None, int, float or numpy ndarray data types')
    elif isinstance(flame_tilt, ndarray):
        flame_tilt = ma.array(flame_tilt, mask=isnan(flame_tilt))
    elif isinstance(flame_tilt, numbers.Real):
        flame_tilt = ma.array([float(flame_tilt)], mask=isnan([flame_tilt]))

    # Verify slope_angle
    if not isinstance(slope_angle, (numbers.Real, ndarray, type(None))):
        raise TypeError('slope_angle must be either None, int, float or numpy ndarray data types')
    elif isinstance(slope_angle, ndarray):
        slope_angle = ma.array(slope_angle, mask=isnan(slope_angle))
    elif isinstance(slope_angle, numbers.Real):
        slope_angle = ma.array([slope_angle], mask=isnan([slope_angle]))

    # Verify slope_units
    if not isinstance(slope_units, (str, type(None))):
        raise TypeError('slope_units must be str or None data types')
    elif (slope_units is not None) and (slope_units not in ['degrees', 'percent']):
        raise ValueError(f'The "slope_units" parameter must be one of the following: "degrees", "percent"')

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
            raise Exception('Unable to calculate flame height - Slope tilt')

        # Convert flame tilt so it is relative to horizontal
        tilt_h = pi / 2 - radians(flame_tilt)

        height = ma.where(slope_angle <= 1,
                          flame_length * sin(tilt_h),
                          # Calculate Finney and Martin (1992) flame height
                          flame_length * sin(tilt_h - slope_rad) / sin(radians(90) - slope_rad))

    # Ensure height >= 0
    height[height < 0] = 0

    if return_array:
        return height.data
    else:
        return height.data[0]


# FUNCTION TO CALCULATE FLAME TILT
def getFlameTilt(model: str,
                 flame_length: Optional[Union[int, float, ndarray]] = None,
                 flame_height: Optional[Union[int, float, ndarray]] = None,
                 slope_angle: Optional[Union[int, float, ndarray]] = None,
                 slope_units: Optional[str] = None,
                 wind_speed: Optional[Union[int, float, ndarray]] = None,
                 wind_speed_units: Optional[str] = None,
                 canopy_ht: Optional[Union[int, float, ndarray]] = None) -> Union[int, float, ndarray]:
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
    if any(isinstance(data, ndarray) for data in [flame_length, flame_height, slope_angle, wind_speed, canopy_ht]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify model
    if not isinstance(model, str):
        raise TypeError('The "model" parameter must be a str data type')
    elif model not in ['Standard', 'Finney', 'Butler']:
        raise ValueError(f'The "model" parameter must be one of the following: "Nelson", "Finney"')

    # Verify inputs for the selected model are valid
    if model == 'Standard':
        if any(isinstance(data, type(None)) for data in [flame_length, flame_height]):
            raise ValueError('The "Standard" model requires "flame_length" and "flame_height" as inputs')
    elif model == 'Finney':
        if any(isinstance(data, type(None)) for data in [flame_length, flame_height, slope_angle, slope_units]):
            raise ValueError('The "Finney" model requires "flame_length", "flame_height", '
                             '"slope_angle", and "slope_units" as inputs')
    else:  # model == 'Butler':
        if any(isinstance(data, type(None)) for data in [wind_speed, wind_speed_units, canopy_ht]):
            raise ValueError('The "Butler" model requires "windspeed", "windspeed_units", and "canopy_ht" as inputs')

    # Verify flame_length
    if not isinstance(flame_length, (numbers.Real, ndarray, type(None))):
        raise TypeError('flame_length must be either None, int, float or numpy ndarray data types')
    elif isinstance(flame_length, ndarray):
        flame_length = ma.array(flame_length, mask=isnan(flame_length))
    elif isinstance(flame_length, numbers.Real):
        flame_length = ma.array([flame_length], mask=isnan([flame_length]))

    # Verify flame_height
    if not isinstance(flame_height, (numbers.Real, ndarray, type(None))):
        raise TypeError('flame_height must be either None, int, float or numpy ndarray data types')
    elif isinstance(flame_height, ndarray):
        flame_height = ma.array(flame_height, mask=isnan(flame_height))
    elif isinstance(flame_height, numbers.Real):
        flame_height = ma.array([flame_height], mask=isnan([flame_height]))

    # Verify slope_angle
    if not isinstance(slope_angle, (numbers.Real, ndarray, type(None))):
        raise TypeError('slope_angle must be either None, int, float or numpy ndarray data types')
    elif isinstance(slope_angle, ndarray):
        slope_angle = ma.array(slope_angle, mask=isnan(slope_angle))
    elif isinstance(slope_angle, numbers.Real):
        slope_angle = ma.array([slope_angle], mask=isnan([slope_angle]))

    # Verify slope_units
    if not isinstance(slope_units, (str, type(None))):
        raise TypeError('slope_units must be str or None data types')
    elif slope_units not in ['degrees', 'percent', None]:
        raise ValueError(f'The "slope_units" parameter must be one of the following: "degrees", "percent"')

    # Verify wind_speed
    if not isinstance(wind_speed, (numbers.Real, ndarray, type(None))):
        raise TypeError('wind_speed must be either None, int, float or numpy ndarray data types')
    elif isinstance(wind_speed, ndarray):
        wind_speed = ma.array(wind_speed, mask=isnan(wind_speed))
    elif isinstance(wind_speed, numbers.Real):
        wind_speed = ma.array([wind_speed], mask=isnan([wind_speed]))

    # Verify wind_speed_units
    if not isinstance(wind_speed_units, (str, type(None))):
        raise TypeError('wind_speed_units must be str or None data types')
    elif wind_speed_units not in ['kph', 'mps', 'mph', None]:
        raise ValueError(f'The "wind_speed_units" parameter must be one of the following: "kph", "mps", "mph')

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
            raise Exception('Unable to calculate flame tilt - Invalid slope units provided')

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

    # Ensure tilt_v >= 0
    tilt_v[tilt_v < 0] = 0

    # Return flame tilt angle relative to vertical (degrees)
    if return_array:
        return degrees(tilt_v).data
    else:
        return degrees(tilt_v).data[0]


# FUNCTION TO CALCULATE FLAME RESIDENCE TIME
def getFlameResidenceTime(ros: Union[int, float, ndarray],
                          fuel_consumption: Union[int, float, ndarray],
                          midflame_ws: Union[int, float, ndarray],
                          units: str) -> Union[int, float, ndarray]:
    """
    Function to calculate flame residence time using equation from Nelson and Adkins (1988)
    :param ros: Fire rate of spread (m/min)
    :param fuel_consumption: Amount of fuel consumed by fire front (kg/m^2)
    :param midflame_ws: Mid-flame windspeed (m/s)
    :param units: return flame residence time in seconds or minutes ("sec", "min")
    :return: Flame residence time (seconds or minutes)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, ndarray) for data in [ros, fuel_consumption, midflame_ws]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify ros
    if not isinstance(ros, (numbers.Real, ndarray)):
        raise TypeError('ros must be either int, float or numpy ndarray data types')
    elif isinstance(ros, ndarray):
        ros = ma.array(ros, mask=isnan(ros))
    else:  # if isinstance(ros, numbers.Real):
        ros = ma.array([ros], mask=isnan([ros]))

    # Verify fuel_consumption
    if not isinstance(fuel_consumption, (numbers.Real, ndarray)):
        raise TypeError('fuel_consumption must be either int, float or numpy ndarray data types')
    elif isinstance(fuel_consumption, ndarray):
        fuel_consumption = ma.array(fuel_consumption, mask=isnan(fuel_consumption))
    else:  # isinstance(fuel_consumption, numbers.Real):
        fuel_consumption = ma.array([fuel_consumption], mask=isnan([fuel_consumption]))

    # Verify midflame_ws
    if not isinstance(midflame_ws, (numbers.Real, ndarray)):
        raise TypeError('fuel_consumption must be either int, float or numpy ndarray data types')
    elif isinstance(midflame_ws, ndarray):
        midflame_ws = ma.array(midflame_ws, mask=isnan(midflame_ws))
    else:  # isinstance(midflame_ws, numbers.Real):
        midflame_ws = ma.array([midflame_ws], mask=isnan([midflame_ws]))

    # Calculate flame residence time
    res_time = (0.39 * power(fuel_consumption, 0.25) * power(midflame_ws, 1.51)) / (ros / 60)
    if units == 'min':
        res_time = res_time / 60

    # Ensure res_time >= 0
    res_time[res_time < 0] = 0

    # Return flame residence time
    if return_array:
        return res_time.data
    else:
        return res_time.data[0]


# FUNCTION TO CALCULATE FLAME DEPTH
def getFlameDepth(ros: Union[int, float, ndarray],
                  res_time: Union[int, float, ndarray]) -> Union[int, float, ndarray]:
    """
    Calculate flame depth using equation from Fons et al. (1963)
    :param ros: Fire rate of spread (m/min)
    :param res_time: Time from initial temp rise to the time of definite drop after reaching peak temp (min).
        Definition per Rothermel and Deeming (1980)
    :return: flame depth (m)
    """
    # ### CHECK FOR NUMPY ARRAYS IN INPUT PARAMETERS
    if any(isinstance(data, ndarray) for data in [ros, res_time]):
        return_array = True
    else:
        return_array = False

    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify ros
    if not isinstance(ros, (numbers.Real, ndarray)):
        raise TypeError('ros must be either int, float or numpy ndarray data types')
    elif isinstance(ros, ndarray):
        ros = ma.array(ros, mask=isnan(ros))
    else:  # isinstance(ros, numbers.Real):
        ros = ma.array([ros], mask=isnan([ros]))

    # Verify res_time
    if not isinstance(res_time, (numbers.Real, ndarray)):
        raise TypeError('res_time must be either int, float or numpy ndarray data types')
    elif isinstance(res_time, ndarray):
        res_time = ma.array(res_time, mask=isnan(res_time))
    else:  # isinstance(res_time, numbers.Real):
        res_time = ma.array([res_time], mask=isnan([res_time]))

    # Calculate flame depth
    fd = ros * res_time

    # Ensure fd >= 0
    fd[fd < 0] = 0

    # Return flame depth
    if return_array:
        return fd.data
    else:
        return fd.data[0]


def _gen_blocks(array, block_size, stride):
    """
    Function to generate blocks
    :param array: The array to process
    :param block_size: The size of each block
    :param stride:
    :return:
    """
    num_blocks = (array.shape[0] - block_size) // stride + 1
    blocks = [array[i * stride:i * stride + block_size] for i in range(num_blocks)]
    positions = [(i * stride, (i * stride + block_size)) for i in range(num_blocks)]
    return blocks, positions


def _estimate_optimal_block_size(array_shape: tuple,
                                 num_processors: int) -> int:
    """
    Function to estimate optimal block size
    :param array_shape: Shape of the array being processed
    :param num_processors: Number of processors being used for multiprocessing
    :return: Estimated block size
    """
    # Estimate block size based on array shape and number of processors being used
    return array_shape[0] // num_processors


# TODO - Verify that this function works...
def flameComponent_ArrayMultiprocessing(flame_function: str,
                                        num_processors: int = 2,
                                        block_size: int = None,
                                        *kwargs) -> list:
    """
    Function breaks input arrays into blocks and processes each block with a different worker/processor.
    Uses the function requested in the "flame_function" parameter.

    **flame_function options**
        "midflame_ws", "flame_length", "flame_height",
        "flame_tilt", "flame_residence", "flame_depth"

    :param flame_function: The flame components function to implement.
    :param num_processors: Number of cores for multiprocessing
    :param block_size: Size of blocks (# raster cells) for multiprocessing.
        If block_size is None, an optimal block size will be estimated automatically.
    :param kwargs: A dictionary of inputs for the requested flame_components function.
        The dictionary keys must match the required input parameters for the requested function.
        Refer to the function docstring for parameter requirements.
    :return: Concatenated output array from all workers
    """
    flame_func_dict = {
        'midflame_ws': 'getMidFlameWS',
        'flame_length': 'getFlameLength',
        'flame_height': 'getFlameHeight',
        'flame_tilt': 'getFlameTilt',
        'flame_residence': 'getFlameResidence',
        'flame_depth': 'getFlameDepth'
    }
    # Get the function object from the global scope
    function_to_run = globals().get(flame_func_dict.get(flame_function))

    # Verify the function request
    if function_to_run is None:
        raise ValueError(f'Function for {flame_function} does not exist.'
                         f'The options are: {list(flame_func_dict.keys())}')

    # Extract array datasets from kwargs
    array_kwargs = {key: val for key, val in kwargs.items() if isinstance(val, ndarray)}
    array_list = list(array_kwargs.values())

    # Verify there is at least one input array
    if len(array_list) == 0:
        raise ValueError('Unable to use the multiprocessing function. There are no arrays in the kwargs.')

    # If more than one array, verify they are all the same shape
    if len(array_list) > 1:
        shapes = {arr.shape for arr in array_list}
        if len(shapes) > 1:
            raise ValueError(f'All arrays must have the same dimensions. '
                             f'The following range of dimensions exists: {shapes}')

    # Verify num_processors is greater than 1
    if num_processors < 2:
        num_processors = 2
        raise UserWarning('Multiprocessing requires at least two cores.\n'
                          'Defaulting num_processors to 2 for this run')

    # Verify block size
    if block_size is None:
        block_size = _estimate_optimal_block_size(array_shape=array_list[0].shape,
                                                  num_processors=num_processors)

    # Split input arrays into blocks and track their positions
    array_blocks = []
    block_positions = None  # Will hold the block positions from the first array

    for array in array_list:
        blocks, positions = _gen_blocks(array=array, block_size=block_size, stride=block_size)
        array_blocks.append(blocks)
        if block_positions is None:
            block_positions = positions

    # Generate final input_block list for multiprocessing
    input_blocks = []
    num_blocks = len(array_blocks[0])  # Number of blocks should be the same for all arrays

    for idx in range(num_blocks):
        block_set = [array_blocks[i][idx] for i in range(len(array_blocks))]
        row = {key: None for key in kwargs.keys()}

        # Assign blocks to the correct indices
        for i, block in zip(array_kwargs.keys(), block_set):
            row[i] = block

        # Assign non-array inputs
        for key, value in kwargs.items():
            if key not in array_kwargs:
                row[key] = value

        input_blocks.append((row, block_positions[idx]))  # Attach the position to each block

    # Define a wrapper for multiprocessing
    def worker(chunk):
        return function_to_run(**chunk)

    # Run the multiprocessing with starmap_async
    with mp.Pool(processes=num_processors) as pool:
        async_result = pool.starmap_async(worker, [(block[0],) for block in input_blocks])

        # Retrieve the results asynchronously
        results = async_result.get()

    return results
