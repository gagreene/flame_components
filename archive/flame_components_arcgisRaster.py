"""
Created on Wed Oct  11 13:25:52 2023

@author: Gregory A. Greene
"""


import arcpy
import numpy as np
from arcpy.sa import *


def getDegrees(in_data: float | arcpy.Raster) -> float | arcpy.Raster:
    return in_data * 180 / np.pi


def getRadians(in_data: float | arcpy.Raster) -> float | arcpy.Raster:
    return in_data * np.pi / 180


# FUNCTION TO CALCULATE MID-FLAME WIND SPEED
def getMidFlameWS(windspeed: arcpy.Raster,
                  canopy_cover: arcpy.Raster,
                  canopy_ht: arcpy.Raster,
                  canopy_baseht: arcpy.Raster,
                  units: str) -> arcpy.Raster:
    """
    Function calculates mid-flame wind speed
    :param windspeed: wind speed; if units == "SI": 10m wind speed (km/h); if units == "IMP"" 20ft wind speed (mi/h)
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
    if units == 'SI':
        windspeed /= (3.6 * 1.15)   # convert 10m (km/hr) windspeed to 20ft equivalent (1.15) and m/s (3.6)
        canopy_ht *= 3.28084        # convert height in meters to feet
        canopy_baseht *= 3.28084    # convert cbh in meters to feet
    elif units == 'IMP':
        windspeed /= 2.23694        # convert mi/h to m/s

    # calculate crown ratio
    crown_ratio = Con(canopy_ht == 0, 0, (canopy_ht - canopy_baseht) / canopy_ht)
    f = crown_ratio * canopy_cover / 300

    # Replace canopy height values of 0 with 0.5 m
    canopy_ht = Con(canopy_ht == 0, 0.5 * 3.28084, canopy_ht)

    return Con(f <= 5,
               windspeed * 1.83 / Ln((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht)),
               windspeed * 0.555 / (SquareRoot(f * canopy_ht) * Ln((20 + (0.36 * canopy_ht)) / (0.13 * canopy_ht)))
               )


# FUNCTION TO CALCULATE FLAME LENGTH
def getFlameLength(model: str,
                   fire_intensity: arcpy.Raster,
                   flame_depth: arcpy.Raster = None,
                   params_only: bool = False) -> arcpy.Raster:
    """
    Function to estimate flame length from a variety of published models.
    Equation from Nelson and Adkins (1986) - referenced in Cruz and Alexander (2018)
    :param model: flame length model (refer to "model_dict" for list of options)
        All models come from Finney and Grumstrup (2023), including their own 2023 model
    :param fire_intensity: head fire intensity (kW/m)
    :param flame_depth: head fire flame depth (m) [OPTIONAL]
        Only required for "Finney_HEAD" model
    :param params_only: return only the model parameters ("True" or "False"; Default = False)
    :return: flame length (m) or model parameters
    """
    #  Published correlations of flame length with fire intensity (from Finney and Grumstrup 2023)
    model_dict = {
        # Fire = No wind, flat
        'Fons_NOWIND': (0.024018, 2/3),         # Fons et al. (1963); Source = Cribs; Lab
        'Thomas_NOWIND': (0.026700, 2/3),       # Thomas (1963); Source = Cribs; Lab + Field
        'Yuana_NOWIND': (0.034000, 2/3),        # Yuana and Cox (1996); Source = Gas slot burner; Lab
        'Barbon_iNOWIND': (0.062000, 0.5336),   # Yuana and Cox (1996); Source = Pine needles; Lab + Field
        # Fire = Backing
        'Nelson_BACK': (0.027973, 2/3),         # Nelson (1980); Source = Needles; Lab + Field
        'Fernandes_BACK': (0.029000, 0.7240),   # Fernandes et al. (2009); Source = Pine Needles; Field
        'Clark_BACK': (0.001600, 1.7450),       # Clark (1983); Source = Grass; Field
        'Vega_BACK': (0.087000, 0.4930),        # Vega et al. (1998); Source = Shrubs; Field
        # Fire = Heading
        'Byram_HEAD': (0.0775, 0.4600),         # Byram (1959); Source = Needles; Field
        'Anderson1_HEAD': (0.013876, 0.6510),   # Anderson et al. (1966); Source = Lodgepole pine slash; Field
        'Anderson2_HEAD': (0.008800, 0.6700),   # Anderson et al. (1996); Source = Douglas-fir slash; Field
        'Newman_HEAD': (0.05770, 0.5000),       # Newman (1974); Source = Unkn; Field
        'Sneewujagt_HEAD': (0.037680, 0.5000),  # Sneewujagt and Frandsen (1977); Source = Needles; Field
        'Nelson1_HEAD': (0.044230, 0.5000),     # Nelson (1980); Source = Needles; Field
        'Clark_HEAD': (0.000722, 0.9934),       # Clark (1983); Source = Grass; Field
        'Nelson2_HEAD': (0.047500, 0.4930),     # Nelson and Adkins (1986); Source = Needles/Palmetto; Lab + Field
        'VanWilgen_HEAD': (0.046000, 0.4128),   # Van Wilgen (1986); Source = Grass; Field
        'Burrows_HEAD': (0.040480, 0.5740),     # Burrows (1994, p. 102); Source = Needles; Field
        'MarsdenSmedley_HEAD': (0.148, 0.403),  # Marsden-Smedley and Catchpole (1995); Source = Button grass; Field
        'Weise1_HEAD': (0.016000, 0.7000),      # Weise and Biging (1996); Source = Excelsior & birch stir sticks; Lab
        'Catchpole_HEAD': (0.032500, 0.5600),   # Catchpole et al. (1998); Source = Heath; Field
        'Fernandes1_HEAD': (0.051600, 0.4530),  # Fernandes et al. (2000); Source = Shrubs; Field
        'Butler_HEAD': (0.017500, 2/3),         # Butler et al. (2004); Source = Crownfire (add to avg. stand ht); Unkn
        'Fernandes_HEAD': (0.045000, 0.5430),   # Fernandes et al. (2009); Source = Needles; Field
        'Nelson3_HEAD': (0.014200, 2/3),        # Nelson et al. (2012); Source = Needles/southern Fuel; Lab
        'Nelson4_HEAD': (0.015500, 2/3),        # Nelson et al. (2012); Source = Needles/southern Fuel; Field
        'Weise2_HEAD': (0.2000000, 0.3400),     # Weise et al. (2016); Source = Chaparral; Lab
        'Davies_HEAD': (0.220000, 0.2900),      # Davies et al. (2019); Source = Heathlands; Field
        'Finney_HEAD': (0.01051, 0.774, 0.161)  # Finney and Grumstrup (2023); Source = Gas slot burner; Lab
    }

    # Get model parameters
    model_params = model_dict.get(model)

    if model_params is None:
        raise ValueError('Unable to calculate flame length - Model choice is invalid')

    if params_only:
        return model_params

    if model == 'Finney_HEAD':
        return (model_params[0] *
                Power(fire_intensity, model_params[1]) /
                Power(flame_depth, model_params[2]))
    else:
        return model_params[0] * Power(fire_intensity, model_params[1])


# FUNCTION TO CALCULATE FLAME HEIGHT
def getFlameHeight(model: str,
                   flame_length: arcpy.Raster,
                   fire_type: arcpy.Raster = None,          # Only for Nelson model
                   fire_intensity: arcpy.Raster = None,     # Only for Nelson model
                   midflame_ws: arcpy.Raster = None,        # Only for Nelson model
                   flame_tilt: arcpy.Raster = None,         # Only for Finney model
                   slope_angle: arcpy.Raster = None,        # Only for Finney model
                   slope_units: str | None = None                  # Only for Finney model
                   ) -> arcpy.Raster:
    """
    Equations from Nelson and Adkins (1986) or Finney and Martin (1992) - referenced in Cruz and Alexander (2018)
    :param model: model used to estimate flame height ("Nelson", "Finney")
        Simard is suggested if you already know tilt angle
    :param fire_type: [Nelson model only] type of fire ("surface", "passive crown", "active crown")
    :param fire_intensity: [Nelson model only] head fire intensity (kW/m)
    :param midflame_ws: [Nelson model only] mid-flame wind speed (m/s)
    :param flame_length: [Finney model only] head fire flame length (m)
    :param flame_tilt: [Finney model only] head fire flame tilt relative to vertical (degrees)
    :param slope_angle: [Finney model only] Slope angle of ground (degrees or percent)
    :param slope_units: [Finney model only] Units of slope-angle ("degrees" or "percent")
    :return: head fire flame height (m)
    """
    if model == 'Nelson':
        a = Con((fire_type == 1) | (fire_type == 2),
                1 / 360,    # parameter for experimental lab and field fires (Nelson and Adkins 1986; Nelson et al. 2012)
                Con(fire_type == 3,
                    0.0175,     # parameter for crown fires (Butler et al. 2004)
                    0))

        # Calculate height
        height = Con(midflame_ws == 0, flame_length, a * fire_intensity / midflame_ws)

        # Rescale height to match flame length if it is predicted to exceed flame length
        return Con(height > flame_length, flame_length, height)
    elif model == 'Finney':
        # Convert slope to radians
        if slope_units == 'percent':
            slope_rad = ATan(slope_angle / 100)
        elif slope_units == 'degrees':
            slope_rad = getRadians(slope_angle)
        else:
            raise ValueError('Unable to calculate flame height - Slope tilt')

        # Convert flame tilt so it is relative to horizontal
        tilt_h = np.pi / 2 - getRadians(flame_tilt)

        # Calculate Finney and Martin (1992) flame height
        return Con(slope_angle <= 1,
                   flame_length * Sin(tilt_h),
                   flame_length * Sin(tilt_h - slope_rad) / Sin(getRadians(90) - slope_rad))
    else:
        raise ValueError('Unable to calculate flame height - Model choice is invalid')


# FUNCTION TO CALCULATE FLAME TILT
def getFlameTilt(model: str,
                 flame_length: arcpy.Raster = None,
                 flame_height: arcpy.Raster = None,
                 slope_angle: arcpy.Raster = None,
                 slope_units: str | None = None,
                 windspeed: arcpy.Raster = None,
                 windspeed_units: str | None = None,
                 canopy_ht: arcpy.Raster = None) -> arcpy.Raster:
    """
    Function calculates flame tilt using Finney and Martin (1992) and Butler et al. (2004) equations
    :param model: Flame tilt model to use ("Finney", "Butler")
        Simard = Finney and Martin (1992) model
        Butler = Butler et al. (2004) model
    :param flame_length: [Standard & Finney models only]
        Head fire flame length (m)
    :param flame_height: [Standard & Finney models only]
        Head fire flame height (m)
    :param slope_angle: [Finney model only]
        Slope angle of ground (degrees or percent)
    :param slope_units: [Finney model only]
        Units of slope-angle ("degrees" or "percent")
    :param windspeed: [Butler model only]
        10m wind speed (i.e., measured 10m above open ground or forest canopy)
    :param windspeed_units: [Butler model only]
        Units of "windspeed" parameter ("kph", "mps", "mph")
            kph = kilometers per hour
            mps = meters per second
            mph = miles per hour
    :param canopy_ht: [Butler model only]
        Height of the canopy above the ground (m)
    :return: angle of head fire flame tilt (degrees)
    """

    # Calculate flame tilt angle (radians)
    if model == 'Standard':
        tilt_v = ACos(flame_height/flame_length)
    elif model == 'Finney':
        # Convert slope to radians
        if slope_units == 'percent':
            slope_rad = ATan(slope_angle / 100)
        elif slope_units == 'degrees':
            slope_rad = getRadians(slope_angle)
        else:
            raise ValueError('Unable to calculate flame tilt - Invalid slope units provided')

        # Calculate Finney and Martin (1992) flame tilt angle
        if flame_height == flame_length:
            tilt_v = 0
        else:
            # This equation calculates tilt relative to horizontal (tilting up from horizontal flat ground)
            tilt_h = ASin(getRadians(flame_height * getDegrees(Sin(getRadians(90) - slope_rad)) / flame_length)) + slope_rad
            # Get equivalent tilt relative to vertical rather than horizontal (tilting down from vertical)
            tilt_v = np.pi/2 - tilt_h
    elif model == 'Butler':
        # THIS MODEL (Butler et al. 2004) IS MADE FOR TILT OF CROWN FIRES
        # ONLY REQUIRES 10m WIND SPEED AS AN INPUT
        # ONLY USE FOR CROWN FIRES, OTHERWISE TILT WILL BE TOO LOW
        if any(elem is None for elem in [windspeed, windspeed_units, canopy_ht]):
            raise ValueError('Unable to calculate "Butler" flame tilt - '
                            'Did not pass at least one of the required input parameters')

        # Convert wind speed units if necessary
        if windspeed_units == 'kph':
            windspeed = windspeed / 3.6         # convert kilometers/hour to meters/second
        elif windspeed_units == 'mph':
            windspeed = windspeed / 2.23694     # convert miles/hour to meters/second

        # Wind speed at the top of the forest canopy (Albini and Baughman 1979; Butler et al. 2004)
        uc = windspeed / (3.6 * (1 + Ln(1 + (28 / canopy_ht))))

        # acceleration of gravity (m/s^2)
        g = 9.81

        # Calculate Butler et al. (2004) flame tilt angle
        # This equation already calculates tilt relative to vertical rather than horizontal
        tilt_v = ATan(SquareRoot((3 * Power(uc, 3)) / (2 * g * 10)))
    else:
        raise ValueError('Unable to calculate flame tilt - Model choice is invalid')

    # Return flame tilt angle relative to vertical (degrees)
    return Con(tilt_v >= 0, getDegrees(tilt_v), 0)


# FUNCTION TO CALCULATE FLAME RESIDENCE TIME
def getFlameResidenceTime(ros: float | arcpy.Raster,
                          fuel_consumption: float | arcpy.Raster,
                          midflame_ws: float | arcpy.Raster,
                          units: str) -> float | arcpy.Raster:
    """
    Function to calculate flame residence time using equation from Nelson and Adkins (1988)
    :param ros: Fire rate of spread (m/min)
    :param fuel_consumption: Amount of fuel consumed by fire front (kg/m^2)
    :param midflame_ws: Mid-flame windspeed (m/s)
    :param units: return flame residence time in seconds or minutes ("sec", "min")
    :return: Flame residence time (secods or minutes)
    """
    res_time = (0.39 * Power(fuel_consumption, 0.25) *
                Power(midflame_ws, 1.51)) / (ros / 60)
    if units == 'min':
        return res_time / 60
    else:
        return res_time


# FUNCTION TO CALCULATE FLAME DEPTH
def getFlameDepth(ros: float | arcpy.Raster,
                  res_time: float | arcpy.Raster):
    """
    Calculate flame depth using equation from Fons et al. (1963)
    :param ros: Fire rate of spread (m/min)
    :param res_time: Time from initial temp rise to the time of definite drop after reaching peak temp (min)
        Definition per Rothermel and Deeming (1980)
    :return: flame depth (m)
    """
    return ros * res_time
