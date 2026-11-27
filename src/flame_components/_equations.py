"""NumPy reference equations operating on validated, normalized inputs."""

from numpy import degrees, ma, pi, radians
from numpy.ma import arccos, arcsin, arctan, log, power, sin, sqrt

# Published correlations of flame length with fire intensity (from Finney and Grumstrup 2023)
LENGTH_MODELS = {
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


def prepare_mid_flame_ws(wind_speed, canopy_ht, canopy_baseht, units):
    # Convert input units
    if units == 'SI':
        wind_speed = wind_speed / (3.6 * 1.15)  # convert 10m (km/hr) windspeed to 20ft equivalent (1.15) and m/s (3.6)
        canopy_ht = canopy_ht * 3.28084  # convert height in meters to feet
        canopy_baseht = canopy_baseht * 3.28084  # convert cbh in meters to feet
    elif units == 'IMP':
        wind_speed = wind_speed / 2.23694  # convert mi/h to m/s

    return wind_speed, canopy_ht, canopy_baseht, bool(ma.any(canopy_ht == 0))


def mid_flame_ws(wind_speed, canopy_cover, canopy_ht, canopy_baseht):
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

    return midflame_ws


def flame_length_parameters(model):
    return LENGTH_MODELS[model]


def flame_length(model, fire_intensity, flame_depth):
    model_params = flame_length_parameters(model)

    if model == 'Finney_HEAD':
        fl = (model_params[0] *
              power(fire_intensity, model_params[1]) /
              power(flame_depth, model_params[2]))
    else:
        fl = model_params[0] * power(fire_intensity, model_params[1])

    return fl


def flame_height(model, flame_length, fire_type, fire_intensity, midflame_ws, flame_tilt, slope_angle, slope_units):
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

    return height


def flame_tilt(model, flame_length, flame_height, slope_angle, slope_units, wind_speed, wind_speed_units, canopy_ht):
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

    return degrees(tilt_v)


def flame_residence_time(ros, fuel_consumption, midflame_ws, units):
    # Calculate flame residence time
    res_time = (0.39 * power(fuel_consumption, 0.25) * power(midflame_ws, 1.51)) / (ros / 60)
    if units == 'min':
        res_time = res_time / 60

    return res_time


def flame_depth(ros, res_time):
    # Calculate flame depth
    fd = ros * res_time

    return fd
