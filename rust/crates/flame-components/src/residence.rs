//! Nelson and Adkins (1988) flame residence time.

use ndarray::{ArrayD, ArrayViewD};

use crate::precision::{constant, ModelValue, Precision, Stage};
use crate::{array::broadcast, Cell, Error};

/// Unit for the output residence time. Spread rate is always in m/min.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidenceUnits {
    Seconds,
    Minutes,
}

/// Calculate residence time from rate of spread (m/min), fuel consumption
/// (kg/m²), and mid-flame wind speed (m/s).
///
/// A quotient outside the masked division domain yields a masked result.
pub fn flame_residence_time(
    ros: Cell<f64>,
    fuel_consumption: Cell<f64>,
    midflame_ws: Cell<f64>,
    units: ResidenceUnits,
) -> Cell<f64> {
    residence_time(
        ros,
        fuel_consumption,
        midflame_ws,
        units,
        [Precision::Double; 7],
    )
}

fn residence_time<T: ModelValue>(
    ros: Cell<T>,
    fuel: Cell<T>,
    wind: Cell<T>,
    units: ResidenceUnits,
    precision: [T::Stage; 7],
) -> Cell<T> {
    let [fuel_power, coefficient, wind_power, product, rate, division, minutes] = precision;
    let fuel_factor = coefficient.mul(constant(0.39), fuel_power.pow(fuel, 0.25));
    let numerator = product.mul(fuel_factor, wind_power.pow(wind, 1.51));
    let residence_time = division.div(numerator, rate.div(ros, constant(60.0)));
    match units {
        ResidenceUnits::Seconds => residence_time,
        ResidenceUnits::Minutes => minutes.div(residence_time, constant(60.0)),
    }
}

/// Apply [`flame_residence_time`] to broadcast-compatible arrays.
pub fn flame_residence_time_array(
    ros: ArrayViewD<'_, Cell<f64>>,
    fuel_consumption: ArrayViewD<'_, Cell<f64>>,
    midflame_ws: ArrayViewD<'_, Cell<f64>>,
    units: ResidenceUnits,
) -> Result<ArrayD<Cell<f64>>, Error> {
    flame_residence_time_array_with_precision(ros, fuel_consumption, midflame_ws, units, [8; 7])
}

/// Broadcast residence time using the supplied widths in bytes for the fuel
/// power, coefficient product, wind power, numerator product, spread-rate
/// division, residence-time division, and conversion to minutes.
pub fn flame_residence_time_array_with_precision(
    ros: ArrayViewD<'_, Cell<f64>>,
    fuel_consumption: ArrayViewD<'_, Cell<f64>>,
    midflame_ws: ArrayViewD<'_, Cell<f64>>,
    units: ResidenceUnits,
    widths: [u8; 7],
) -> Result<ArrayD<Cell<f64>>, Error> {
    flame_residence_time_array_at_widths(ros, fuel_consumption, midflame_ws, units, widths)
}

pub(crate) fn flame_residence_time_array_at_widths<T: ModelValue>(
    ros: ArrayViewD<'_, Cell<T>>,
    fuel_consumption: ArrayViewD<'_, Cell<T>>,
    midflame_ws: ArrayViewD<'_, Cell<T>>,
    units: ResidenceUnits,
    widths: [u8; 7],
) -> Result<ArrayD<Cell<T>>, Error> {
    let precision = T::Stage::widths(widths)?;
    broadcast(
        [ros.view(), fuel_consumption.view(), midflame_ws.view()],
        |[ros, fuel_consumption, midflame_ws]| {
            residence_time(ros, fuel_consumption, midflame_ws, units, precision)
        },
    )
}
