//! Nelson and Adkins (1988) flame residence time.

use ndarray::{ArrayD, ArrayViewD};

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
    let residence_time =
        (0.39 * fuel_consumption.powf(0.25) * midflame_ws.powf(1.51)) / (ros / 60.0);
    match units {
        ResidenceUnits::Seconds => residence_time,
        ResidenceUnits::Minutes => residence_time / 60.0,
    }
}

/// Apply [`flame_residence_time`] to broadcast-compatible arrays.
pub fn flame_residence_time_array(
    ros: ArrayViewD<'_, Cell<f64>>,
    fuel_consumption: ArrayViewD<'_, Cell<f64>>,
    midflame_ws: ArrayViewD<'_, Cell<f64>>,
    units: ResidenceUnits,
) -> Result<ArrayD<Cell<f64>>, Error> {
    broadcast(
        [ros.view(), fuel_consumption.view(), midflame_ws.view()],
        |[ros, fuel_consumption, midflame_ws]| {
            flame_residence_time(ros, fuel_consumption, midflame_ws, units)
        },
    )
}
