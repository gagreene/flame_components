//! Standard, Finney, and Butler flame-tilt calculations.

use std::f64::consts::FRAC_PI_2;

use ndarray::{ArrayD, ArrayViewD, NdFloat};

use crate::{array::broadcast, Cell, Error, SlopeUnits, WindSpeedUnits};

/// Model-specific flame-tilt inputs, as cells or array views.
#[derive(Clone, Copy, Debug)]
pub enum TiltModel<T> {
    Standard {
        /// Flame length in metres.
        flame_length: T,
        /// Flame height in metres.
        flame_height: T,
    },
    Finney {
        /// Flame length in metres.
        flame_length: T,
        /// Flame height in metres.
        flame_height: T,
        /// Ground slope in the selected units.
        slope_angle: T,
        slope_units: SlopeUnits,
    },
    Butler {
        /// Wind speed at 10 m in the selected units.
        wind_speed: T,
        wind_speed_units: WindSpeedUnits,
        /// Canopy height in metres, strictly positive if present.
        canopy_ht: T,
    },
}

/// Calculate Standard tilt in degrees while preserving the input float type.
///
/// Both lengths are in metres. Binary32 and binary64 preserve their own
/// intermediate rounding before the final clamp.
pub fn standard_tilt<F: NdFloat>(flame_length: Cell<F>, flame_height: Cell<F>) -> Cell<F> {
    (flame_height / flame_length).acos().degrees()
}

/// Apply [`standard_tilt`] to broadcast-compatible array views.
pub fn standard_tilt_array<F: NdFloat>(
    flame_length: ArrayViewD<'_, Cell<F>>,
    flame_height: ArrayViewD<'_, Cell<F>>,
) -> Result<ArrayD<Cell<F>>, Error> {
    broadcast(
        [flame_length.view(), flame_height.view()],
        |[length, height]| standard_tilt(length, height),
    )
}

fn finney_tilt(
    flame_length: Cell<f64>,
    flame_height: Cell<f64>,
    slope_angle: Cell<f64>,
    slope_units: SlopeUnits,
) -> Cell<f64> {
    let slope_rad = slope_units.radians(slope_angle);
    let right_angle = Cell::present(90.0).radians();
    // Keep the reference angle conversions in order because they affect rounding.
    let tilt_h = (flame_height * (right_angle - slope_rad).sin().degrees() / flame_length)
        .radians()
        .asin()
        + slope_rad;
    Cell::select(
        flame_height.eq_cell(flame_length),
        Cell::present(0.0),
        FRAC_PI_2 - tilt_h,
    )
    .degrees()
}

fn validate_canopy_height(canopy_ht: Cell<f64>) -> Result<(), Error> {
    if !canopy_ht.masked && canopy_ht.value <= 0.0 {
        return Err(Error::NonPositiveCanopyHeight(canopy_ht.value));
    }
    Ok(())
}

fn butler_tilt(
    wind_speed: Cell<f64>,
    wind_speed_units: WindSpeedUnits,
    canopy_ht: Cell<f64>,
) -> Cell<f64> {
    let wind_speed = match wind_speed_units {
        WindSpeedUnits::Kph => wind_speed / 3.6,
        WindSpeedUnits::Mps => wind_speed,
        WindSpeedUnits::Mph => wind_speed / 2.23694,
    };
    let canopy_wind = wind_speed / (3.6 * (1.0 + (1.0 + (28.0 / canopy_ht)).log()));
    let gravity = 9.81;
    ((3.0 * canopy_wind.powf(3.0)) / (2.0 * gravity * 10.0))
        .sqrt()
        .atan()
        .degrees()
}

/// Calculate raw flame tilt in degrees measured from vertical.
///
/// Apply [Cell::nonnegative] to enforce the reference minimum of zero.
pub fn flame_tilt(model: TiltModel<Cell<f64>>) -> Result<Cell<f64>, Error> {
    match model {
        TiltModel::Standard {
            flame_length,
            flame_height,
        } => Ok(standard_tilt(flame_length, flame_height)),
        TiltModel::Finney {
            flame_length,
            flame_height,
            slope_angle,
            slope_units,
        } => Ok(finney_tilt(
            flame_length,
            flame_height,
            slope_angle,
            slope_units,
        )),
        TiltModel::Butler {
            wind_speed,
            wind_speed_units,
            canopy_ht,
        } => {
            validate_canopy_height(canopy_ht)?;
            Ok(butler_tilt(wind_speed, wind_speed_units, canopy_ht))
        }
    }
}

/// Broadcast the selected model inputs to calculate raw flame tilt in
/// degrees measured from vertical.
pub fn flame_tilt_array(
    model: TiltModel<ArrayViewD<'_, Cell<f64>>>,
) -> Result<ArrayD<Cell<f64>>, Error> {
    match model {
        TiltModel::Standard {
            flame_length,
            flame_height,
        } => standard_tilt_array(flame_length, flame_height),
        TiltModel::Finney {
            flame_length,
            flame_height,
            slope_angle,
            slope_units,
        } => broadcast(
            [flame_length, flame_height, slope_angle],
            |[length, height, slope]| finney_tilt(length, height, slope, slope_units),
        ),
        TiltModel::Butler {
            wind_speed,
            wind_speed_units,
            canopy_ht,
        } => {
            canopy_ht
                .iter()
                .copied()
                .try_for_each(validate_canopy_height)?;
            broadcast([wind_speed, canopy_ht], |[wind, canopy]| {
                butler_tilt(wind, wind_speed_units, canopy)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1};

    #[test]
    fn standard_masks_height_greater_than_length() {
        let result = flame_tilt(TiltModel::Standard {
            flame_length: Cell::present(2.0),
            flame_height: Cell::present(3.0),
        })
        .unwrap();
        assert!(result.masked);
    }

    #[test]
    fn finney_returns_zero_for_equal_lengths_even_with_masked_slope() {
        let result = flame_tilt(TiltModel::Finney {
            flame_length: Cell::present(5.0),
            flame_height: Cell::present(5.0),
            slope_angle: Cell::input(f64::NAN),
            slope_units: SlopeUnits::Degrees,
        })
        .unwrap();
        assert_eq!(result, Cell::present(0.0));
    }

    #[test]
    fn butler_applies_wind_conversion_and_preserves_a_canopy_mask() {
        let wind = arr0(Cell::present(11.1847));
        let canopy = arr1(&[Cell::present(10.0), Cell::missing(0.0)]);
        let result = flame_tilt_array(TiltModel::Butler {
            wind_speed: wind.view().into_dyn(),
            wind_speed_units: WindSpeedUnits::Mph,
            canopy_ht: canopy.view().into_dyn(),
        })
        .unwrap();
        assert!(!result[0].masked);
        assert!((result[0].value - 3.2466779228941935).abs() < 1e-14);
        assert!(result[1].masked);
    }

    #[test]
    fn butler_rejects_unmasked_zero_canopy() {
        assert!(matches!(
            flame_tilt(TiltModel::Butler {
                wind_speed: Cell::present(5.0),
                wind_speed_units: WindSpeedUnits::Mps,
                canopy_ht: Cell::present(0.0),
            }),
            Err(Error::NonPositiveCanopyHeight(0.0))
        ));
    }
}
