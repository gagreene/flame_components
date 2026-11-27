//! Standard, Finney, and Butler flame-tilt calculations.

use std::f64::consts::FRAC_PI_2;

use ndarray::{ArrayD, ArrayViewD, NdFloat};

use crate::precision::Precision;
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
    precision: [Precision; 7],
) -> Cell<f64> {
    let [slope, complement, product, ratio, horizontal, vertical, output] = precision;
    let slope_rad = slope.slope(slope_angle, slope_units);
    let angle = complement.sub(FRAC_PI_2.into(), slope_rad);
    // Keep the reference angle conversions in order because they affect rounding.
    let projected = product.mul(flame_height, complement.degrees(complement.sin(angle)));
    let tilt_h = horizontal.add(
        ratio.asin(ratio.radians(ratio.div(projected, flame_length))),
        slope_rad,
    );
    output.degrees(Cell::select(
        flame_height.eq_cell(flame_length),
        Cell::present(0.0),
        vertical.sub(FRAC_PI_2.into(), tilt_h),
    ))
}

/// Broadcast Finney tilt at the given floating-point widths (bytes) for slope
/// radians, slope complement, height product, division by length, horizontal
/// tilt, vertical tilt, and final degrees.
pub fn finney_tilt_array_with_precision(
    length: ArrayViewD<'_, Cell<f64>>,
    height: ArrayViewD<'_, Cell<f64>>,
    slope: ArrayViewD<'_, Cell<f64>>,
    units: SlopeUnits,
    widths: [u8; 7],
) -> Result<ArrayD<Cell<f64>>, Error> {
    let precision = Precision::widths(widths)?;
    broadcast(
        [length.view(), height.view(), slope.view()],
        |[length, height, slope]| finney_tilt(length, height, slope, units, precision),
    )
}

/// Broadcast Standard tilt using 2-, 4-, or 8-byte floating-point arithmetic.
pub fn standard_tilt_array_with_precision(
    length: ArrayViewD<'_, Cell<f64>>,
    height: ArrayViewD<'_, Cell<f64>>,
    width: u8,
) -> Result<ArrayD<Cell<f64>>, Error> {
    let [p] = Precision::widths([width])?;
    broadcast([length.view(), height.view()], |[length, height]| {
        p.degrees(p.acos(p.div(height, length)))
    })
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
    precision: [Precision; 10],
) -> Cell<f64> {
    let [units, reciprocal, log_arg, log_sum, denom, canopy, cube, numerator, ratio, output] =
        precision;
    let wind_speed = match wind_speed_units {
        WindSpeedUnits::Kph => units.div(wind_speed, 3.6.into()),
        WindSpeedUnits::Mps => wind_speed,
        WindSpeedUnits::Mph => units.div(wind_speed, 2.23694.into()),
    };
    let roughness = log_arg.add(1.0.into(), reciprocal.div(28.0.into(), canopy_ht));
    let profile = log_sum.add(1.0.into(), log_arg.log(roughness));
    let canopy_wind = canopy.div(wind_speed, denom.mul(3.6.into(), profile));
    let buoyancy_ratio = ratio.div(
        numerator.mul(3.0.into(), cube.pow(canopy_wind, 3.0)),
        (2.0 * 9.81 * 10.0).into(),
    );
    output.degrees(ratio.atan(ratio.sqrt(buoyancy_ratio)))
}

/// Broadcast Butler tilt with widths in bytes for unit conversion, reciprocal
/// height, log argument, log plus one, wind denominator, canopy wind, cubing,
/// multiplication by three, gravity division, and final degrees.
pub fn butler_tilt_array_with_precision(
    wind_speed: ArrayViewD<'_, Cell<f64>>,
    units: WindSpeedUnits,
    canopy_ht: ArrayViewD<'_, Cell<f64>>,
    widths: [u8; 10],
) -> Result<ArrayD<Cell<f64>>, Error> {
    canopy_ht
        .iter()
        .copied()
        .try_for_each(validate_canopy_height)?;
    let precision = Precision::widths(widths)?;
    broadcast([wind_speed.view(), canopy_ht.view()], |[wind, canopy]| {
        butler_tilt(wind, units, canopy, precision)
    })
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
            [Precision::Double; 7],
        )),
        TiltModel::Butler {
            wind_speed,
            wind_speed_units,
            canopy_ht,
        } => {
            validate_canopy_height(canopy_ht)?;
            Ok(butler_tilt(
                wind_speed,
                wind_speed_units,
                canopy_ht,
                [Precision::Double; 10],
            ))
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
        } => finney_tilt_array_with_precision(
            flame_length,
            flame_height,
            slope_angle,
            slope_units,
            [8; 7],
        ),
        TiltModel::Butler {
            wind_speed,
            wind_speed_units,
            canopy_ht,
        } => butler_tilt_array_with_precision(wind_speed, wind_speed_units, canopy_ht, [8; 10]),
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
