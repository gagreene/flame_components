//! Nelson and Finney flame-height calculations.

use std::f64::consts::FRAC_PI_2;

use ndarray::{ArrayD, ArrayViewD};

use crate::precision::{constant, ModelValue, Precision, Stage};
use crate::{array::broadcast, Cell, Error, SlopeUnits};

/// Fire classes used by the Nelson flame-height model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FireType {
    Surface = 1,
    PassiveCrown = 2,
    ActiveCrown = 3,
}

impl TryFrom<f64> for FireType {
    type Error = Error;

    fn try_from(code: f64) -> Result<Self, Self::Error> {
        match code {
            1.0 => Ok(Self::Surface),
            2.0 => Ok(Self::PassiveCrown),
            3.0 => Ok(Self::ActiveCrown),
            _ => Err(Error::InvalidFireType(code)),
        }
    }
}

impl From<FireType> for Cell<f64> {
    fn from(fire_type: FireType) -> Self {
        Self::present(f64::from(fire_type as u8))
    }
}

/// Model-specific flame-height inputs, as cells or array views.
#[derive(Clone, Copy, Debug)]
pub enum HeightModel<T> {
    Nelson {
        /// Fire class code (1 for surface, 2 for passive crown, 3 for active crown).
        fire_type: T,
        /// Fire intensity in kW/m.
        fire_intensity: T,
        /// Mid-flame wind speed in m/s.
        midflame_ws: T,
    },
    Finney {
        /// Flame tilt from vertical in degrees.
        flame_tilt: T,
        /// Ground slope in the selected units.
        slope_angle: T,
        slope_units: SlopeUnits,
    },
}

fn validate_fire_type<T: ModelValue>(fire_type: Cell<T>) -> Result<(), Error> {
    if !fire_type.masked
        && ![1.0, 2.0, 3.0]
            .iter()
            .any(|&code| fire_type.value == T::from_f64(code))
    {
        return Err(Error::InvalidFireType(fire_type.value.to_f64()));
    }
    Ok(())
}

fn nelson_height<T: ModelValue>(
    flame_length: Cell<T>,
    fire_type: Cell<T>,
    fire_intensity: Cell<T>,
    midflame_ws: Cell<T>,
    precision: [T::Stage; 3],
) -> Cell<T> {
    let [product, division, selected] = precision;
    // The reference membership test returns false for masked fire types.
    let coefficient = if !fire_type.masked
        && (fire_type.value == T::from_f64(1.0) || fire_type.value == T::from_f64(2.0))
    {
        1.0 / 360.0
    } else {
        0.0175
    };
    let height = selected.convert(Cell::select(
        midflame_ws.eq_cell(constant(0.0)),
        flame_length,
        division.div(
            product.mul(constant(coefficient), fire_intensity),
            midflame_ws,
        ),
    ));
    Cell::select(height.gt(flame_length), flame_length, height)
}

/// Broadcast Nelson height with widths in bytes for the coefficient product,
/// division by wind speed, and the selected height at zero or nonzero wind.
pub fn nelson_height_array_with_precision(
    flame_length: ArrayViewD<'_, Cell<f64>>,
    fire_type: ArrayViewD<'_, Cell<f64>>,
    fire_intensity: ArrayViewD<'_, Cell<f64>>,
    midflame_ws: ArrayViewD<'_, Cell<f64>>,
    widths: [u8; 3],
) -> Result<ArrayD<Cell<f64>>, Error> {
    nelson_height_array_at_widths(flame_length, fire_type, fire_intensity, midflame_ws, widths)
}

pub(crate) fn nelson_height_array_at_widths<T: ModelValue>(
    flame_length: ArrayViewD<'_, Cell<T>>,
    fire_type: ArrayViewD<'_, Cell<T>>,
    fire_intensity: ArrayViewD<'_, Cell<T>>,
    midflame_ws: ArrayViewD<'_, Cell<T>>,
    widths: [u8; 3],
) -> Result<ArrayD<Cell<T>>, Error> {
    fire_type.iter().copied().try_for_each(validate_fire_type)?;
    let precision = T::Stage::widths(widths)?;
    broadcast(
        [
            flame_length.view(),
            fire_type.view(),
            fire_intensity.view(),
            midflame_ws.view(),
        ],
        |[length, kind, intensity, wind]| nelson_height(length, kind, intensity, wind, precision),
    )
}

fn finney_height<T: ModelValue>(
    flame_length: Cell<T>,
    flame_tilt: Cell<T>,
    slope_angle: Cell<T>,
    slope_units: SlopeUnits,
    precision: [T::Stage; 8],
) -> Cell<T> {
    let [tilt_rad, horizontal, slope, flat, difference, product, complement, quotient] = precision;
    let slope_rad = slope.slope(slope_angle, slope_units);
    let tilt_h = horizontal.sub(constant(FRAC_PI_2), tilt_rad.radians(flame_tilt));
    let tilted = difference.sin(difference.sub(tilt_h, slope_rad));
    let denominator = complement.sin(complement.sub(constant(FRAC_PI_2), slope_rad));
    Cell::select(
        slope_angle.le(constant(1.0)),
        flat.mul(flame_length, horizontal.sin(tilt_h)),
        quotient.div(product.mul(flame_length, tilted), denominator),
    )
}

/// Broadcast Finney height with independently selected floating-point widths.
///
/// Widths in bytes describe tilt radians, horizontal tilt, slope radians, flat
/// product, angle difference, slope product, slope complement, and final division.
pub fn finney_height_array_with_precision(
    length: ArrayViewD<'_, Cell<f64>>,
    tilt: ArrayViewD<'_, Cell<f64>>,
    slope: ArrayViewD<'_, Cell<f64>>,
    units: SlopeUnits,
    widths: [u8; 8],
) -> Result<ArrayD<Cell<f64>>, Error> {
    finney_height_array_at_widths(length, tilt, slope, units, widths)
}

pub(crate) fn finney_height_array_at_widths<T: ModelValue>(
    length: ArrayViewD<'_, Cell<T>>,
    tilt: ArrayViewD<'_, Cell<T>>,
    slope: ArrayViewD<'_, Cell<T>>,
    units: SlopeUnits,
    widths: [u8; 8],
) -> Result<ArrayD<Cell<T>>, Error> {
    let precision = T::Stage::widths(widths)?;
    broadcast(
        [length.view(), tilt.view(), slope.view()],
        |[length, tilt, slope]| finney_height(length, tilt, slope, units, precision),
    )
}

/// Calculate raw flame height in metres from flame length in metres.
///
/// Apply [Cell::nonnegative] to enforce the reference minimum of zero.
pub fn flame_height(
    flame_length: Cell<f64>,
    model: HeightModel<Cell<f64>>,
) -> Result<Cell<f64>, Error> {
    match model {
        HeightModel::Nelson {
            fire_type,
            fire_intensity,
            midflame_ws,
        } => {
            validate_fire_type(fire_type)?;
            Ok(nelson_height(
                flame_length,
                fire_type,
                fire_intensity,
                midflame_ws,
                [Precision::DOUBLE; 3],
            ))
        }
        HeightModel::Finney {
            flame_tilt,
            slope_angle,
            slope_units,
        } => Ok(finney_height(
            flame_length,
            flame_tilt,
            slope_angle,
            slope_units,
            [Precision::DOUBLE; 8],
        )),
    }
}

/// Broadcast flame length (m) and the selected model inputs to calculate
/// an array of raw flame heights in metres.
pub fn flame_height_array(
    flame_length: ArrayViewD<'_, Cell<f64>>,
    model: HeightModel<ArrayViewD<'_, Cell<f64>>>,
) -> Result<ArrayD<Cell<f64>>, Error> {
    match model {
        HeightModel::Nelson {
            fire_type,
            fire_intensity,
            midflame_ws,
        } => nelson_height_array_with_precision(
            flame_length,
            fire_type,
            fire_intensity,
            midflame_ws,
            [8; 3],
        ),
        HeightModel::Finney {
            flame_tilt,
            slope_angle,
            slope_units,
        } => finney_height_array_with_precision(
            flame_length,
            flame_tilt,
            slope_angle,
            slope_units,
            [8; 8],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1};

    #[test]
    fn nelson_selects_length_at_zero_wind_and_masks_missing_wind() {
        let length = arr0(Cell::present(2.0));
        let fire_type = arr0(Cell::present(1.0));
        let intensity = arr0(Cell::input(f64::NAN));
        let wind = arr1(&[Cell::present(0.0), Cell::missing(0.0), Cell::present(1.0)]);
        let result = flame_height_array(
            length.view().into_dyn(),
            HeightModel::Nelson {
                fire_type: fire_type.view().into_dyn(),
                fire_intensity: intensity.view().into_dyn(),
                midflame_ws: wind.view().into_dyn(),
            },
        )
        .unwrap();

        assert_eq!(result[0], Cell::present(2.0));
        assert!(result[1].masked);
        assert!(result[2].masked);
    }

    #[test]
    fn nelson_uses_crown_coefficient_for_masked_fire_type() {
        let height = flame_height(
            Cell::present(10.0),
            HeightModel::Nelson {
                fire_type: Cell::missing(2.0),
                fire_intensity: Cell::present(360.0),
                midflame_ws: Cell::present(1.0),
            },
        )
        .unwrap();

        assert!(!height.masked);
        assert!((height.value - 6.3).abs() < 1e-14);
    }

    #[test]
    fn nelson_rejects_unmasked_unknown_fire_type_even_at_zero_wind() {
        assert!(matches!(
            flame_height(
                Cell::present(2.0),
                HeightModel::Nelson {
                    fire_type: Cell::present(1.5),
                    fire_intensity: Cell::present(360.0),
                    midflame_ws: Cell::present(0.0),
                },
            ),
            Err(Error::InvalidFireType(1.5))
        ));
    }

    #[test]
    fn finney_compares_raw_slope_against_one_in_percent_units() {
        let calc = |slope| {
            flame_height(
                Cell::present(8.0),
                HeightModel::Finney {
                    flame_tilt: Cell::present(25.0),
                    slope_angle: Cell::present(slope),
                    slope_units: SlopeUnits::Percent,
                },
            )
            .unwrap()
        };

        let at_boundary = calc(1.0);
        let above_boundary = calc(f64::from_bits(1.0_f64.to_bits() + 1));
        assert!(!at_boundary.masked);
        assert!(!above_boundary.masked);
        assert!((at_boundary.value - 7.2504622962931995).abs() < 1e-14);
        assert!((above_boundary.value - 7.216652835353943).abs() < 1e-14);
    }
}
