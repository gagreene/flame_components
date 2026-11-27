//! Mid-flame wind speed under sheltered and unsheltered canopies.

use std::fmt;

use ndarray::{ArrayD, ArrayViewD};

use crate::{array, Cell, Error};

const FEET_PER_METRE: f64 = 3.28084;

/// Input unit system for mid-flame wind speed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindUnits {
    /// Wind at 10 m in km/h and canopy heights in metres.
    Si,
    /// Wind at 20 ft in mi/h and canopy heights in feet.
    Imperial,
}

/// Mid-flame wind speed in m/s, with canopy substitution diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct WindResult<T> {
    /// Unclamped speed with the calculation's domain mask.
    pub value: T,
    /// An unmasked zero stand height was replaced by the equivalent of 0.5 m.
    pub substituted_canopy_height: bool,
}

/// Array wind calculation error with any canopy substitution diagnostic.
#[derive(Debug)]
pub struct WindError {
    /// Underlying error from broadcasting the inputs.
    pub error: Error,
    /// The canopy height input contains an unmasked zero.
    pub substituted_canopy_height: bool,
}

impl fmt::Display for WindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for WindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

fn canopy_height_feet(height: Cell<f64>, units: WindUnits) -> Cell<f64> {
    match units {
        WindUnits::Si => height * FEET_PER_METRE,
        WindUnits::Imperial => height,
    }
}

/// Calculate mid-flame wind speed in m/s from a reference-height wind speed.
///
/// Supply canopy cover as a percentage and both canopy heights in the selected
/// unit system. SI wind speed is converted to its 20-ft equivalent.
pub fn mid_flame_ws(
    wind_speed: Cell<f64>,
    canopy_cover: Cell<f64>,
    canopy_ht: Cell<f64>,
    canopy_baseht: Cell<f64>,
    units: WindUnits,
) -> WindResult<Cell<f64>> {
    let wind_speed = match units {
        WindUnits::Si => wind_speed / (3.6 * 1.15),
        WindUnits::Imperial => wind_speed / 2.23694,
    };
    let canopy_ht = canopy_height_feet(canopy_ht, units);
    let canopy_baseht = canopy_height_feet(canopy_baseht, units);

    let zero_height = canopy_ht.eq_cell(0.0);
    let canopy_ht = Cell::select(zero_height, Cell::present(0.5 * FEET_PER_METRE), canopy_ht);
    let crown_ratio = (canopy_ht - canopy_baseht) / canopy_ht;
    let canopy_fraction = crown_ratio * canopy_cover / 300.0;
    let wind_profile = ((20.0 + 0.36 * canopy_ht) / (0.13 * canopy_ht)).log();
    let unsheltered_speed = wind_speed * 1.83 / wind_profile;
    let sheltered_speed =
        wind_speed * 0.555 / ((canopy_fraction * canopy_ht).sqrt() * wind_profile);

    WindResult {
        value: Cell::select(canopy_fraction.le(5.0), unsheltered_speed, sheltered_speed),
        substituted_canopy_height: !zero_height.masked && zero_height.value,
    }
}

/// Broadcast reference-height wind, canopy cover percentage, stand height,
/// and canopy base height to calculate an array of mid-flame speeds in m/s.
///
/// The stand height diagnostic applies to the original input view, including
/// when the inputs cannot be broadcast to a common shape.
///
/// # Errors
///
/// Returns [`WindError`] when the input shapes are incompatible.
pub fn mid_flame_ws_array(
    wind_speed: ArrayViewD<'_, Cell<f64>>,
    canopy_cover: ArrayViewD<'_, Cell<f64>>,
    canopy_ht: ArrayViewD<'_, Cell<f64>>,
    canopy_baseht: ArrayViewD<'_, Cell<f64>>,
    units: WindUnits,
) -> Result<WindResult<ArrayD<Cell<f64>>>, WindError> {
    let substituted_canopy_height = canopy_ht.iter().any(|&height| {
        let zero_height = canopy_height_feet(height, units).eq_cell(0.0);
        !zero_height.masked && zero_height.value
    });
    let value = array::broadcast(
        [
            wind_speed.view(),
            canopy_cover.view(),
            canopy_ht.view(),
            canopy_baseht.view(),
        ],
        |[wind, cover, height, base_height]| {
            mid_flame_ws(wind, cover, height, base_height, units).value
        },
    )
    .map_err(|error| WindError {
        error,
        substituted_canopy_height,
    })?;

    Ok(WindResult {
        value,
        substituted_canopy_height,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{arr0, arr1};

    use super::*;

    fn assert_wind(cell: Cell<f64>, expected: f64) {
        assert!(!cell.masked);
        assert!((cell.value - expected).abs() <= 2e-14 * expected.abs());
    }

    #[test]
    fn replaces_zero_canopy_height_in_both_unit_systems() {
        for (units, expected) in [
            (WindUnits::Si, 1.740_994_797_470_333_5),
            (WindUnits::Imperial, 3.222_133_120_033_250_4),
        ] {
            let result = mid_flame_ws(
                Cell::input(18.0),
                Cell::input(35.0),
                Cell::input(-0.0),
                Cell::input(0.0),
                units,
            );

            assert!(result.substituted_canopy_height);
            assert_wind(result.value, expected);
        }
    }

    #[test]
    fn converts_both_canopy_heights_for_sheltered_si_wind() {
        let result = mid_flame_ws(
            Cell::input(18.0),
            Cell::input(2400.0),
            Cell::input(10.0),
            Cell::input(2.5),
            WindUnits::Si,
        );

        assert!(!result.substituted_canopy_height);
        assert_wind(result.value, 0.085_593_658_551_620_11);
    }

    #[test]
    fn masked_zero_canopy_height_does_not_trigger_substitution() {
        let result = mid_flame_ws(
            Cell::input(18.0),
            Cell::input(35.0),
            Cell {
                value: 0.0,
                masked: true,
            },
            Cell::input(0.0),
            WindUnits::Si,
        );

        assert!(!result.substituted_canopy_height);
        assert!(result.value.masked);
    }

    #[test]
    fn shelter_threshold_and_branch_masks_are_per_cell() {
        let wind = arr0(Cell::input(20.0));
        let canopy_cover = arr1(&[
            Cell::input(-1.0),
            Cell::input(f64::from_bits(1500.0_f64.to_bits() - 1)),
            Cell::input(1500.0),
            Cell::input(f64::from_bits(1500.0_f64.to_bits() + 1)),
            Cell::input(f64::NAN),
        ]);
        let canopy_ht = arr0(Cell::input(10.0));
        let canopy_baseht = arr0(Cell::input(0.0));
        let result = mid_flame_ws_array(
            wind.view().into_dyn(),
            canopy_cover.view().into_dyn(),
            canopy_ht.view().into_dyn(),
            canopy_baseht.view().into_dyn(),
            WindUnits::Imperial,
        )
        .unwrap();

        assert!(!result.substituted_canopy_height);
        assert_eq!(result.value.shape(), &[5]);
        for index in 0..3 {
            assert_wind(result.value[[index]], 5.644_118_785_395_857);
        }
        assert_wind(result.value[[3]], 0.242_076_725_701_810_24);
        assert!(result.value[[4]].masked);
    }

    #[test]
    fn shape_error_retains_canopy_substitution_diagnostic() {
        let wind = arr0(Cell::input(20.0));
        let canopy_cover = arr0(Cell::input(50.0));
        let canopy_ht = arr1(&[Cell::input(0.0), Cell::input(1.0)]);
        let canopy_baseht = arr1(&[Cell::input(1.0); 3]);
        let error = mid_flame_ws_array(
            wind.view().into_dyn(),
            canopy_cover.view().into_dyn(),
            canopy_ht.view().into_dyn(),
            canopy_baseht.view().into_dyn(),
            WindUnits::Si,
        )
        .unwrap_err();

        assert!(error.substituted_canopy_height);
        assert!(matches!(error.error, Error::IncompatibleShapes { .. }));
    }
}
