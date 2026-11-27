//! Published flame-length correlations.

use std::str::FromStr;

use ndarray::{ArrayD, ArrayViewD};

use crate::{array::broadcast, Cell, Error};

macro_rules! length_models {
    ($($variant:ident => $selector:literal, ($coefficient:expr, $intensity:expr, $depth:expr);)+) => {
        /// Published flame-length model.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum LengthModel {
            $($variant,)+
        }

        impl LengthModel {
            /// Return the reference model selector.
            pub const fn selector(self) -> &'static str {
                match self {
                    $(Self::$variant => $selector,)+
                }
            }

            /// Return the multiplier, intensity exponent, and optional depth exponent.
            pub const fn coefficients(self) -> (f64, f64, Option<f64>) {
                match self {
                    $(Self::$variant => ($coefficient, $intensity, $depth),)+
                }
            }
        }

        impl FromStr for LengthModel {
            type Err = Error;

            fn from_str(selector: &str) -> Result<Self, Self::Err> {
                match selector {
                    $($selector => Ok(Self::$variant),)+
                    _ => Err(Error::UnknownLengthModel(selector.to_owned())),
                }
            }
        }
    };
}

const TWO_THIRDS: f64 = 2.0 / 3.0;

// Finney and Grumstrup (2023) report these coefficients for no-wind,
// backing, and heading fires, respectively.
length_models! {
    FonsNoWind => "Fons_NOWIND", (0.024018, TWO_THIRDS, None);
    ThomasNoWind => "Thomas_NOWIND", (0.026700, TWO_THIRDS, None);
    YuanaNoWind => "Yuana_NOWIND", (0.034000, TWO_THIRDS, None);
    BarbonNoWind => "Barbon_NOWIND", (0.062000, 0.5336, None);
    NelsonBack => "Nelson_BACK", (0.027973, TWO_THIRDS, None);
    FernandesBack => "Fernandes_BACK", (0.029000, 0.7240, None);
    ClarkBack => "Clark_BACK", (0.001600, 1.7450, None);
    VegaBack => "Vega_BACK", (0.087000, 0.4930, None);
    ByramHead => "Byram_HEAD", (0.0775, 0.4600, None);
    Anderson1Head => "Anderson1_HEAD", (0.013876, 0.6510, None);
    Anderson2Head => "Anderson2_HEAD", (0.008800, 0.6700, None);
    NewmanHead => "Newman_HEAD", (0.05770, 0.5000, None);
    SneewujagtHead => "Sneewujagt_HEAD", (0.037680, 0.5000, None);
    Nelson1Head => "Nelson1_HEAD", (0.044230, 0.5000, None);
    ClarkHead => "Clark_HEAD", (0.000722, 0.9934, None);
    Nelson2Head => "Nelson2_HEAD", (0.047500, 0.4930, None);
    VanWilgenHead => "VanWilgen_HEAD", (0.046000, 0.4128, None);
    BurrowsHead => "Burrows_HEAD", (0.040480, 0.5740, None);
    MarsdenSmedleyHead => "MarsdenSmedley_HEAD", (0.148, 0.403, None);
    Weise1Head => "Weise1_HEAD", (0.016000, 0.7000, None);
    CatchpoleHead => "Catchpole_HEAD", (0.032500, 0.5600, None);
    Fernandes1Head => "Fernandes1_HEAD", (0.051600, 0.4530, None);
    ButlerHead => "Butler_HEAD", (0.017500, TWO_THIRDS, None);
    FernandesHead => "Fernandes_HEAD", (0.045000, 0.5430, None);
    Nelson3Head => "Nelson3_HEAD", (0.014200, TWO_THIRDS, None);
    Nelson4Head => "Nelson4_HEAD", (0.015500, TWO_THIRDS, None);
    Weise2Head => "Weise2_HEAD", (0.2000000, 0.3400, None);
    DaviesHead => "Davies_HEAD", (0.220000, 0.2900, None);
    FinneyHead => "Finney_HEAD", (0.01051, 0.774, Some(0.161));
}

/// Calculate raw flame length in metres from fire intensity in kW/m.
///
/// [LengthModel::FinneyHead] requires flame depth in metres. Other models ignore it.
/// Apply [Cell::nonnegative] to enforce the reference minimum of zero.
pub fn flame_length(
    model: LengthModel,
    fire_intensity: Cell<f64>,
    flame_depth: Option<Cell<f64>>,
) -> Result<Cell<f64>, Error> {
    let (coefficient, intensity_exponent, depth_exponent) = model.coefficients();
    let length = coefficient * fire_intensity.powf(intensity_exponent);
    match depth_exponent {
        Some(exponent) => {
            let depth = flame_depth.ok_or(Error::MissingFlameDepth)?;
            Ok(length / depth.powf(exponent))
        }
        None => Ok(length),
    }
}

/// Calculate raw flame length in metres from fire intensity (kW/m) and,
/// for [LengthModel::FinneyHead], flame depth (m).
///
/// The inputs used by the selected model broadcast to the output shape.
pub fn flame_length_array(
    model: LengthModel,
    fire_intensity: ArrayViewD<'_, Cell<f64>>,
    flame_depth: Option<ArrayViewD<'_, Cell<f64>>>,
) -> Result<ArrayD<Cell<f64>>, Error> {
    let (coefficient, intensity_exponent, depth_exponent) = model.coefficients();
    let numerator = |intensity: Cell<f64>| coefficient * intensity.powf(intensity_exponent);
    match depth_exponent {
        Some(exponent) => {
            let flame_depth = flame_depth.ok_or(Error::MissingFlameDepth)?;
            broadcast(
                [fire_intensity.view(), flame_depth.view()],
                |[intensity, depth]| numerator(intensity) / depth.powf(exponent),
            )
        }
        None => broadcast([fire_intensity], |[intensity]| numerator(intensity)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn finney_broadcasts_depths_and_masks_zero_depth() {
        let intensity = arr2(&[[Cell::present(425.0)], [Cell::present(0.0)]]);
        let depth = arr1(&[Cell::present(1.8), Cell::present(0.0)]);
        let result = flame_length_array(
            LengthModel::FinneyHead,
            intensity.view().into_dyn(),
            Some(depth.view().into_dyn()),
        )
        .unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert!(!result[[0, 0]].masked);
        assert!((result[[0, 0]].value - 1.0348487384429612).abs() < 1e-14);
        assert!(result[[0, 1]].masked);
        assert_eq!(result[[1, 0]], Cell::present(0.0));
        assert!(result[[1, 1]].masked);
    }

    #[test]
    fn finney_requires_flame_depth() {
        assert!(matches!(
            flame_length(LengthModel::FinneyHead, Cell::present(425.0), None),
            Err(Error::MissingFlameDepth)
        ));
    }

    #[test]
    fn oversized_length_output_returns_a_typed_error() {
        let intensity = ndarray::arr0(Cell::present(1.0));
        let shape = [isize::MAX as usize];
        let view = intensity.broadcast(ndarray::IxDyn(&shape)).unwrap();
        assert_eq!(
            flame_length_array(LengthModel::ByramHead, view, None),
            Err(Error::OutputTooLarge {
                shape: shape.to_vec()
            })
        );
    }

    #[test]
    fn byram_uses_the_published_selector_and_masks_negative_intensity() {
        let model = "Byram_HEAD".parse::<LengthModel>().unwrap();
        assert_eq!(model.coefficients(), (0.0775, 0.46, None));
        assert!(
            flame_length(model, Cell::present(-1.0), None)
                .unwrap()
                .masked
        );
    }
}
