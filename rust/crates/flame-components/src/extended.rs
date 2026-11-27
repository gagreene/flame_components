//! Model arrays with native C long-double payloads and per-stage precision.
//!
//! A width of 2, 4, or 8 selects the existing binary16, binary32, or binary64
//! kernel. A larger width must match the host long-double storage size.

use ndarray::{ArrayD, ArrayViewD};

pub use crate::depth::flame_depth_array;
pub use flame_components_wide::{format, LongDouble, BYTE_CAPACITY};

use crate::depth::DepthValue;
use crate::length::LengthModel;
use crate::precision::{ModelValue, Precision, Stage};
use crate::residence::ResidenceUnits;
use crate::wind::{WindError, WindResult, WindUnits};
use crate::{Cell, Error, SlopeUnits, WindSpeedUnits};

type View<'a> = ArrayViewD<'a, Cell<LongDouble>>;

impl DepthValue for LongDouble {
    fn depth_product(self, time: Self) -> Self {
        self.mul(time)
    }

    fn clamp_nonnegative(self) -> Self {
        let zero = Self::from_f64(0.0);
        if self > zero {
            self
        } else {
            zero
        }
    }
}

impl ModelValue for LongDouble {
    type Stage = LongDoubleStage;

    fn from_f64(value: f64) -> Self {
        LongDouble::from_f64(value)
    }
    fn to_f64(self) -> f64 {
        LongDouble::to_f64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum LongDoubleStage {
    Narrow(Precision),
    Native,
}

fn narrow(cell: Cell<LongDouble>, precision: Precision) -> Cell<f64> {
    cell.map(|value| match precision {
        Precision::Single => f64::from(value.to_f32()),
        _ => value.to_f64(),
    })
}

fn finite(mut cell: Cell<LongDouble>) -> Cell<LongDouble> {
    cell.masked |= !cell.value.is_finite();
    cell
}

macro_rules! unary_stage {
    ($method:ident, $domain:expr) => {
        fn $method(self, value: Cell<LongDouble>) -> Cell<LongDouble> {
            match self {
                Self::Narrow(p) => p.$method(narrow(value, p)).map(LongDouble::from_f64),
                Self::Native => {
                    let result = value.map(LongDouble::$method);
                    if $domain {
                        finite(result)
                    } else {
                        result
                    }
                }
            }
        }
    };
}

macro_rules! binary_stage {
    ($method:ident) => {
        fn $method(self, left: Cell<LongDouble>, right: Cell<LongDouble>) -> Cell<LongDouble> {
            match self {
                Self::Narrow(p) => p
                    .$method(narrow(left, p), narrow(right, p))
                    .map(LongDouble::from_f64),
                Self::Native => Cell {
                    value: left.value.$method(right.value),
                    masked: left.masked || right.masked,
                },
            }
        }
    };
}

impl Stage<LongDouble> for LongDoubleStage {
    fn widths<const N: usize>(widths: [u8; N]) -> Result<[Self; N], Error> {
        let host_width = format().0;
        let mut stages = [Self::Native; N];
        for (stage, width) in stages.iter_mut().zip(widths) {
            if width <= 8 || usize::from(width) != host_width {
                let [narrow] = Precision::widths([width])?;
                *stage = Self::Narrow(narrow);
            }
        }
        Ok(stages)
    }

    fn convert(self, cell: Cell<LongDouble>) -> Cell<LongDouble> {
        match self {
            Self::Native => cell,
            Self::Narrow(p) => p.convert(narrow(cell, p)).map(LongDouble::from_f64),
        }
    }

    binary_stage!(add);
    binary_stage!(sub);
    binary_stage!(mul);

    fn div(self, left: Cell<LongDouble>, right: Cell<LongDouble>) -> Cell<LongDouble> {
        match self {
            Self::Narrow(p) => p
                .div(narrow(left, p), narrow(right, p))
                .map(LongDouble::from_f64),
            Self::Native => {
                // numpy.ma uses binary64's minimum normal even for long doubles.
                let cutoff = left
                    .value
                    .abs()
                    .mul(LongDouble::from_f64(f64::MIN_POSITIVE));
                finite(Cell {
                    value: left.value.div(right.value),
                    masked: left.masked || right.masked || cutoff >= right.value.abs(),
                })
            }
        }
    }

    fn pow(self, base: Cell<LongDouble>, exponent: f64) -> Cell<LongDouble> {
        match self {
            Self::Narrow(p) => p.pow(narrow(base, p), exponent).map(LongDouble::from_f64),
            Self::Native => finite(base.map(|x| x.pow(LongDouble::from_f64(exponent)))),
        }
    }

    unary_stage!(sin, false);
    unary_stage!(sqrt, true);
    unary_stage!(log, true);
    unary_stage!(atan, false);
    unary_stage!(asin, true);
    unary_stage!(acos, true);
    unary_stage!(radians, false);
    unary_stage!(degrees, false);
}

/// Extended-precision counterpart of [`crate::residence::flame_residence_time_array_with_precision`].
pub fn flame_residence_time_array_with_precision(
    ros: View<'_>,
    fuel_consumption: View<'_>,
    midflame_ws: View<'_>,
    units: ResidenceUnits,
    widths: [u8; 7],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::residence::flame_residence_time_array_at_widths(
        ros,
        fuel_consumption,
        midflame_ws,
        units,
        widths,
    )
}

/// Extended-precision counterpart of [`crate::height::nelson_height_array_with_precision`].
pub fn nelson_height_array_with_precision(
    flame_length: View<'_>,
    fire_type: View<'_>,
    fire_intensity: View<'_>,
    midflame_ws: View<'_>,
    widths: [u8; 3],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::height::nelson_height_array_at_widths(
        flame_length,
        fire_type,
        fire_intensity,
        midflame_ws,
        widths,
    )
}

/// Extended-precision counterpart of [`crate::height::finney_height_array_with_precision`].
pub fn finney_height_array_with_precision(
    length: View<'_>,
    tilt: View<'_>,
    slope: View<'_>,
    units: SlopeUnits,
    widths: [u8; 8],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::height::finney_height_array_at_widths(length, tilt, slope, units, widths)
}

/// Extended-precision counterpart of [`crate::wind::mid_flame_ws_array_with_precision`].
pub fn mid_flame_ws_array_with_precision(
    wind_speed: View<'_>,
    canopy_cover: View<'_>,
    canopy_ht: View<'_>,
    canopy_baseht: View<'_>,
    units: WindUnits,
    widths: [u8; 18],
) -> Result<WindResult<ArrayD<Cell<LongDouble>>>, WindError> {
    crate::wind::mid_flame_ws_array_at_widths(
        wind_speed,
        canopy_cover,
        canopy_ht,
        canopy_baseht,
        units,
        widths,
    )
}

/// Extended-precision counterpart of [`crate::length::flame_length_array_with_precision`].
pub fn flame_length_array_with_precision(
    model: LengthModel,
    fire_intensity: View<'_>,
    flame_depth: Option<View<'_>>,
    widths: [u8; 4],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::length::flame_length_array_at_widths(model, fire_intensity, flame_depth, widths)
}

/// Extended-precision counterpart of [`crate::tilt::finney_tilt_array_with_precision`].
pub fn finney_tilt_array_with_precision(
    length: View<'_>,
    height: View<'_>,
    slope: View<'_>,
    units: SlopeUnits,
    widths: [u8; 7],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::tilt::finney_tilt_array_at_widths(length, height, slope, units, widths)
}

/// Extended-precision counterpart of [`crate::tilt::finney_tilt_array_with_equality`].
pub fn finney_tilt_array_with_equality(
    length: View<'_>,
    height: View<'_>,
    slope: View<'_>,
    units: SlopeUnits,
    widths: [u8; 7],
    equality: ArrayViewD<'_, Cell<bool>>,
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::tilt::finney_tilt_array_at_widths_with_equality(
        length, height, slope, units, widths, equality,
    )
}

/// Extended-precision counterpart of [`crate::tilt::standard_tilt_array_with_precision`].
pub fn standard_tilt_array_with_precision(
    length: View<'_>,
    height: View<'_>,
    width: u8,
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::tilt::standard_tilt_array_at_widths(length, height, width)
}

/// Extended-precision counterpart of [`crate::tilt::butler_tilt_array_with_precision`].
pub fn butler_tilt_array_with_precision(
    wind_speed: View<'_>,
    units: WindSpeedUnits,
    canopy_ht: View<'_>,
    widths: [u8; 10],
) -> Result<ArrayD<Cell<LongDouble>>, Error> {
    crate::tilt::butler_tilt_array_at_widths(wind_speed, units, canopy_ht, widths)
}
