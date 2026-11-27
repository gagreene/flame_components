//! Floating-point evaluation at the width chosen for each array operation.

use half::f16;

use crate::{Cell, Error, SlopeUnits};

pub(crate) trait ModelValue: Copy + PartialOrd {
    type Stage: Stage<Self>;

    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

pub(crate) fn constant<T: ModelValue>(value: f64) -> Cell<T> {
    Cell::present(T::from_f64(value))
}

impl ModelValue for f64 {
    type Stage = Precision;

    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Operations at a single stage of a model's precision plan.
pub(crate) trait Stage<T: ModelValue>: Copy {
    fn widths<const N: usize>(widths: [u8; N]) -> Result<[Self; N], Error>;
    fn convert(self, cell: Cell<T>) -> Cell<T>;
    fn add(self, left: Cell<T>, right: Cell<T>) -> Cell<T>;
    fn sub(self, left: Cell<T>, right: Cell<T>) -> Cell<T>;
    fn mul(self, left: Cell<T>, right: Cell<T>) -> Cell<T>;
    fn div(self, left: Cell<T>, right: Cell<T>) -> Cell<T>;
    fn pow(self, base: Cell<T>, exponent: f64) -> Cell<T>;
    fn sin(self, value: Cell<T>) -> Cell<T>;
    fn sqrt(self, value: Cell<T>) -> Cell<T>;
    fn log(self, value: Cell<T>) -> Cell<T>;
    fn atan(self, value: Cell<T>) -> Cell<T>;
    fn asin(self, value: Cell<T>) -> Cell<T>;
    fn acos(self, value: Cell<T>) -> Cell<T>;
    fn radians(self, value: Cell<T>) -> Cell<T>;
    fn degrees(self, value: Cell<T>) -> Cell<T>;

    fn slope(self, angle: Cell<T>, units: SlopeUnits) -> Cell<T> {
        match units {
            SlopeUnits::Degrees => self.radians(angle),
            SlopeUnits::Percent => self.atan(self.div(angle, constant(100.0))),
        }
    }
}

/// NumPy's safe-divide cutoff can have a different dtype from the quotient.
/// The low five bits encode the quotient width in bytes. The upper bits select
/// the cutoff arithmetic or ordinary scalar division. Zero preserves the
/// explicit-width Rust caller behavior.
#[derive(Clone, Copy)]
pub(crate) enum DivideCutoff {
    Legacy,
    RoundedToZero,
    Double,
    Native,
    Scalar,
}

#[derive(Clone, Copy)]
pub(crate) struct Division {
    pub(crate) cutoff: DivideCutoff,
    restore_numerator: bool,
}

impl Division {
    pub(crate) const LEGACY: Self = Self {
        cutoff: DivideCutoff::Legacy,
        restore_numerator: false,
    };

    pub(crate) fn decode(stage: u8) -> (u8, Self) {
        let flag = stage >> 5;
        let cutoff = match flag {
            0 => DivideCutoff::Legacy,
            1 | 5 => DivideCutoff::RoundedToZero,
            2 | 6 => DivideCutoff::Double,
            4 => DivideCutoff::Scalar,
            _ => DivideCutoff::Native, // 3 or 7
        };
        (
            stage & 31,
            Self {
                cutoff,
                restore_numerator: flag >= 5,
            },
        )
    }

    pub(crate) fn finish<T: ModelValue>(self, mut quotient: Cell<T>, numerator: T) -> Cell<T> {
        // NumPy masked-array division adds mask * numerator back to the result.
        // For an unmasked zero, adding a positive zero clears its sign bit.
        if self.restore_numerator && !quotient.masked {
            let zero = T::from_f64(0.0);
            if quotient.value == zero && numerator.to_f64().is_sign_positive() {
                quotient.value = zero;
            }
        }
        quotient
    }
}

#[derive(Clone, Copy)]
enum Width {
    Half,
    Single,
    Double,
}

#[derive(Clone, Copy)]
pub(crate) struct Precision {
    width: Width,
    division: Division,
}

impl Precision {
    pub(crate) const DOUBLE: Self = Self {
        width: Width::Double,
        division: Division::LEGACY,
    };

    fn from_stage(stage: u8) -> Result<Self, Error> {
        let (width, division) = Division::decode(stage);
        let width = match width {
            2 => Width::Half,
            4 => Width::Single,
            8 => Width::Double,
            _ => return Err(Error::InvalidFloatWidth(stage)),
        };
        Ok(Self { width, division })
    }

    #[cfg(feature = "extended")]
    pub(crate) fn is_single(self) -> bool {
        matches!(self.width, Width::Single)
    }

    pub(crate) fn quotient(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a / b)
    }

    fn cast(self, value: f64) -> f64 {
        match self.width {
            Width::Half => f16::from_f64(value).to_f64(),
            Width::Single => f64::from(value as f32),
            Width::Double => value,
        }
    }

    fn unary(
        self,
        input: Cell<f64>,
        single: impl FnOnce(f32) -> f32,
        double: impl FnOnce(f64) -> f64,
    ) -> Cell<f64> {
        input.map(|value| match self.width {
            Width::Double => double(value),
            _ => self.cast(f64::from(single(self.cast(value) as f32))),
        })
    }

    fn binary(
        self,
        left: Cell<f64>,
        right: Cell<f64>,
        op: impl FnOnce(f64, f64) -> f64,
    ) -> Cell<f64> {
        Cell {
            value: self.cast(op(self.cast(left.value), self.cast(right.value))),
            masked: left.masked || right.masked,
        }
    }
}

impl Stage<f64> for Precision {
    fn widths<const N: usize>(widths: [u8; N]) -> Result<[Self; N], Error> {
        let mut result = [Self::DOUBLE; N];
        for (slot, stage) in result.iter_mut().zip(widths) {
            *slot = Self::from_stage(stage)?;
        }
        Ok(result)
    }

    fn convert(self, cell: Cell<f64>) -> Cell<f64> {
        cell.map(|value| self.cast(value))
    }

    fn add(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a + b)
    }

    fn sub(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a - b)
    }

    fn mul(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a * b)
    }

    fn div(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        let mut quotient = self.quotient(left, right);
        if !quotient.masked {
            quotient.masked = match self.division.cutoff {
                DivideCutoff::Scalar => false,
                _ if !quotient.value.is_finite() => true,
                DivideCutoff::RoundedToZero => false,
                DivideCutoff::Legacy => {
                    self.cast(left.value).abs() * f64::MIN_POSITIVE >= self.cast(right.value).abs()
                }
                DivideCutoff::Double | DivideCutoff::Native => {
                    left.value.abs() * f64::MIN_POSITIVE >= right.value.abs()
                }
            };
        }
        self.division.finish(quotient, left.value)
    }

    fn pow(self, base: Cell<f64>, exponent: f64) -> Cell<f64> {
        let mut result = self.unary(
            base,
            |x| x.powf(self.cast(exponent) as f32),
            |x| x.powf(exponent),
        );
        result.masked |= !result.value.is_finite();
        result
    }

    fn sin(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::sin, f64::sin)
    }

    fn sqrt(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::sqrt, f64::sqrt);
        result.masked |= !result.value.is_finite();
        result
    }

    fn log(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::ln, f64::ln);
        result.masked |= !result.value.is_finite();
        result
    }

    fn atan(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::atan, f64::atan)
    }

    fn asin(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::asin, f64::asin);
        result.masked |= !result.value.is_finite();
        result
    }

    fn acos(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::acos, f64::acos);
        result.masked |= !result.value.is_finite();
        result
    }

    fn radians(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::to_radians, f64::to_radians)
    }

    fn degrees(self, value: Cell<f64>) -> Cell<f64> {
        // NumPy's binary32 loop forms 180 / pi in single precision.
        self.unary(
            value,
            |x| x * (180.0 / std::f32::consts::PI),
            f64::to_degrees,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_zero_signs<T: ModelValue>(width: u8) {
        for flag in 0..=7 {
            let [stage] = T::Stage::widths([width | (flag << 5)]).unwrap();
            for (numerator, denominator, negative) in [
                (0.0, -2.0, flag < 5),
                (-0.0, 2.0, true),
                (-0.0, -2.0, false),
                (1.0, f64::NEG_INFINITY, flag < 5),
                (-1.0, f64::INFINITY, true),
            ] {
                let result = stage.div(constant(numerator), constant(denominator));
                assert!(!result.masked);
                assert_eq!(result.value.to_f64(), 0.0);
                assert_eq!(result.value.to_f64().is_sign_negative(), negative);
            }
        }
    }

    #[test]
    fn array_division_restores_the_numerators_zero_sign() {
        for width in [2, 4, 8] {
            check_zero_signs::<f64>(width);
        }
    }

    #[cfg(feature = "extended")]
    #[test]
    fn extended_array_division_restores_the_numerators_zero_sign() {
        for width in [4, crate::extended::format().0 as u8] {
            check_zero_signs::<crate::extended::LongDouble>(width);
        }
    }
}
