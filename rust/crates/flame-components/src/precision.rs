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

#[derive(Clone, Copy)]
pub(crate) enum Precision {
    Half,
    Single,
    Double,
}

impl Precision {
    fn cast(self, value: f64) -> f64 {
        match self {
            Self::Half => f16::from_f64(value).to_f64(),
            Self::Single => f64::from(value as f32),
            Self::Double => value,
        }
    }

    fn unary(
        self,
        input: Cell<f64>,
        single: impl FnOnce(f32) -> f32,
        double: impl FnOnce(f64) -> f64,
    ) -> Cell<f64> {
        input.map(|value| match self {
            Self::Double => double(value),
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
        let mut result = [Self::Double; N];
        for (slot, width) in result.iter_mut().zip(widths) {
            *slot = match width {
                2 => Self::Half,
                4 => Self::Single,
                8 => Self::Double,
                _ => return Err(Error::InvalidFloatWidth(width)),
            };
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
        let left = left.map(|x| self.cast(x));
        let right = right.map(|x| self.cast(x));
        let mut quotient = (left / right).map(|x| self.cast(x));
        quotient.masked |= !quotient.value.is_finite();
        quotient
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
