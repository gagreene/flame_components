//! Floating-point evaluation at the width chosen for each array operation.

use half::f16;

use crate::{Cell, Error, SlopeUnits};

#[derive(Clone, Copy)]
pub(crate) enum Precision {
    Half,
    Single,
    Double,
}

impl Precision {
    pub(crate) fn widths<const N: usize>(widths: [u8; N]) -> Result<[Self; N], Error> {
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

    fn cast(self, value: f64) -> f64 {
        match self {
            Self::Half => f16::from_f64(value).to_f64(),
            Self::Single => f64::from(value as f32),
            Self::Double => value,
        }
    }

    pub(crate) fn convert(self, cell: Cell<f64>) -> Cell<f64> {
        cell.map(|value| self.cast(value))
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

    pub(crate) fn add(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a + b)
    }

    pub(crate) fn sub(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a - b)
    }

    pub(crate) fn mul(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        self.binary(left, right, |a, b| a * b)
    }

    pub(crate) fn div(self, left: Cell<f64>, right: Cell<f64>) -> Cell<f64> {
        let left = left.map(|x| self.cast(x));
        let right = right.map(|x| self.cast(x));
        let mut quotient = (left / right).map(|x| self.cast(x));
        quotient.masked |= !quotient.value.is_finite();
        quotient
    }

    pub(crate) fn pow(self, base: Cell<f64>, exponent: f64) -> Cell<f64> {
        let mut result = self.unary(
            base,
            |x| x.powf(self.cast(exponent) as f32),
            |x| x.powf(exponent),
        );
        result.masked |= !result.value.is_finite();
        result
    }

    pub(crate) fn sin(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::sin, f64::sin)
    }

    pub(crate) fn sqrt(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::sqrt, f64::sqrt);
        result.masked |= !result.value.is_finite();
        result
    }

    pub(crate) fn log(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::ln, f64::ln);
        result.masked |= !result.value.is_finite();
        result
    }

    pub(crate) fn atan(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::atan, f64::atan)
    }

    pub(crate) fn asin(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::asin, f64::asin);
        result.masked |= !result.value.is_finite();
        result
    }

    pub(crate) fn acos(self, value: Cell<f64>) -> Cell<f64> {
        let mut result = self.unary(value, f32::acos, f64::acos);
        result.masked |= !result.value.is_finite();
        result
    }

    pub(crate) fn radians(self, value: Cell<f64>) -> Cell<f64> {
        self.unary(value, f32::to_radians, f64::to_radians)
    }

    pub(crate) fn degrees(self, value: Cell<f64>) -> Cell<f64> {
        // NumPy's binary32 loop forms 180 / pi in single precision.
        self.unary(
            value,
            |x| x * (180.0 / std::f32::consts::PI),
            f64::to_degrees,
        )
    }

    pub(crate) fn slope(self, angle: Cell<f64>, units: SlopeUnits) -> Cell<f64> {
        match units {
            SlopeUnits::Degrees => self.radians(angle),
            SlopeUnits::Percent => self.unary(self.div(angle, 100.0.into()), f32::atan, f64::atan),
        }
    }
}
