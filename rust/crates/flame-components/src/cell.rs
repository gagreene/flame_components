use std::ops::{Add, Div, Mul, Sub};

use ndarray::NdFloat;

/// A value with an independent missing-data flag.
///
/// The payload of a masked equation result is unspecified. For floating-point
/// inputs, use [`Cell::input`] to mask NaN values at the calculation boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell<T> {
    pub value: T,
    pub masked: bool,
}

impl<T> Cell<T> {
    /// Construct an unmasked value.
    pub const fn present(value: T) -> Self {
        Self {
            value,
            masked: false,
        }
    }

    /// Construct a masked value.
    pub const fn missing(value: T) -> Self {
        Self {
            value,
            masked: true,
        }
    }

    /// Transform the payload while preserving the mask.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Cell<U> {
        Cell {
            value: f(self.value),
            masked: self.masked,
        }
    }

    /// Select the branch and mask for one cell of a masked condition.
    pub fn select(condition: Cell<bool>, yes: Self, no: Self) -> Self {
        let mut selected = if condition.value { yes } else { no };
        selected.masked |= condition.masked;
        selected
    }
}

impl From<f64> for Cell<f64> {
    fn from(value: f64) -> Self {
        Self::present(value)
    }
}

impl<F: NdFloat> Cell<F> {
    /// Normalize a floating-point input. NaN denotes missing data.
    pub fn input(value: F) -> Self {
        Self {
            value,
            masked: value.is_nan(),
        }
    }

    fn finite_result(self, f: impl FnOnce(F) -> F) -> Self {
        let value = f(self.value);
        Self {
            value,
            masked: self.masked || !value.is_finite(),
        }
    }

    pub(crate) fn acos(self) -> Self {
        self.finite_result(F::acos)
    }

    pub(crate) fn degrees(self) -> Self {
        self.map(|radians| {
            let degrees = radians.to_f64().unwrap() * (180.0 / std::f64::consts::PI);
            F::from(degrees).unwrap()
        })
    }
}

impl Cell<f64> {
    fn binary<U>(self, rhs: Self, f: impl FnOnce(f64, f64) -> U) -> Cell<U> {
        Cell {
            value: f(self.value, rhs.value),
            masked: self.masked || rhs.masked,
        }
    }

    pub(crate) fn eq_cell(self, rhs: impl Into<Self>) -> Cell<bool> {
        self.binary(rhs.into(), |a, b| a == b)
    }

    pub(crate) fn le(self, rhs: impl Into<Self>) -> Cell<bool> {
        self.binary(rhs.into(), |a, b| a <= b)
    }

    pub(crate) fn gt(self, rhs: impl Into<Self>) -> Cell<bool> {
        self.binary(rhs.into(), |a, b| a > b)
    }

    pub(crate) fn sqrt(self) -> Self {
        self.finite_result(f64::sqrt)
    }

    pub(crate) fn log(self) -> Self {
        self.finite_result(f64::ln)
    }

    pub(crate) fn powf(self, exponent: f64) -> Self {
        self.finite_result(|value| value.powf(exponent))
    }

    pub(crate) fn asin(self) -> Self {
        self.finite_result(f64::asin)
    }

    pub(crate) fn atan(self) -> Self {
        self.map(f64::atan)
    }

    pub(crate) fn sin(self) -> Self {
        self.map(f64::sin)
    }

    pub(crate) fn radians(self) -> Self {
        self.map(f64::to_radians)
    }
}

macro_rules! cell_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Cell<f64> {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self {
                self.binary(rhs, |a, b| a $op b)
            }
        }
        cell_scalar_operands!($trait, $method);
    };
}

macro_rules! cell_scalar_operands {
    ($trait:ident, $method:ident) => {
        impl $trait<f64> for Cell<f64> {
            type Output = Self;

            fn $method(self, rhs: f64) -> Self {
                self.$method(Self::present(rhs))
            }
        }

        impl $trait<Cell<f64>> for f64 {
            type Output = Cell<f64>;

            fn $method(self, rhs: Cell<f64>) -> Cell<f64> {
                Cell::present(self).$method(rhs)
            }
        }
    };
}

cell_binary_op!(Add, add, +);
cell_binary_op!(Sub, sub, -);
cell_binary_op!(Mul, mul, *);

impl<F: NdFloat> Div for Cell<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        let numerator = self.value.abs().to_f64().unwrap();
        let denominator = rhs.value.abs().to_f64().unwrap();
        // NumPy ma uses binary64's minimum normal for its safe-divide check.
        Self {
            value,
            masked: self.masked
                || rhs.masked
                || !value.is_finite()
                || numerator * f64::MIN_POSITIVE >= denominator,
        }
    }
}

cell_scalar_operands!(Div, div);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_masks_domain_and_non_finite_quotients() {
        let one = Cell::present(1.0);
        assert!((one / 0.0).masked);
        assert!((one / f64::MIN_POSITIVE).masked);
        assert!(!(one / (2.0 * f64::MIN_POSITIVE)).masked);
        assert_eq!(one / f64::INFINITY, Cell::present(0.0));
        assert!((f64::INFINITY / one).masked);
        assert!((one / Cell::missing(2.0)).masked);
        let small = Cell::present(1e-20_f32) / Cell::present(1e-40_f32);
        assert!(!small.masked);
        assert!(small.value.is_finite());
    }

    #[test]
    fn operation_domains_and_unrestricted_arithmetic_have_distinct_masks() {
        let infinity = Cell::input(f64::INFINITY);
        assert!(!(infinity * 0.0).masked);
        assert!((infinity * 0.0).value.is_nan());
        assert!(infinity.sqrt().masked);
        assert!(Cell::present(0.0).log().masked);
        assert!(Cell::present(1.01).acos().masked);
        assert_eq!(Cell::present(-2.0).powf(3.0), Cell::present(-8.0));
        assert!(Cell::present(-2.0).powf(0.5).masked);
    }

    #[test]
    fn selection_uses_only_the_taken_branch_and_condition_mask() {
        assert_eq!(
            Cell::select(Cell::present(true), Cell::present(2), Cell::missing(1)),
            Cell::present(2)
        );
        assert_eq!(
            Cell::select(Cell::present(false), Cell::missing(2), Cell::present(1)),
            Cell::present(1)
        );
        assert!(Cell::select(Cell::missing(true), Cell::present(2), Cell::present(1)).masked);
    }
}
