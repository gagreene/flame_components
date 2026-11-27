//! Flame depth in metres from spread rate and residence time.

use ndarray::{ArrayD, ArrayViewD};

use crate::{array::broadcast, Cell, Error};

/// Arithmetic supported by the depth equation and the nonnegative clamp.
pub trait DepthValue: Copy {
    /// Multiply at the given precision. Integer arithmetic wraps at its width.
    fn depth_product(self, time: Self) -> Self;

    /// Clamp the value using the NumPy reference's comparison with zero.
    fn clamp_nonnegative(self) -> Self;
}

impl DepthValue for bool {
    fn depth_product(self, time: Self) -> Self {
        self && time
    }

    fn clamp_nonnegative(self) -> Self {
        self
    }
}

macro_rules! integer_depth {
    ($($kind:ty),+) => {
        $(impl DepthValue for $kind {
            fn depth_product(self, time: Self) -> Self {
                self.wrapping_mul(time)
            }

            fn clamp_nonnegative(self) -> Self {
                self.max(0)
            }
        })+
    };
}

integer_depth!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! float_depth {
    ($($kind:ty),+) => {
        $(impl DepthValue for $kind {
            fn depth_product(self, time: Self) -> Self {
                self * time
            }

            fn clamp_nonnegative(self) -> Self {
                if self > 0.0 { self } else { 0.0 }
            }
        })+
    };
}

float_depth!(f32, f64);

impl<T: DepthValue> Cell<T> {
    /// Apply the final nonnegative clamp while retaining the value's type and mask.
    pub fn nonnegative(self) -> Self {
        self.map(T::clamp_nonnegative)
    }
}

/// Fons et al. (1963) depth equation with rate in m/min and time in minutes.
///
/// Both operands use the same arithmetic type. Callers choose the common type
/// before calling this function. The result preserves that type and combines
/// both input masks.
pub fn flame_depth<T: DepthValue>(ros: Cell<T>, res_time: Cell<T>) -> Cell<T> {
    Cell {
        value: ros.value.depth_product(res_time.value),
        masked: ros.masked || res_time.masked,
    }
}

/// Apply [`flame_depth`] to broadcast-compatible rate and time arrays.
pub fn flame_depth_array<T: DepthValue>(
    ros: ArrayViewD<'_, Cell<T>>,
    res_time: ArrayViewD<'_, Cell<T>>,
) -> Result<ArrayD<Cell<T>>, Error> {
    broadcast([ros.view(), res_time.view()], |[ros, res_time]| {
        flame_depth(ros, res_time)
    })
}
