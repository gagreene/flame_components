//! Native C long double arithmetic.
//!
//! Values cross the Rust/C boundary as native-endian byte buffers. C owns the
//! floating-point representation, alignment, arithmetic and direct conversions.

use std::cmp::Ordering;
use std::ffi::c_int;

/// Maximum object size across the supported C long double representations.
pub const BYTE_CAPACITY: usize = 16;

/// A native long double with zeroed padding.
#[derive(Clone, Copy, Debug)]
pub struct LongDouble([u8; BYTE_CAPACITY]);

extern "C" {
    fn fc_wide_format(precision: *mut c_int, min_exp: *mut c_int, max_exp: *mut c_int) -> usize;
    fn fc_wide_from_f64(out: *mut u8, value: f64);
    fn fc_wide_to_f64(input: *const u8) -> f64;
    fn fc_wide_to_f32(input: *const u8) -> f32;
    fn fc_wide_is_finite(input: *const u8) -> c_int;
    fn fc_wide_is_nan(input: *const u8) -> c_int;
    fn fc_wide_compare(left: *const u8, right: *const u8) -> c_int;
    fn fc_wide_import(out: *mut u8, input: *const u8);
}

/// Returns the native object size, significand bits (including the integer bit),
/// minimum normal binary exponent and exclusive maximum binary exponent.
pub fn format() -> (usize, u32, i32, i32) {
    let (mut precision, mut min_exp, mut max_exp) = (0, 0, 0);
    // SAFETY: Each pointer refers to a distinct writable C int for this call.
    let size = unsafe { fc_wide_format(&mut precision, &mut min_exp, &mut max_exp) };
    (size, precision as u32, min_exp, max_exp)
}

impl LongDouble {
    /// Imports a native-endian C object. Padding is discarded. Unsupported x87
    /// encodings become NaN and pseudo-denormals become equivalent normal values.
    pub fn from_bytes(bytes: [u8; BYTE_CAPACITY]) -> Self {
        let mut result = Self([0; BYTE_CAPACITY]);
        // SAFETY: Both buffers are disjoint and hold BYTE_CAPACITY bytes.
        unsafe { fc_wide_import(result.0.as_mut_ptr(), bytes.as_ptr()) };
        result
    }

    /// Returns native-endian value bytes followed by zero padding.
    pub fn to_bytes(self) -> [u8; BYTE_CAPACITY] {
        self.0
    }

    pub fn from_f64(value: f64) -> Self {
        let mut result = Self([0; BYTE_CAPACITY]);
        // SAFETY: The output buffer holds BYTE_CAPACITY writable bytes.
        unsafe { fc_wide_from_f64(result.0.as_mut_ptr(), value) };
        result
    }

    pub fn to_f64(self) -> f64 {
        // SAFETY: The input contains an initialized, canonical native value.
        unsafe { fc_wide_to_f64(self.0.as_ptr()) }
    }

    pub fn to_f32(self) -> f32 {
        // SAFETY: The input contains an initialized, canonical native value.
        unsafe { fc_wide_to_f32(self.0.as_ptr()) }
    }

    pub fn is_finite(self) -> bool {
        // SAFETY: The input contains an initialized, canonical native value.
        unsafe { fc_wide_is_finite(self.0.as_ptr()) != 0 }
    }

    pub fn is_nan(self) -> bool {
        // SAFETY: The input contains an initialized, canonical native value.
        unsafe { fc_wide_is_nan(self.0.as_ptr()) != 0 }
    }
}

macro_rules! unary_methods {
    ($($method:ident => $native:ident),+ $(,)?) => {
        extern "C" {
            $(fn $native(out: *mut u8, input: *const u8);)+
        }
        impl LongDouble {
            $(pub fn $method(self) -> Self {
                let mut result = Self([0; BYTE_CAPACITY]);
                // SAFETY: The output and initialized native input are distinct
                // BYTE_CAPACITY-byte buffers. The C function retains neither.
                unsafe { $native(result.0.as_mut_ptr(), self.0.as_ptr()) };
                result
            })+
        }
    };
}

macro_rules! binary_methods {
    ($($method:ident => $native:ident),+ $(,)?) => {
        extern "C" {
            $(fn $native(out: *mut u8, left: *const u8, right: *const u8);)+
        }
        // The model kernels select arithmetic through named operations.
        #[allow(clippy::should_implement_trait)]
        impl LongDouble {
            $(pub fn $method(self, other: Self) -> Self {
                let mut result = Self([0; BYTE_CAPACITY]);
                // SAFETY: Both native inputs are initialized. The output is a
                // separate BYTE_CAPACITY-byte buffer. C retains no pointers.
                unsafe { $native(result.0.as_mut_ptr(), self.0.as_ptr(), other.0.as_ptr()) };
                result
            })+
        }
    };
}

unary_methods! {
    abs => fc_wide_abs,
    sin => fc_wide_sin,
    sqrt => fc_wide_sqrt,
    log => fc_wide_log,
    atan => fc_wide_atan,
    asin => fc_wide_asin,
    acos => fc_wide_acos,
    radians => fc_wide_radians,
    degrees => fc_wide_degrees,
}

binary_methods! {
    add => fc_wide_add,
    sub => fc_wide_sub,
    mul => fc_wide_mul,
    div => fc_wide_div,
    pow => fc_wide_pow,
}

impl PartialEq for LongDouble {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for LongDouble {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // SAFETY: Both inputs contain initialized, canonical native values.
        match unsafe { fc_wide_compare(self.0.as_ptr(), other.0.as_ptr()) } {
            -1 => Some(Ordering::Less),
            0 => Some(Ordering::Equal),
            1 => Some(Ordering::Greater),
            _ => None,
        }
    }
}
