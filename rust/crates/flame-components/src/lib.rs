//! Standalone flame-component calculations for surface and crown fires.
//!
//! The calculation functions operate on masked cells and multidimensional array
//! views in the units documented by each module. They return equation results;
//! use [`Cell::nonnegative`] to apply the public Python API's final clamp.

#![forbid(unsafe_code)]

mod array;
mod cell;
mod precision;

pub mod depth;
pub mod height;
pub mod length;
pub mod residence;
pub mod tilt;
pub mod wind;

pub use cell::Cell;
pub use height::FireType;
pub use ndarray;

use std::fmt;

/// Failures in requests to the standalone engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A kernel precision must be a 2-, 4-, or 8-byte IEEE floating-point type.
    InvalidFloatWidth(u8),
    /// The trailing dimensions do not broadcast.
    IncompatibleShapes { shapes: Vec<Vec<usize>> },
    /// The broadcast output is too large to allocate.
    OutputTooLarge { shape: Vec<usize> },
    /// Finney's heading-fire length model requires flame depth.
    MissingFlameDepth,
    /// Nelson's height model accepts fire type codes 1, 2, and 3.
    InvalidFireType(f64),
    /// Butler's tilt model requires positive canopy height in metres.
    NonPositiveCanopyHeight(f64),
    /// The supplied selector does not identify a published length model.
    UnknownLengthModel(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFloatWidth(width) => write!(f, "unsupported floating-point width {width}"),
            Self::IncompatibleShapes { shapes } => {
                write!(f, "input shapes cannot be broadcast together: {shapes:?}")
            }
            Self::OutputTooLarge { shape } => {
                write!(f, "broadcast output is too large to allocate: {shape:?}")
            }
            Self::MissingFlameDepth => write!(f, "Finney_HEAD requires flame depth"),
            Self::InvalidFireType(value) => {
                write!(f, "fire type must be 1, 2, or 3, received {value}")
            }
            Self::NonPositiveCanopyHeight(value) => {
                write!(f, "Butler canopy height must be positive, received {value}")
            }
            Self::UnknownLengthModel(selector) => {
                write!(f, "unknown flame length model {selector:?}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Units of the terrain slope angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeUnits {
    Degrees,
    Percent,
}

/// Units of wind speed for the Butler tilt equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindSpeedUnits {
    Kph,
    Mps,
    Mph,
}
