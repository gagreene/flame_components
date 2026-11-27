//! Translate Rust broadcasting failures to the NumPy expression's error.

use flame_components::Error;
use pyo3::exceptions::{PyKeyError, PyMemoryError, PyValueError};
use pyo3::PyErr;

pub(super) enum ShapeModel {
    Wind,
    NelsonHeight,
    FinneyHeight,
    StandardTilt,
    FinneyTilt,
    Residence,
}

fn shape_text(shape: &[usize]) -> String {
    let dimensions: Vec<_> = shape.iter().map(usize::to_string).collect();
    let trailing_comma = if dimensions.len() == 1 { "," } else { "" };
    format!("({}{trailing_comma})", dimensions.join(","))
}

fn broadcast_error(shapes: &[&[usize]]) -> PyErr {
    let shapes: Vec<_> = shapes.iter().map(|shape| shape_text(shape)).collect();
    PyValueError::new_err(format!(
        "operands could not be broadcast together with shapes {} ",
        shapes.join(" "),
    ))
}

fn broadcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>, PyErr> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1; rank];
    for shape in shapes {
        for (output_len, &input_len) in output.iter_mut().rev().zip(shape.iter().rev()) {
            if *output_len == 1 {
                *output_len = input_len;
            } else if input_len != 1 && *output_len != input_len {
                return Err(broadcast_error(shapes));
            }
        }
    }
    Ok(output)
}

fn check_shape_order(model: ShapeModel, shapes: &[&[usize]]) -> Result<(), PyErr> {
    // Replay only the shape joins of operations in the NumPy reference order.
    // The Rust kernels broadcast once, so this runs only after they report a failure.
    match (model, shapes) {
        (ShapeModel::Wind, &[wind, cover, height, base]) => {
            let crown = broadcast_shape(&[height, base])?;
            let fraction = broadcast_shape(&[&crown, cover])?;
            broadcast_shape(&[wind, height])?;
            broadcast_shape(&[wind, &fraction])?;
        }
        (ShapeModel::NelsonHeight, &[length, fire_type, intensity, wind]) => {
            let numerator = broadcast_shape(&[fire_type, intensity])?;
            let height = broadcast_shape(&[&numerator, wind])?;
            broadcast_shape(&[wind, length, &height])?;
        }
        (ShapeModel::FinneyHeight, &[length, tilt, slope]) => {
            broadcast_shape(&[length, tilt])?;
            let angle = broadcast_shape(&[tilt, slope])?;
            broadcast_shape(&[length, &angle])?;
        }
        (ShapeModel::StandardTilt, &[length, height]) => {
            broadcast_shape(&[height, length])?;
        }
        (ShapeModel::FinneyTilt, &[length, height, slope, ..]) => {
            let numerator = broadcast_shape(&[height, slope])?;
            broadcast_shape(&[&numerator, length])?;
        }
        (ShapeModel::Residence, &[ros, fuel, wind]) => {
            let numerator = broadcast_shape(&[fuel, wind])?;
            broadcast_shape(&[&numerator, ros])?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn python_error_for(model: ShapeModel, error: Error) -> PyErr {
    match error {
        Error::IncompatibleShapes { shapes } => {
            let shapes: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
            check_shape_order(model, &shapes)
                .err()
                .unwrap_or_else(|| broadcast_error(&shapes))
        }
        _ => python_error(error),
    }
}

pub(super) fn python_error(error: Error) -> PyErr {
    match error {
        Error::OutputTooLarge { .. } => PyMemoryError::new_err(error.to_string()),
        Error::UnknownLengthModel(model) => PyKeyError::new_err(model),
        Error::MissingFlameDepth => {
            PyValueError::new_err("The Finney_HEAD model requires flame_depth")
        }
        Error::NonPositiveCanopyHeight(_) => {
            PyValueError::new_err("canopy_ht must be greater than zero")
        }
        Error::InvalidFireType(_) => PyValueError::new_err(
            "fire_type must be 1 (surface), 2 (passive crown), or 3 (active crown)",
        ),
        Error::IncompatibleShapes { shapes } => {
            broadcast_error(&shapes.iter().map(Vec::as_slice).collect::<Vec<_>>())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}
