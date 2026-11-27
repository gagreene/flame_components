//! Adapt NumPy byte views to the host's long double kernels.

use flame_components::extended::{self as kernels, LongDouble, BYTE_CAPACITY};
use flame_components::{Cell, Error};
use ndarray::{Array1, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::{
    aligned_readonly, bool_bytes, parse_residence_units, parse_slope_units, parse_wind_speed_units,
    parse_wind_units, python_error, python_error_for, read_bool_cells, wind_error, ShapeModel,
};

type Cells = ArrayD<Cell<LongDouble>>;
type FlatBuffers<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Vec<usize>);
type FlatOutput<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyArrayDyn<bool>>,
    Vec<usize>,
    bool,
);

#[pyfunction]
pub(super) fn extended_format() -> Option<(usize, u32, i32, i32)> {
    let format = kernels::format();
    (format.0 > 0 && format.0 <= BYTE_CAPACITY).then_some(format)
}

fn read_cells(buffers: &FlatBuffers<'_>) -> PyResult<Cells> {
    let data = buffers.0.cast::<PyArrayDyn<u8>>()?;
    let mask = bool_bytes(&buffers.1)?;
    let size = kernels::format().0;
    let data = aligned_readonly(data)?;
    let mask = aligned_readonly(&mask)?;
    let data = data.as_array();
    let mask = mask.as_array();
    if mask.ndim() != 1 || data.shape() != [mask.len(), size] {
        return Err(PyValueError::new_err(
            "extended data shape must have one byte row per mask element",
        ));
    }
    let cells = data
        .outer_iter()
        .zip(mask.iter())
        .map(|(bytes, &masked)| {
            let mut value = [0; BYTE_CAPACITY];
            for (target, &byte) in value.iter_mut().zip(bytes.iter()) {
                *target = byte;
            }
            Cell {
                value: LongDouble::from_bytes(value),
                masked: masked != 0,
            }
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&buffers.2), cells)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn required_cells(input: Option<FlatBuffers<'_>>, name: &str) -> PyResult<Cells> {
    let input = input.ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    read_cells(&input)
}

fn split_cells(py: Python<'_>, cells: Cells, substituted: bool) -> PyResult<FlatOutput<'_>> {
    let size = kernels::format().0;
    let shape = [cells.len(), size];
    let bytes = cells
        .iter()
        .flat_map(|cell| cell.value.to_bytes().into_iter().take(size))
        .collect();
    let data = ArrayD::from_shape_vec(IxDyn(&shape), bytes)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mask = Array1::from_iter(cells.iter().map(|cell| cell.masked));
    Ok((
        data.into_pyarray(py).into_any(),
        mask.into_dyn().into_pyarray(py),
        cells.shape().to_vec(),
        substituted,
    ))
}

fn output(
    py: Python<'_>,
    result: Result<Cells, Error>,
    model: Option<ShapeModel>,
) -> PyResult<FlatOutput<'_>> {
    let cells = result.map_err(|error| match model {
        Some(model) => python_error_for(model, error),
        None => python_error(error),
    })?;
    split_cells(py, cells, false)
}

fn widths<const N: usize>(precision: Option<&Bound<'_, PyAny>>) -> PyResult<[u8; N]> {
    match precision {
        Some(precision) => precision.extract(),
        None => Ok([kernels::format().0 as u8; N]),
    }
}

type HeightInputs<'py> = (
    String,
    FlatBuffers<'py>,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<String>,
);

fn height<'py>(
    py: Python<'py>,
    arguments: &Bound<'py, PyTuple>,
    precision: Option<&Bound<'py, PyAny>>,
) -> PyResult<FlatOutput<'py>> {
    let (model, length, fire_type, intensity, wind, tilt, slope, units): HeightInputs<'py> =
        arguments.extract()?;
    let length = read_cells(&length)?;
    let (model, result) = match model.as_str() {
        "Nelson" => (
            ShapeModel::NelsonHeight,
            kernels::nelson_height_array_with_precision(
                length.view(),
                required_cells(fire_type, "fire_type")?.view(),
                required_cells(intensity, "fire_intensity")?.view(),
                required_cells(wind, "midflame_ws")?.view(),
                widths(precision)?,
            ),
        ),
        "Finney" => (
            ShapeModel::FinneyHeight,
            kernels::finney_height_array_with_precision(
                length.view(),
                required_cells(tilt, "flame_tilt")?.view(),
                required_cells(slope, "slope_angle")?.view(),
                parse_slope_units(units.as_deref())?,
                widths(precision)?,
            ),
        ),
        _ => return Err(PyValueError::new_err("invalid flame height model")),
    };
    output(py, result, Some(model))
}

type TiltInputs<'py> = (
    String,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<FlatBuffers<'py>>,
    Option<String>,
    Option<FlatBuffers<'py>>,
    Option<String>,
    Option<FlatBuffers<'py>>,
);

fn tilt<'py>(
    py: Python<'py>,
    arguments: &Bound<'py, PyTuple>,
    precision: Option<&Bound<'py, PyAny>>,
    equality: Option<FlatBuffers<'py>>,
) -> PyResult<FlatOutput<'py>> {
    let (model, length, height, slope, slope_units, wind, wind_units, canopy): TiltInputs<'py> =
        arguments.extract()?;
    let (model, result) = match model.as_str() {
        "Standard" => {
            let [width, ..] = widths::<7>(precision)?;
            let result = kernels::standard_tilt_array_with_precision(
                required_cells(length, "flame_length")?.view(),
                required_cells(height, "flame_height")?.view(),
                width,
            );
            (Some(ShapeModel::StandardTilt), result)
        }
        "Finney" => {
            let length = required_cells(length, "flame_length")?;
            let height = required_cells(height, "flame_height")?;
            let slope = required_cells(slope, "slope_angle")?;
            let units = parse_slope_units(slope_units.as_deref())?;
            let widths = widths(precision)?;
            let result = match equality {
                Some((data, mask, shape)) => kernels::finney_tilt_array_with_equality(
                    length.view(),
                    height.view(),
                    slope.view(),
                    units,
                    widths,
                    read_bool_cells(&(data, mask))?
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|error| PyValueError::new_err(error.to_string()))?
                        .view(),
                ),
                None => kernels::finney_tilt_array_with_precision(
                    length.view(),
                    height.view(),
                    slope.view(),
                    units,
                    widths,
                ),
            };
            (Some(ShapeModel::FinneyTilt), result)
        }
        "Butler" => {
            let wind = required_cells(wind, "wind_speed")?;
            let canopy = required_cells(canopy, "canopy_ht")?;
            let result = kernels::butler_tilt_array_with_precision(
                wind.view(),
                parse_wind_speed_units(wind_units.as_deref())?,
                canopy.view(),
                widths(precision)?,
            );
            (None, result)
        }
        _ => return Err(PyValueError::new_err("invalid flame tilt model")),
    };
    output(py, result, model)
}

#[pyfunction(signature = (equation, arguments, precision=None, *, equality=None))]
pub(super) fn extended<'py>(
    py: Python<'py>,
    equation: &str,
    arguments: &Bound<'py, PyTuple>,
    precision: Option<Bound<'py, PyAny>>,
    equality: Option<FlatBuffers<'py>>,
) -> PyResult<FlatOutput<'py>> {
    if extended_format().is_none() {
        return Err(PyTypeError::new_err(
            "unsupported native long double layout",
        ));
    }
    let precision = precision.as_ref();
    match equation {
        "flame_depth" => {
            let (ros, res_time): (FlatBuffers<'py>, FlatBuffers<'py>) = arguments.extract()?;
            let result =
                kernels::flame_depth_array(read_cells(&ros)?.view(), read_cells(&res_time)?.view());
            output(py, result, None)
        }
        "flame_length" => {
            let (model, intensity, depth): (String, FlatBuffers<'py>, Option<FlatBuffers<'py>>) =
                arguments.extract()?;
            let model = model.parse().map_err(python_error)?;
            let intensity = read_cells(&intensity)?;
            let depth = depth.as_ref().map(read_cells).transpose()?;
            let result = kernels::flame_length_array_with_precision(
                model,
                intensity.view(),
                depth.as_ref().map(ArrayD::view),
                widths(precision)?,
            );
            output(py, result, None)
        }
        "flame_height" => height(py, arguments, precision),
        "flame_tilt" => tilt(py, arguments, precision, equality),
        "flame_residence_time" => {
            let (ros, fuel, wind, units): (
                FlatBuffers<'py>,
                FlatBuffers<'py>,
                FlatBuffers<'py>,
                String,
            ) = arguments.extract()?;
            let units = parse_residence_units(&units)?;
            let result = kernels::flame_residence_time_array_with_precision(
                read_cells(&ros)?.view(),
                read_cells(&fuel)?.view(),
                read_cells(&wind)?.view(),
                units,
                widths(precision)?,
            );
            output(py, result, Some(ShapeModel::Residence))
        }
        "mid_flame_ws" => {
            let (wind, cover, canopy, base, units): (_, _, _, _, String) = arguments.extract()?;
            let units = parse_wind_units(&units)?;
            match kernels::mid_flame_ws_array_with_precision(
                read_cells(&wind)?.view(),
                read_cells(&cover)?.view(),
                read_cells(&canopy)?.view(),
                read_cells(&base)?.view(),
                units,
                widths(precision)?,
            ) {
                Ok(result) => split_cells(py, result.value, result.substituted_canopy_height),
                Err(failure) => Err(wind_error(py, failure)?),
            }
        }
        _ => Err(PyValueError::new_err("invalid extended-precision equation")),
    }
}
