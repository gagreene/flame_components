//! NumPy adapters for the standalone flame-component kernels.

#![forbid(unsafe_code)]

mod errors;

use std::mem::{align_of, size_of};

use errors::{python_error, python_error_for, ShapeModel};
use flame_components::depth::{flame_depth_array, DepthValue};
use flame_components::height::{self, HeightModel};
use flame_components::length::{self, LengthModel};
use flame_components::residence::{self, ResidenceUnits};
use flame_components::tilt::{self, TiltModel};
use flame_components::wind::{self, WindUnits};
use flame_components::{Cell, Error, SlopeUnits, WindSpeedUnits};
use ndarray::{ArrayD, Zip};
use numpy::{Element, IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

type Buffers<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>);
type Output<'py> = (Bound<'py, PyAny>, Bound<'py, PyArrayDyn<bool>>, bool);

fn read_cells<T: Element + Copy>(buffers: &Buffers<'_>) -> PyResult<ArrayD<Cell<T>>> {
    read_cells_as(buffers, |value: T| value)
}

fn read_cells_as<T: Element + Copy, U>(
    buffers: &Buffers<'_>,
    convert: impl Fn(T) -> U,
) -> PyResult<ArrayD<Cell<U>>> {
    let data = buffers.0.cast::<PyArrayDyn<T>>()?;
    read_with_mask(data, &buffers.1, convert)
}

fn bool_bytes<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let data = array.cast::<PyArrayDyn<bool>>()?;
    let array_type = data.py().get_type::<PyArrayDyn<bool>>();
    // NumPy treats any nonzero byte as true. Rust bool only permits 0 and 1.
    Ok(array_type
        .call_method1("view", (data, "uint8"))?
        .cast_into::<PyArrayDyn<u8>>()?)
}

fn read_with_mask<T: Element + Copy, U>(
    data: &Bound<'_, PyArrayDyn<T>>,
    mask: &Bound<'_, PyAny>,
    convert: impl Fn(T) -> U,
) -> PyResult<ArrayD<Cell<U>>> {
    let mask = bool_bytes(mask)?;
    if data.shape() != mask.shape() {
        return Err(PyValueError::new_err("data and mask shapes must match"));
    }
    if data.data().align_offset(align_of::<T>()) != 0
        || data
            .shape()
            .iter()
            .zip(data.strides())
            .any(|(&length, &stride)| length > 1 && stride % size_of::<T>() as isize != 0)
    {
        return Err(PyValueError::new_err("data must be aligned"));
    }
    let borrow_error = |error: numpy::BorrowError| PyValueError::new_err(error.to_string());
    let data = data.try_readonly().map_err(borrow_error)?;
    let mask = mask.try_readonly().map_err(borrow_error)?;
    Ok(Zip::from(data.as_array())
        .and(mask.as_array())
        .map_collect(|&value, &masked| Cell {
            value: convert(value),
            masked: masked != 0,
        }))
}

fn split_cells<'py, T: Element + Copy>(
    py: Python<'py>,
    cells: ArrayD<Cell<T>>,
    substituted_canopy_height: bool,
) -> Output<'py> {
    let data = cells.mapv(|cell| cell.value).into_pyarray(py).into_any();
    let mask = cells.mapv(|cell| cell.masked).into_pyarray(py);
    (data, mask, substituted_canopy_height)
}

fn required_cells<T: Element + Copy>(
    input: Option<&Buffers<'_>>,
    name: &str,
) -> PyResult<ArrayD<Cell<T>>> {
    let input = input.ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    read_cells(input)
}

fn parse_slope_units(units: Option<&str>) -> PyResult<SlopeUnits> {
    match units {
        Some("degrees") => Ok(SlopeUnits::Degrees),
        Some("percent") => Ok(SlopeUnits::Percent),
        _ => Err(PyValueError::new_err("invalid slope units")),
    }
}

fn output<'py, T: Element + Copy>(
    py: Python<'py>,
    result: Result<ArrayD<Cell<T>>, Error>,
) -> PyResult<Output<'py>> {
    result
        .map(|cells| split_cells(py, cells, false))
        .map_err(python_error)
}

fn output_for<'py, T: Element + Copy>(
    py: Python<'py>,
    model: ShapeModel,
    result: Result<ArrayD<Cell<T>>, Error>,
) -> PyResult<Output<'py>> {
    result
        .map(|cells| split_cells(py, cells, false))
        .map_err(|error| python_error_for(model, error))
}

fn depth_for_dtype<'py, T: DepthValue + Element>(
    py: Python<'py>,
    ros: &Buffers<'py>,
    res_time: &Buffers<'py>,
) -> PyResult<Output<'py>> {
    let ros = read_cells::<T>(ros)?;
    let res_time = read_cells::<T>(res_time)?;
    output(py, flame_depth_array(ros.view(), res_time.view()))
}

fn bool_depth<'py>(
    py: Python<'py>,
    ros: &Buffers<'py>,
    res_time: &Buffers<'py>,
) -> PyResult<Output<'py>> {
    let ros = read_with_mask(&bool_bytes(&ros.0)?, &ros.1, |value| value != 0)?;
    let res_time = read_with_mask(&bool_bytes(&res_time.0)?, &res_time.1, |value| value != 0)?;
    output(py, flame_depth_array(ros.view(), res_time.view()))
}

#[pyfunction]
fn flame_depth<'py>(
    py: Python<'py>,
    ros: Buffers<'py>,
    res_time: Buffers<'py>,
) -> PyResult<Output<'py>> {
    macro_rules! dispatch_depth {
        ($($dtype:ty),+ $(,)?) => {
            $(if ros.0.cast::<PyArrayDyn<$dtype>>().is_ok() {
                return depth_for_dtype::<$dtype>(py, &ros, &res_time);
            })+
        };
    }
    if ros.0.cast::<PyArrayDyn<bool>>().is_ok() {
        return bool_depth(py, &ros, &res_time);
    }
    dispatch_depth!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
    Err(PyTypeError::new_err(
        "unsupported dtype for native flame depth",
    ))
}

#[pyfunction(signature = (wind_speed, canopy_cover, canopy_ht, canopy_baseht, units, precision=None))]
fn mid_flame_ws<'py>(
    py: Python<'py>,
    wind_speed: Buffers<'py>,
    canopy_cover: Buffers<'py>,
    canopy_ht: Buffers<'py>,
    canopy_baseht: Buffers<'py>,
    units: &str,
    precision: Option<[u8; 18]>,
) -> PyResult<Output<'py>> {
    let units = match units {
        "SI" => WindUnits::Si,
        "IMP" => WindUnits::Imperial,
        _ => return Err(PyValueError::new_err("invalid wind unit system")),
    };
    let wind_speed = read_cells(&wind_speed)?;
    let canopy_cover = read_cells(&canopy_cover)?;
    let canopy_ht = read_cells(&canopy_ht)?;
    let canopy_baseht = read_cells(&canopy_baseht)?;
    let result = match precision {
        Some(widths) => wind::mid_flame_ws_array_with_precision(
            wind_speed.view(),
            canopy_cover.view(),
            canopy_ht.view(),
            canopy_baseht.view(),
            units,
            widths,
        ),
        None => wind::mid_flame_ws_array(
            wind_speed.view(),
            canopy_cover.view(),
            canopy_ht.view(),
            canopy_baseht.view(),
            units,
        ),
    };
    match result {
        Ok(result) => Ok(split_cells(
            py,
            result.value,
            result.substituted_canopy_height,
        )),
        Err(failure) => {
            let error = python_error_for(ShapeModel::Wind, failure.error);
            if failure.substituted_canopy_height {
                error.value(py).setattr("substituted_canopy_height", true)?;
            }
            Err(error)
        }
    }
}

#[pyfunction]
fn flame_length_parameters<'py>(py: Python<'py>, model: &str) -> PyResult<Bound<'py, PyTuple>> {
    let model: LengthModel = model.parse().map_err(python_error)?;
    let (coefficient, intensity, depth) = model.coefficients();
    if let Some(depth) = depth {
        PyTuple::new(py, [coefficient, intensity, depth])
    } else {
        PyTuple::new(py, [coefficient, intensity])
    }
}

#[pyfunction(signature = (model, fire_intensity, flame_depth, precision=None))]
fn flame_length<'py>(
    py: Python<'py>,
    model: &str,
    fire_intensity: Buffers<'py>,
    flame_depth: Option<Buffers<'py>>,
    precision: Option<[u8; 4]>,
) -> PyResult<Output<'py>> {
    let model: LengthModel = model.parse().map_err(python_error)?;
    let fire_intensity = read_cells(&fire_intensity)?;
    let flame_depth = flame_depth.as_ref().map(read_cells).transpose()?;
    let flame_depth = flame_depth.as_ref().map(ArrayD::view);
    let result = match precision {
        Some(widths) => length::flame_length_array_with_precision(
            model,
            fire_intensity.view(),
            flame_depth,
            widths,
        ),
        None => length::flame_length_array(model, fire_intensity.view(), flame_depth),
    };
    output(py, result)
}

#[pyfunction(signature = (model, flame_length, fire_type, fire_intensity, midflame_ws, flame_tilt, slope_angle, slope_units, precision=None))]
#[allow(clippy::too_many_arguments)]
fn flame_height<'py>(
    py: Python<'py>,
    model: &str,
    flame_length: Buffers<'py>,
    fire_type: Option<Buffers<'py>>,
    fire_intensity: Option<Buffers<'py>>,
    midflame_ws: Option<Buffers<'py>>,
    flame_tilt: Option<Buffers<'py>>,
    slope_angle: Option<Buffers<'py>>,
    slope_units: Option<&str>,
    precision: Option<Bound<'py, PyAny>>,
) -> PyResult<Output<'py>> {
    let length = read_cells(&flame_length)?;
    match model {
        "Nelson" => {
            let fire_type = required_cells(fire_type.as_ref(), "fire_type")?;
            let fire_intensity = required_cells(fire_intensity.as_ref(), "fire_intensity")?;
            let midflame_ws = required_cells(midflame_ws.as_ref(), "midflame_ws")?;
            let result = match precision {
                Some(widths) => height::nelson_height_array_with_precision(
                    length.view(),
                    fire_type.view(),
                    fire_intensity.view(),
                    midflame_ws.view(),
                    widths.extract::<[u8; 3]>()?,
                ),
                None => height::flame_height_array(
                    length.view(),
                    HeightModel::Nelson {
                        fire_type: fire_type.view(),
                        fire_intensity: fire_intensity.view(),
                        midflame_ws: midflame_ws.view(),
                    },
                ),
            };
            output_for(py, ShapeModel::NelsonHeight, result)
        }
        "Finney" => {
            let flame_tilt = required_cells(flame_tilt.as_ref(), "flame_tilt")?;
            let slope_angle = required_cells(slope_angle.as_ref(), "slope_angle")?;
            let units = parse_slope_units(slope_units)?;
            let result = match precision {
                Some(widths) => height::finney_height_array_with_precision(
                    length.view(),
                    flame_tilt.view(),
                    slope_angle.view(),
                    units,
                    widths.extract::<[u8; 8]>()?,
                ),
                None => height::flame_height_array(
                    length.view(),
                    HeightModel::Finney {
                        flame_tilt: flame_tilt.view(),
                        slope_angle: slope_angle.view(),
                        slope_units: units,
                    },
                ),
            };
            output_for(py, ShapeModel::FinneyHeight, result)
        }
        _ => Err(PyValueError::new_err("invalid flame height model")),
    }
}

#[pyfunction(signature = (model, flame_length, flame_height, slope_angle, slope_units, wind_speed, wind_speed_units, canopy_ht, precision=None))]
#[allow(clippy::too_many_arguments)]
fn flame_tilt<'py>(
    py: Python<'py>,
    model: &str,
    flame_length: Option<Buffers<'py>>,
    flame_height: Option<Buffers<'py>>,
    slope_angle: Option<Buffers<'py>>,
    slope_units: Option<&str>,
    wind_speed: Option<Buffers<'py>>,
    wind_speed_units: Option<&str>,
    canopy_ht: Option<Buffers<'py>>,
    precision: Option<Bound<'py, PyAny>>,
) -> PyResult<Output<'py>> {
    match model {
        "Standard" => {
            let length = flame_length
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing flame_length"))?;
            let height = flame_height
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing flame_height"))?;
            if length.0.cast::<PyArrayDyn<f32>>().is_ok() {
                let length = read_cells_as::<f32, _>(length, f64::from)?;
                let height = read_cells_as::<f32, _>(height, f64::from)?;
                let result =
                    tilt::standard_tilt_array_with_precision(length.view(), height.view(), 4)
                        .map(|cells| cells.mapv(|cell| cell.map(|value| value as f32)));
                return output_for(py, ShapeModel::StandardTilt, result);
            }
            let length = read_cells::<f64>(length)?;
            let height = read_cells::<f64>(height)?;
            let result = match precision {
                Some(widths) => {
                    let [width, ..] = widths.extract::<[u8; 7]>()?;
                    tilt::standard_tilt_array_with_precision(length.view(), height.view(), width)
                }
                None => tilt::standard_tilt_array(length.view(), height.view()),
            };
            output_for(py, ShapeModel::StandardTilt, result)
        }
        "Finney" => {
            let flame_length = required_cells(flame_length.as_ref(), "flame_length")?;
            let flame_height = required_cells(flame_height.as_ref(), "flame_height")?;
            let slope_angle = required_cells(slope_angle.as_ref(), "slope_angle")?;
            let units = parse_slope_units(slope_units)?;
            let result = match precision {
                Some(widths) => tilt::finney_tilt_array_with_precision(
                    flame_length.view(),
                    flame_height.view(),
                    slope_angle.view(),
                    units,
                    widths.extract::<[u8; 7]>()?,
                ),
                None => tilt::flame_tilt_array(TiltModel::Finney {
                    flame_length: flame_length.view(),
                    flame_height: flame_height.view(),
                    slope_angle: slope_angle.view(),
                    slope_units: units,
                }),
            };
            output_for(py, ShapeModel::FinneyTilt, result)
        }
        "Butler" => {
            let wind_speed = required_cells(wind_speed.as_ref(), "wind_speed")?;
            let canopy_ht = required_cells(canopy_ht.as_ref(), "canopy_ht")?;
            let wind_speed_units = match wind_speed_units {
                Some("kph") => WindSpeedUnits::Kph,
                Some("mps") => WindSpeedUnits::Mps,
                Some("mph") => WindSpeedUnits::Mph,
                _ => return Err(PyValueError::new_err("invalid wind speed units")),
            };
            let result = match precision {
                Some(widths) => tilt::butler_tilt_array_with_precision(
                    wind_speed.view(),
                    wind_speed_units,
                    canopy_ht.view(),
                    widths.extract::<[u8; 10]>()?,
                ),
                None => tilt::flame_tilt_array(TiltModel::Butler {
                    wind_speed: wind_speed.view(),
                    wind_speed_units,
                    canopy_ht: canopy_ht.view(),
                }),
            };
            output(py, result)
        }
        _ => Err(PyValueError::new_err("invalid flame tilt model")),
    }
}

#[pyfunction(signature = (ros, fuel_consumption, midflame_ws, units, precision=None))]
fn flame_residence_time<'py>(
    py: Python<'py>,
    ros: Buffers<'py>,
    fuel_consumption: Buffers<'py>,
    midflame_ws: Buffers<'py>,
    units: &str,
    precision: Option<[u8; 7]>,
) -> PyResult<Output<'py>> {
    let units = match units {
        "sec" => ResidenceUnits::Seconds,
        "min" => ResidenceUnits::Minutes,
        _ => return Err(PyValueError::new_err("invalid residence time units")),
    };
    let ros = read_cells(&ros)?;
    let fuel_consumption = read_cells(&fuel_consumption)?;
    let midflame_ws = read_cells(&midflame_ws)?;
    let result = match precision {
        Some(widths) => residence::flame_residence_time_array_with_precision(
            ros.view(),
            fuel_consumption.view(),
            midflame_ws.view(),
            units,
            widths,
        ),
        None => residence::flame_residence_time_array(
            ros.view(),
            fuel_consumption.view(),
            midflame_ws.view(),
            units,
        ),
    };
    output_for(py, ShapeModel::Residence, result)
}

#[pymodule(gil_used = true)]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(flame_depth, module)?)?;
    module.add_function(wrap_pyfunction!(flame_height, module)?)?;
    module.add_function(wrap_pyfunction!(flame_length, module)?)?;
    module.add_function(wrap_pyfunction!(flame_length_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(flame_residence_time, module)?)?;
    module.add_function(wrap_pyfunction!(flame_tilt, module)?)?;
    module.add_function(wrap_pyfunction!(mid_flame_ws, module)?)?;
    Ok(())
}
