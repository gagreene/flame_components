use std::collections::BTreeSet;
use std::fmt::Debug;
use std::str::FromStr;

use flame_components::depth::{flame_depth_array, DepthValue};
use flame_components::height::{flame_height_array, HeightModel};
use flame_components::length::{flame_length_array, LengthModel};
use flame_components::residence::{flame_residence_time_array, ResidenceUnits};
use flame_components::tilt::{flame_tilt_array, standard_tilt_array, TiltModel};
use flame_components::wind::{mid_flame_ws_array, WindError, WindResult, WindUnits};
use flame_components::{Cell, Error, SlopeUnits, WindSpeedUnits};
use ndarray::{ArrayD, Axis, IxDyn};
use serde_json::Value;

const REFERENCE: &str = include_str!("../../../../tests/fixtures/parity/reference.jsonl");

fn cases() -> impl Iterator<Item = Value> {
    REFERENCE
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
}

fn kernel_cases(function: &'static str) -> impl Iterator<Item = Value> {
    cases().filter(move |case| case["function"] == function && case.get("kernel_result").is_some())
}

fn case_id(case: &Value) -> &str {
    case["id"].as_str().unwrap()
}

fn shape(array: &Value) -> Vec<usize> {
    array["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|size| usize::try_from(size.as_u64().unwrap()).unwrap())
        .collect()
}

fn number_text(value: &Value) -> &str {
    match value {
        Value::Bool(true) => "1",
        Value::Bool(false) => "0",
        _ => value["int"]
            .as_str()
            .or_else(|| value["float"].as_str())
            .unwrap_or_else(|| panic!("expected fixture number, got {value}")),
    }
}

fn number<T: FromStr>(value: &Value) -> T
where
    T::Err: Debug,
{
    number_text(value).parse().unwrap()
}

fn input<T: FromStr + Copy>(value: &Value) -> ArrayD<Cell<T>>
where
    T::Err: Debug,
{
    let Some(data) = value["data"].as_array() else {
        // A public scalar is normalized to shape [1]. An explicit 0-D array
        // follows the array path below and keeps shape [].
        return ArrayD::from_elem(IxDyn(&[1]), input_cell(value, false));
    };

    let mask = value["mask"].as_array();
    let cells = data
        .iter()
        .enumerate()
        .map(|(index, item)| input_cell(item, mask.is_some_and(|v| v[index] == true)))
        .collect();
    let mut array = ArrayD::from_shape_vec(IxDyn(&shape(value)), cells).unwrap();
    if value["layout"] == "reversed_first_axis" {
        // The fixture stores logical order. Reverse that order in storage,
        // then restore the logical order with a negative-stride view.
        array.invert_axis(Axis(0));
        array = array.as_standard_layout().into_owned();
        array.invert_axis(Axis(0));
        assert!(array.strides()[0] < 0);
    }
    array
}

fn input_cell<T: FromStr>(value: &Value, masked: bool) -> Cell<T>
where
    T::Err: Debug,
{
    Cell {
        value: number(value),
        masked: masked || value["float"] == "nan",
    }
}

fn assert_float(actual: f64, expected: f64, tolerance: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
    } else if !expected.is_finite() || expected == 0.0 || tolerance == 0.0 {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: got {actual}, expected {expected}"
        );
    } else {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs(),
            "{context}: got {actual}, expected {expected} (rtol {tolerance})",
        );
    }
}

trait KernelValue {
    const DTYPE: &'static str;
    fn assert_value(self, expected: &Value, tolerance: f64, context: &str);
}

impl KernelValue for f64 {
    const DTYPE: &'static str = "float64";

    fn assert_value(self, expected: &Value, tolerance: f64, context: &str) {
        assert_float(self, number(expected), tolerance, context);
    }
}

impl KernelValue for f32 {
    const DTYPE: &'static str = "float32";

    fn assert_value(self, expected: &Value, tolerance: f64, context: &str) {
        assert_float(f64::from(self), number(expected), tolerance, context);
    }
}

macro_rules! integer_kernel_value {
    ($type:ty, $dtype:literal) => {
        impl KernelValue for $type {
            const DTYPE: &'static str = $dtype;

            fn assert_value(self, expected: &Value, _tolerance: f64, context: &str) {
                assert_eq!(self, number::<Self>(expected), "{context}");
            }
        }
    };
}

integer_kernel_value!(i64, "int64");
integer_kernel_value!(u64, "uint64");
integer_kernel_value!(u8, "uint8");

fn assert_kernel<T: KernelValue + Copy>(actual: &ArrayD<Cell<T>>, case: &Value) {
    let id = case_id(case);
    let expected = &case["kernel_result"];
    if id == "depth_masked_zero_dimensional" {
        // NumPy returns its float64 masked scalar sentinel here. Rust keeps
        // the multiplication dtype even though the only cell is masked.
        assert_eq!(T::DTYPE, "int64", "{id}");
        assert_eq!(expected["dtype"], "float64", "{id}");
        assert_eq!(
            expected["mask"].as_array().unwrap(),
            &[Value::Bool(true)],
            "{id}"
        );
    } else {
        assert_eq!(T::DTYPE, expected["dtype"].as_str().unwrap(), "{id}");
    }
    assert_eq!(actual.shape(), shape(expected), "{id}");
    let data = expected["data"].as_array().unwrap();
    assert_eq!(actual.len(), data.len(), "{id}");
    let mask = expected["mask"].as_array();
    let tolerance = case["rtol"].as_f64().unwrap_or(0.0);
    for (index, (cell, expected_value)) in actual.iter().zip(data).enumerate() {
        let context = format!("{id} cell {index}");
        assert_eq!(
            cell.masked,
            mask.is_some_and(|items| items[index] == true),
            "{context}"
        );
        if !cell.masked {
            cell.value.assert_value(expected_value, tolerance, &context);
        }
    }
}

fn depth_input_dtype(input: &Value) -> &str {
    if let Some(dtype) = input["dtype"].as_str() {
        dtype
    } else if input.is_boolean() {
        "bool"
    } else if input["int"]
        .as_str()
        .is_some_and(|value| value.parse::<i64>().is_ok())
    {
        "int64"
    } else {
        // The native multiplication receives the floating-point representation
        // of Python scalars that do not fit int64.
        "float64"
    }
}

fn check_depth<T>(case: &Value)
where
    T: DepthValue + KernelValue + FromStr,
    T::Err: Debug,
{
    let args = &case["inputs"];
    let ros = input::<T>(&args["ros"]);
    let residence = input::<T>(&args["res_time"]);
    let actual = flame_depth_array(ros.view(), residence.view()).unwrap();
    assert_kernel(&actual, case);
}

#[test]
fn depth_preserves_native_widths_shape_and_masks() {
    for case in kernel_cases("get_flame_depth") {
        let args = &case["inputs"];
        let dtypes = (
            depth_input_dtype(&args["ros"]),
            depth_input_dtype(&args["res_time"]),
        );
        match dtypes {
            ("int64" | "bool", "int64") => check_depth::<i64>(&case),
            ("uint8", "uint8") => check_depth::<u8>(&case),
            ("uint64", "uint64") => check_depth::<u64>(&case),
            ("float32", "float32") => check_depth::<f32>(&case),
            ("float64", "float64" | "int64") | ("uint64", "int64") => check_depth::<f64>(&case),
            _ => panic!(
                "{} has unsupported fixture operands {dtypes:?}",
                case_id(&case)
            ),
        }
    }
}

fn residence_units(value: &Value) -> ResidenceUnits {
    match value.as_str().unwrap() {
        "sec" => ResidenceUnits::Seconds,
        "min" => ResidenceUnits::Minutes,
        units => panic!("unknown fixture residence units {units}"),
    }
}

#[test]
fn residence_matches_expected_units_and_domain_masks() {
    for case in kernel_cases("get_flame_residence_time") {
        let args = &case["inputs"];
        let ros = input(&args["ros"]);
        let fuel_consumption = input(&args["fuel_consumption"]);
        let midflame_ws = input(&args["midflame_ws"]);
        let actual = flame_residence_time_array(
            ros.view(),
            fuel_consumption.view(),
            midflame_ws.view(),
            residence_units(&args["units"]),
        )
        .unwrap();
        assert_kernel(&actual, &case);
    }
}

fn length_model(case: &Value) -> LengthModel {
    case["inputs"]["model"].as_str().unwrap().parse().unwrap()
}

#[test]
fn every_length_selector_has_its_exact_coefficients() {
    let mut selectors = BTreeSet::new();
    for case in cases().filter(|case| case["output"].get("tuple").is_some()) {
        assert!(selectors.insert(case["inputs"]["model"].as_str().unwrap().to_owned()));
        let (first, second, third) = length_model(&case).coefficients();
        let expected = case["output"]["tuple"].as_array().unwrap();
        let id = case_id(&case);
        assert_eq!(expected.len(), if third.is_some() { 3 } else { 2 }, "{id}");
        assert_float(first, number(&expected[0]), 0.0, id);
        assert_float(second, number(&expected[1]), 0.0, id);
        if let Some(third) = third {
            assert_float(third, number(&expected[2]), 0.0, id);
        }
    }
    assert_eq!(selectors.len(), 29);
}

#[test]
fn length_matches_every_model_and_masks_invalid_cells() {
    for case in kernel_cases("get_flame_length") {
        let args = &case["inputs"];
        let intensity = input(&args["fire_intensity"]);
        let depth = args.get("flame_depth").map(input);
        let actual = flame_length_array(
            length_model(&case),
            intensity.view(),
            depth.as_ref().map(ArrayD::view),
        )
        .unwrap();
        assert_kernel(&actual, &case);
    }
}

fn slope_units(value: &Value) -> SlopeUnits {
    match value.as_str().unwrap() {
        "degrees" => SlopeUnits::Degrees,
        "percent" => SlopeUnits::Percent,
        units => panic!("unknown fixture slope units {units}"),
    }
}

fn fire_type(value: &Value) -> ArrayD<Cell<f64>> {
    let code = match value.as_str() {
        Some("surface") => 1.0,
        Some("intermittent") => 2.0,
        Some("crown") => 3.0,
        Some(other) => panic!("unknown fixture fire type {other}"),
        None => return input(value),
    };
    scalar(code)
}

#[test]
fn height_selects_the_expected_nelson_and_finney_branches() {
    for case in kernel_cases("get_flame_height") {
        let args = &case["inputs"];
        let length = input(&args["flame_length"]);
        let actual = match args["model"].as_str().unwrap() {
            "Nelson" => {
                let fire_type = fire_type(&args["fire_type"]);
                let intensity = input(&args["fire_intensity"]);
                let wind = input(&args["midflame_ws"]);
                flame_height_array(
                    length.view(),
                    HeightModel::Nelson {
                        fire_type: fire_type.view(),
                        fire_intensity: intensity.view(),
                        midflame_ws: wind.view(),
                    },
                )
            }
            "Finney" => {
                let tilt = input(&args["flame_tilt"]);
                let slope = input(&args["slope_angle"]);
                flame_height_array(
                    length.view(),
                    HeightModel::Finney {
                        flame_tilt: tilt.view(),
                        slope_angle: slope.view(),
                        slope_units: slope_units(&args["slope_units"]),
                    },
                )
            }
            model => panic!("unknown fixture height model {model}"),
        }
        .unwrap();
        // The fixture pins the distinct output and mask on both sides of each
        // threshold, with an exact (zero tolerance) comparison for the Nelson cap.
        assert_kernel(&actual, &case);
    }
}

fn wind_speed_units(value: &Value) -> WindSpeedUnits {
    match value.as_str().unwrap() {
        "kph" => WindSpeedUnits::Kph,
        "mps" => WindSpeedUnits::Mps,
        "mph" => WindSpeedUnits::Mph,
        units => panic!("unknown fixture wind-speed units {units}"),
    }
}

fn butler_tilt(args: &Value) -> Result<ArrayD<Cell<f64>>, Error> {
    let wind = input(&args["wind_speed"]);
    let canopy_ht = input(&args["canopy_ht"]);
    flame_tilt_array(TiltModel::Butler {
        wind_speed: wind.view(),
        wind_speed_units: wind_speed_units(&args["wind_speed_units"]),
        canopy_ht: canopy_ht.view(),
    })
}

#[test]
fn tilt_distinguishes_equality_from_adjacent_floats() {
    for case in kernel_cases("get_flame_tilt") {
        let args = &case["inputs"];
        if args["flame_length"]["dtype"] == "float32" {
            let length = input::<f32>(&args["flame_length"]);
            let height = input::<f32>(&args["flame_height"]);
            let actual = standard_tilt_array(length.view(), height.view()).unwrap();
            assert_kernel(&actual, &case);
            continue;
        }
        let actual = match args["model"].as_str().unwrap() {
            "Standard" => {
                let length = input(&args["flame_length"]);
                let height = input(&args["flame_height"]);
                flame_tilt_array(TiltModel::Standard {
                    flame_length: length.view(),
                    flame_height: height.view(),
                })
            }
            "Finney" => {
                let length = input(&args["flame_length"]);
                let height = input(&args["flame_height"]);
                let slope = input(&args["slope_angle"]);
                flame_tilt_array(TiltModel::Finney {
                    flame_length: length.view(),
                    flame_height: height.view(),
                    slope_angle: slope.view(),
                    slope_units: slope_units(&args["slope_units"]),
                })
            }
            "Butler" => butler_tilt(args),
            model => panic!("unknown fixture tilt model {model}"),
        }
        .unwrap();
        assert_kernel(&actual, &case);
    }
}

fn wind_from_fixture(args: &Value) -> Result<WindResult<ArrayD<Cell<f64>>>, WindError> {
    let wind_speed = input(&args["wind_speed"]);
    let canopy_cover = input(&args["canopy_cover"]);
    let canopy_ht = input(&args["canopy_ht"]);
    let canopy_baseht = input(&args["canopy_baseht"]);
    let units = match args["units"].as_str().unwrap_or("SI") {
        "SI" => WindUnits::Si,
        "IMP" => WindUnits::Imperial,
        units => panic!("unknown fixture wind units {units}"),
    };
    mid_flame_ws_array(
        wind_speed.view(),
        canopy_cover.view(),
        canopy_ht.view(),
        canopy_baseht.view(),
        units,
    )
}

fn assert_wind_diagnostic(substituted_canopy_height: bool, case: &Value) {
    // The native diagnostic is a Boolean representing one warning per call.
    // The Python facade owns warning text and caller attribution.
    let diagnostics = case["diagnostics"].as_array().unwrap();
    assert_eq!(
        usize::from(substituted_canopy_height),
        diagnostics.len(),
        "{}",
        case_id(case)
    );
    if let Some(diagnostic) = diagnostics.first() {
        assert_eq!(diagnostic["category"], "UserWarning");
        assert!(diagnostic["message"]
            .as_str()
            .unwrap()
            .starts_with("One or more canopy_ht values are 0."));
    }
}

#[test]
fn wind_selects_shelter_branches_and_reports_canopy_substitution() {
    for case in kernel_cases("get_mid_flame_ws") {
        let actual = wind_from_fixture(&case["inputs"]).unwrap();
        assert_kernel(&actual.value, &case);
        assert_wind_diagnostic(actual.substituted_canopy_height, &case);
    }
}

fn assert_broadcast_error(error: Error, left: &Value, right: &Value) {
    let Error::IncompatibleShapes { shapes } = error else {
        panic!("expected incompatible shapes, got {error:?}");
    };
    assert!(
        shapes.contains(&shape(left)),
        "missing left shape in {shapes:?}"
    );
    assert!(
        shapes.contains(&shape(right)),
        "missing right shape in {shapes:?}"
    );
}

#[test]
fn native_errors_classify_invalid_inputs_and_preserve_diagnostics() {
    let mut matched = 0;
    for case in cases().filter(|case| case.get("error").is_some()) {
        let args = &case["inputs"];
        match case_id(&case) {
            "error_depth_broadcast" => {
                let ros = input::<f64>(&args["ros"]);
                let residence = input::<f64>(&args["res_time"]);
                let error = flame_depth_array(ros.view(), residence.view()).unwrap_err();
                assert_broadcast_error(error, &args["ros"], &args["res_time"]);
            }
            "error_length_invalid_model_first" => {
                let model = args["model"].as_str().unwrap();
                let error = model.parse::<LengthModel>().unwrap_err();
                assert!(matches!(error, Error::UnknownLengthModel(name) if name == model));
            }
            "error_length_params_only_missing_depth" => {
                let intensity = input(&args["fire_intensity"]);
                let error =
                    flame_length_array(length_model(&case), intensity.view(), None).unwrap_err();
                assert!(matches!(error, Error::MissingFlameDepth));
            }
            "error_tilt_butler_canopy" => {
                let error = butler_tilt(args).unwrap_err();
                assert!(matches!(error, Error::NonPositiveCanopyHeight(height) if height == 0.0));
            }
            "error_wind_after_diagnostic" => {
                let error = wind_from_fixture(args).unwrap_err();
                assert_wind_diagnostic(error.substituted_canopy_height, &case);
                assert_broadcast_error(error.error, &args["canopy_ht"], &args["canopy_baseht"]);
            }
            // Rust encodes required model fields and numeric arguments in the
            // public API's types. The Python tests cover these validation errors.
            "error_depth_list"
            | "error_height_missing_inputs_first"
            | "error_residence_validation_order" => continue,
            id => panic!("unclassified error fixture {id}"),
        }
        matched += 1;
    }
    assert_eq!(matched, 5);
}

fn scalar(value: f64) -> ArrayD<Cell<f64>> {
    ArrayD::from_elem(IxDyn(&[1]), Cell::present(value))
}

fn vector(values: &[Cell<f64>]) -> ArrayD<Cell<f64>> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()
}

#[test]
fn residence_minutes_feed_depth_in_metres() {
    let ros = vector(&[Cell::present(3.0), Cell::present(4.0)]);
    let fuel = vector(&[Cell::present(0.8), Cell::missing(0.8)]);
    let wind = scalar(1.4);
    let residence = flame_residence_time_array(
        ros.view(),
        fuel.view(),
        wind.view(),
        ResidenceUnits::Minutes,
    )
    .unwrap();
    let depth = flame_depth_array(ros.view(), residence.view()).unwrap();
    assert_eq!(depth.shape(), [2]);
    assert_float(
        depth[[0]].value,
        0.6130422939576858,
        2e-14,
        "depth in metres",
    );
    assert!(!depth[[0]].masked);
    assert!(depth[[1]].masked);
}

#[test]
fn si_midflame_wind_feeds_nelson_height_in_metres() {
    let cover = vector(&[Cell::present(35.0), Cell::missing(35.0)]);
    let wind = mid_flame_ws_array(
        scalar(18.0).view(),
        cover.view(),
        scalar(10.0).view(),
        scalar(0.0).view(),
        WindUnits::Si,
    )
    .unwrap();
    let height = flame_height_array(
        scalar(8.0).view(),
        HeightModel::Nelson {
            fire_type: scalar(1.0).view(),
            fire_intensity: scalar(720.0).view(),
            midflame_ws: wind.value.view(),
        },
    )
    .unwrap();
    assert_eq!(height.shape(), [2]);
    assert_float(
        height[[0]].value,
        0.505082298677979,
        2e-14,
        "Nelson flame height",
    );
    assert!(!height[[0]].masked);
    assert!(height[[1]].masked);
    assert!(!wind.substituted_canopy_height);
}

#[test]
fn byram_length_and_nelson_height_feed_standard_tilt() {
    let fire_intensity = scalar(425.0);
    let wind = vector(&[Cell::present(2.0), Cell::present(0.0), Cell::missing(2.0)]);
    let length =
        flame_length_array("Byram_HEAD".parse().unwrap(), fire_intensity.view(), None).unwrap();
    let height = flame_height_array(
        length.view(),
        HeightModel::Nelson {
            fire_type: scalar(1.0).view(),
            fire_intensity: fire_intensity.view(),
            midflame_ws: wind.view(),
        },
    )
    .unwrap();
    let tilt = flame_tilt_array(TiltModel::Standard {
        flame_length: length.view(),
        flame_height: height.view(),
    })
    .unwrap();
    assert_eq!(tilt.shape(), [3]);
    assert_float(
        tilt[[0]].value,
        61.923659902542404,
        2e-14,
        "tilt in degrees",
    );
    assert_float(tilt[[1]].value, 0.0, 0.0, "zero-wind tilt");
    assert!(!tilt[[0]].masked && !tilt[[1]].masked);
    assert!(tilt[[2]].masked);
}
