use flame_components::{
    height::{finney_height_array_with_precision, nelson_height_array_with_precision},
    length::{flame_length_array_with_precision, LengthModel},
    ndarray::{arr1, Array1},
    residence::{flame_residence_time_array_with_precision, ResidenceUnits},
    tilt::{
        butler_tilt_array_with_precision, finney_tilt_array_with_precision,
        standard_tilt_array_with_precision,
    },
    wind::{mid_flame_ws_array_with_precision, WindUnits},
    Cell, SlopeUnits, WindSpeedUnits,
};

fn input(value: f64) -> Array1<Cell<f64>> {
    arr1(&[Cell::present(value)])
}

#[test]
fn nelson_multiplies_a_scalar_coefficient_at_the_selected_precision() {
    let (length, intensity, wind) = (input(100.0), input(10000.0), input(1.0));
    let fire_type = flame_components::ndarray::arr0(Cell::present(1.0));
    let result = nelson_height_array_with_precision(
        length.view().into_dyn(),
        fire_type.view().into_dyn(),
        intensity.view().into_dyn(),
        wind.view().into_dyn(),
        [2; 3],
    )
    .unwrap();
    assert_eq!(result[0], Cell::present(27.765625));
}

#[test]
fn wind_selects_the_sheltered_branch_after_single_precision_si_conversion() {
    let (wind, cover, height, base) = (input(20.0), input(1600.0), input(8.0), input(0.5));
    let result = mid_flame_ws_array_with_precision(
        wind.view().into_dyn(),
        cover.view().into_dyn(),
        height.view().into_dyn(),
        base.view().into_dyn(),
        WindUnits::Si,
        [4; 18],
    )
    .unwrap();
    assert!(!result.value[0].masked);
    assert!((result.value[0].value - 0.10858893394470215).abs() < 2e-8);
}

#[test]
fn butler_masks_half_precision_canopy_wind_cube_overflow() {
    let (wind, height) = (input(1e4), input(10.0));
    let result = butler_tilt_array_with_precision(
        wind.view().into_dyn(),
        WindSpeedUnits::Mps,
        height.view().into_dyn(),
        [2; 10],
    )
    .unwrap();
    assert!(result[0].masked);
}

#[test]
fn finney_tilt_uses_half_precision_slope_before_promoting_the_angle() {
    let (length, height, slope) = (input(8.0), input(5.0), input(90.0));
    let result = finney_tilt_array_with_precision(
        length.view().into_dyn(),
        height.view().into_dyn(),
        slope.view().into_dyn(),
        SlopeUnits::Degrees,
        [2, 8, 8, 8, 8, 8, 8],
    )
    .unwrap();
    assert!((result[0].value - 0.01039546292300021).abs() < 1e-13);
}

#[test]
fn standard_tilt_rounds_acos_before_half_precision_degrees() {
    let (length, height) = (input(2.0), input(1.0));
    let result =
        standard_tilt_array_with_precision(length.view().into_dyn(), height.view().into_dyn(), 2)
            .unwrap();
    assert_eq!(result[0], Cell::present(59.96875));
}

#[test]
fn a_single_precision_overflow_in_length_power_masks_the_cell() {
    let intensity = input(1e30);
    let result = flame_length_array_with_precision(
        LengthModel::ClarkBack,
        intensity.view().into_dyn(),
        None,
        [4; 4],
    )
    .unwrap();
    assert!(result[0].masked);
}

#[test]
fn a_single_precision_overflow_in_wind_power_masks_residence_time() {
    let (ros, fuel, wind) = (input(1.0), input(1.0), input(1e30));
    let result = flame_residence_time_array_with_precision(
        ros.view().into_dyn(),
        fuel.view().into_dyn(),
        wind.view().into_dyn(),
        ResidenceUnits::Seconds,
        [4; 7],
    )
    .unwrap();
    assert!(result[0].masked);
}

#[test]
fn finney_height_preserves_the_single_precision_angle_before_promotion() {
    let (length, tilt, slope) = (input(8.0), input(25.0), input(1.0));
    let actual = finney_height_array_with_precision(
        length.view().into_dyn(),
        tilt.view().into_dyn(),
        slope.view().into_dyn(),
        SlopeUnits::Percent,
        [4, 8, 8, 8, 8, 8, 8, 8],
    )
    .unwrap();
    assert!(!actual[0].masked);
    assert!((actual[0].value - 7.250462288828255).abs() < 1e-14);
}
