#![cfg(feature = "extended")]

use flame_components::extended::{
    flame_depth_array, flame_residence_time_array_with_precision, format,
    standard_tilt_array_with_precision, LongDouble,
};
use flame_components::ndarray::arr0;
use flame_components::residence::ResidenceUnits;
use flame_components::Cell;

#[test]
fn depth_keeps_unmasked_nan_and_negative_zero() {
    for (rate, time, expected) in [(f64::INFINITY, 0.0, f64::NAN), (-0.0, 1.0, -0.0_f64)] {
        let rate = arr0(Cell::present(LongDouble::from_f64(rate)));
        let time = arr0(Cell::present(LongDouble::from_f64(time)));
        let depth = flame_depth_array(rate.view().into_dyn(), time.view().into_dyn()).unwrap();
        let cell = depth[[]].nonnegative();
        assert!(!cell.masked);
        if expected.is_nan() {
            assert!(cell.value.is_nan());
        } else {
            assert_eq!(cell.value.to_f64().to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn depth_preserves_the_last_significand_bit_and_tilt_tests_it() {
    let (width, mantissa, _, _) = format();
    let one = LongDouble::from_f64(1.0);
    let delta = LongDouble::from_f64(2.0f64.powi(1 - mantissa as i32));
    let next = arr0(Cell::present(one.add(delta)));
    let base = arr0(Cell::present(one));
    let depth = flame_depth_array(next.view().into_dyn(), base.view().into_dyn()).unwrap();
    assert_eq!(depth[[]], next[()]);
    assert!(depth[[]].value > one);

    // Arccos rejects a ratio one native ulp above one.
    let tilt = standard_tilt_array_with_precision(
        base.view().into_dyn(),
        next.view().into_dyn(),
        width as u8,
    )
    .unwrap();
    assert!(tilt[[]].masked);
    let single =
        standard_tilt_array_with_precision(base.view().into_dyn(), next.view().into_dyn(), 4)
            .unwrap();
    assert_eq!(single[[]], Cell::present(LongDouble::from_f64(0.0)));
}

#[test]
fn a_single_precision_stage_can_mask_a_wide_residence_result() {
    let width = format().0 as u8;
    let one = arr0(Cell::present(LongDouble::from_f64(1.0)));
    let wind = arr0(Cell::present(LongDouble::from_f64(1e30)));
    let residence = |widths| {
        flame_residence_time_array_with_precision(
            one.view().into_dyn(),
            one.view().into_dyn(),
            wind.view().into_dyn(),
            ResidenceUnits::Seconds,
            widths,
        )
        .unwrap()[[]]
    };
    assert!(!residence([width; 7]).masked);
    let mut mixed = [width; 7];
    mixed[2] = 4;
    assert!(residence(mixed).masked);
}

#[test]
fn residence_division_uses_the_numerators_cutoff_precision() {
    let width = format().0 as u8;
    let ros = arr0(Cell::present(LongDouble::from_f64(3e-307)));
    let one = arr0(Cell::present(LongDouble::from_f64(1.0)));
    let residence = |flag: u8| {
        let widths = [4, 4, 4, 4, width, width | (flag << 5), width];
        flame_residence_time_array_with_precision(
            ros.view().into_dyn(),
            one.view().into_dyn(),
            one.view().into_dyn(),
            ResidenceUnits::Seconds,
            widths,
        )
        .unwrap()[[]]
    };
    assert!(residence(0).masked);
    let single_cutoff = residence(1);
    assert!(!single_cutoff.masked);
    assert!((single_cutoff.value.to_f64() / 7.8e307 - 1.0).abs() < 1e-6);
    assert!(residence(2).masked);
    assert!(residence(3).masked);
}

#[test]
fn ordinary_division_keeps_infinity_and_propagates_operand_masks() {
    let width = format().0 as u8;
    let one = arr0(Cell::present(LongDouble::from_f64(1.0)));
    for quotient_width in [4, width] {
        let widths = [4, 4, 4, 4, width, quotient_width | (4 << 5), width];
        let divide = |denominator| {
            let denominator = arr0(denominator);
            flame_residence_time_array_with_precision(
                denominator.view().into_dyn(),
                one.view().into_dyn(),
                one.view().into_dyn(),
                ResidenceUnits::Seconds,
                widths,
            )
            .unwrap()[[]]
        };
        let zero = LongDouble::from_f64(0.0);
        let quotient = divide(Cell::present(zero));
        assert!(!quotient.masked);
        assert!(quotient.value.to_f64().is_infinite());
        assert!(
            divide(Cell {
                value: zero,
                masked: true
            })
            .masked
        );
    }
}
