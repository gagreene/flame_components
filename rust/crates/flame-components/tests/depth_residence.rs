use flame_components::depth::{flame_depth, flame_depth_array};
use flame_components::residence::{
    flame_residence_time, flame_residence_time_array, ResidenceUnits,
};
use flame_components::{Cell, Error};
use ndarray::{arr0, arr1, array};

#[test]
fn preserves_integer_precision_and_wrapping() {
    assert_eq!(
        flame_depth(Cell::present((1_i64 << 53) + 1), Cell::present(1)),
        Cell::present((1_i64 << 53) + 1)
    );
    assert_eq!(
        flame_depth(Cell::present(200_u8), Cell::present(2)),
        Cell::present(144)
    );
}

#[test]
fn depth_broadcasts_0d_and_noncontiguous_views_without_losing_masks() {
    let ros = array![
        [Cell::present(1), Cell::missing(50)],
        [Cell::present(3), Cell::present(4)]
    ];
    let time = arr0(Cell::present(2));
    let result = flame_depth_array(
        ros.slice(ndarray::s![..;-1, ..]).into_dyn(),
        time.view().into_dyn(),
    )
    .unwrap();
    assert_eq!(
        result,
        array![
            [Cell::present(6), Cell::present(8)],
            [Cell::present(2), Cell::missing(100)]
        ]
        .into_dyn()
    );
}

#[test]
fn depth_reports_incompatible_shapes() {
    let ros = arr1(&[Cell::present(1); 2]);
    let time = arr1(&[Cell::present(1); 3]);
    assert_eq!(
        flame_depth_array(ros.view().into_dyn(), time.view().into_dyn()),
        Err(Error::IncompatibleShapes {
            shapes: vec![vec![2], vec![3]]
        })
    );
}

#[test]
fn residence_masks_zero_tiny_and_infinite_spread_rates() {
    for ros in [0.0, 1e-310, f64::INFINITY] {
        assert!(
            flame_residence_time(
                Cell::input(ros),
                Cell::present(1.0),
                Cell::present(1.0),
                ResidenceUnits::Seconds,
            )
            .masked
        );
    }
}

#[test]
fn residence_to_depth_matches_a_hand_calculation() {
    let ros = Cell::present(2.0);
    let minutes = flame_residence_time(
        ros,
        Cell::present(1.0),
        Cell::present(1.0),
        ResidenceUnits::Minutes,
    );
    let result = flame_depth(ros, minutes);
    assert!(!result.masked);
    assert!((result.value - 0.39).abs() < 1e-15);
}

#[test]
fn residence_broadcasts_constants_across_a_masked_vector() {
    let ros = arr1(&[Cell::present(2.0), Cell::missing(3.0)]);
    let ones = arr0(Cell::present(1.0));
    let result = flame_residence_time_array(
        ros.view().into_dyn(),
        ones.view().into_dyn(),
        ones.view().into_dyn(),
        ResidenceUnits::Seconds,
    )
    .unwrap();
    assert!((result[0].value - 11.7).abs() < 1e-14);
    assert!(result[1].masked);
}
