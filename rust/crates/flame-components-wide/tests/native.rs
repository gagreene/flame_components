use flame_components_wide::{format, LongDouble, BYTE_CAPACITY};
use std::cmp::Ordering;

#[test]
fn arithmetic_resolves_one_native_ulp() {
    let one = LongDouble::from_f64(1.0);
    let (size, precision, min_exp, max_exp) = format();
    if size == 0 {
        assert_eq!((precision, min_exp, max_exp), (0, 0, 0));
        assert_eq!(one.add(one).to_f64(), 2.0);
        return;
    }
    assert!(matches!(
        (precision, min_exp, max_exp),
        (53, -1022, 1024) | (64 | 113, -16382, 16384)
    ));
    let ulp = LongDouble::from_f64(2.0_f64.powi(1 - precision as i32));
    let next = one.add(ulp);
    assert_eq!(next.partial_cmp(&one), Some(Ordering::Greater));
    assert_eq!(next.sub(one), ulp);
    assert_eq!(next, LongDouble::from_bytes(next.to_bytes()));
    assert_eq!(next, next.div(one).mul(one).pow(one));
    if precision > 53 {
        assert_eq!(next.to_f64(), 1.0);
        let square = LongDouble::from_f64(f64::MAX).pow(LongDouble::from_f64(2.0));
        assert!(square.is_finite());
        assert!(square.to_f64().is_infinite());
    }
}

#[test]
fn raw_bytes_ignore_padding_and_keep_negative_zero() {
    let (size, precision, _, _) = format();
    let meaningful = match (size, precision) {
        (0, _) => std::mem::size_of::<f64>(),
        (_, 64) => 10,
        _ => size,
    };
    let mut bytes = LongDouble::from_f64(-0.0).to_bytes();
    bytes[meaningful..].fill(0xab);
    let value = LongDouble::from_bytes(bytes);
    assert_eq!(value.to_f64().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(value.to_f32().to_bits(), (-0.0_f32).to_bits());
    assert!(value.to_bytes()[meaningful..].iter().all(|&b| b == 0));
    assert_eq!(value.abs().to_f64().to_bits(), 0);
}

#[test]
fn narrowing_to_f32_rounds_once() {
    if format().1 > 53 {
        // The exact value sits just above the midpoint between 1 and the next
        // binary32 value. Converting via binary64 loses the final increment.
        let value = LongDouble::from_f64(1.0 + 2.0_f64.powi(-24))
            .add(LongDouble::from_f64(2.0_f64.powi(-63)));
        assert_eq!(value.to_f32().to_bits(), 1.0_f32.to_bits() + 1);
    }
}

#[test]
fn x87_noncanonical_encodings_are_sanitized_before_arithmetic() {
    if format().1 == 64 {
        let mut bytes = [0u8; BYTE_CAPACITY];
        // Positive exponent with the explicit integer bit clear is an x87 unnormal.
        bytes[8] = 1;
        assert!(LongDouble::from_bytes(bytes).is_nan());
        // A pseudo-denormal has the explicit integer bit and exponent zero.
        bytes[8] = 0;
        bytes[7] = 0x80;
        let pseudo = LongDouble::from_bytes(bytes);
        bytes[8] = 1;
        assert_eq!(pseudo.to_bytes(), bytes);
        assert!(pseudo > LongDouble::from_f64(0.0));
    }
}

#[test]
fn transcendental_results_follow_native_math() {
    let n = LongDouble::from_f64;
    assert_eq!(n(4.0).sqrt(), n(2.0));
    assert_eq!(n(1.0).log(), n(0.0));
    assert_eq!(n(2.0).pow(n(3.0)), n(8.0));
    let angles = [n(1.0).atan(), n(0.5).asin(), n(0.5).acos()];
    for (angle, degrees) in angles.into_iter().zip([45.0, 30.0, 60.0]) {
        assert!((angle.degrees().to_f64() - degrees).abs() < 1e-13);
    }
    assert!((n(30.0).radians().sin().to_f64() - 0.5).abs() < 1e-15);
    let nan = n(-1.0).sqrt();
    assert!(nan.is_nan());
    assert!(!nan.is_finite());
    assert!(nan != nan);
    assert_eq!(nan.partial_cmp(&n(0.0)), None);
}
