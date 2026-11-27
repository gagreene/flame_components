use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], input: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_flame-components"))
        .args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Some(input) = input {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}

fn result(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.ends_with(b"\n"));
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_error(output: Output, message: &str) {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(message), "{stderr}");
}

#[test]
fn scalar_flags_print_one_json_result() {
    let output = run(&["flame-depth", "--ros", "3", "--res-time", "2"], None);
    assert_eq!(output.stdout, b"6\n");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
}

#[test]
fn json_reads_a_file_and_stdin() {
    let path =
        std::env::temp_dir().join(format!("flame-components-cli-{}.json", std::process::id()));
    fs::write(&path, r#"{"ros":3,"res_time":2}"#).unwrap();
    let output = run(&["flame-depth", "--json", path.to_str().unwrap()], None);
    fs::remove_file(path).unwrap();
    assert_eq!(result(&output), json!(6));

    let output = run(
        &["flame-depth", "--json", "-"],
        Some(r#"{"ros":2,"res_time":7}"#),
    );
    assert_eq!(result(&output), json!(14));
}

#[test]
fn integer_tokens_preserve_precision_and_core_wrap_behavior() {
    let output = run(
        &[
            "flame-depth",
            "--ros",
            "9007199254740993",
            "--res-time",
            "1",
        ],
        None,
    );
    assert_eq!(result(&output).as_i64(), Some(9_007_199_254_740_993));

    let output = run(
        &["flame-depth", "--json", "-"],
        Some(r#"{"ros":9223372036854775807,"res_time":1}"#),
    );
    assert_eq!(result(&output).as_i64(), Some(i64::MAX));

    let output = run(
        &["flame-depth", "--json", "-"],
        Some(r#"{"ros":4611686018427387904,"res_time":2}"#),
    );
    assert_eq!(result(&output).as_i64(), Some(0));

    let output = run(
        &[
            "flame-depth",
            "--ros",
            "9007199254740993",
            "--res-time",
            "1.0",
        ],
        None,
    );
    assert_eq!(result(&output).as_f64(), Some(9_007_199_254_740_992.0));
}

#[test]
fn rejects_integer_tokens_outside_the_signed_64_bit_range() {
    let args = [
        "flame-depth",
        "--ros",
        "9223372036854775808",
        "--res-time",
        "1",
    ];
    assert_error(run(&args, None), "signed 64-bit");
    let args = ["flame-depth", "--json", "-"];
    assert_error(
        run(&args, Some(r#"{"ros":-9223372036854775809,"res_time":1}"#)),
        "signed 64-bit",
    );
    assert_error(
        run(
            &args,
            Some(r#"{"ros":1180591620717411303424,"res_time":1}"#),
        ),
        "signed 64-bit",
    );
}

#[test]
fn output_mask_is_distinct_from_infinity_and_a_clamped_nan() {
    let args = ["flame-depth", "--json", "-"];
    for (input, expected) in [
        (r#"{"ros":"Infinity","res_time":2}"#, json!("Infinity")),
        (r#"{"ros":"Infinity","res_time":0}"#, json!(0.0)),
        (r#"{"ros":"-Infinity","res_time":2}"#, json!(0.0)),
        (r#"{"ros":"NaN","res_time":2}"#, Value::Null),
        (r#"{"ros":null,"res_time":2}"#, Value::Null),
    ] {
        assert_eq!(result(&run(&args, Some(input))), expected);
    }

    let output = run(
        &[
            "flame-tilt",
            "--model",
            "Standard",
            "--flame-length",
            "2",
            "--flame-height",
            "3",
        ],
        None,
    );
    assert_eq!(result(&output), Value::Null);
}

#[test]
fn canopy_warning_goes_to_stderr_on_success() {
    let output = run(
        &[
            "mid-flame-ws",
            "--wind-speed",
            "18",
            "--canopy-cover",
            "35",
            "--canopy-ht",
            "0",
            "--canopy-baseht",
            "0",
        ],
        None,
    );
    let actual = result(&output).as_f64().unwrap();
    assert!((actual - 1.7409947974703335).abs() < 1e-13);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("0.5 m"), "{stderr}");
    assert_eq!(stderr.lines().count(), 1);
}

#[test]
fn coefficients_can_be_looked_up_without_measurements() {
    let output = run(
        &["flame-length", "--model", "Byram_HEAD", "--params-only"],
        None,
    );
    assert_eq!(result(&output), json!([0.0775, 0.46]));
    let output = run(
        &["flame-length", "--json", "-"],
        Some(r#"{"model":"Finney_HEAD","params_only":true}"#),
    );
    assert_eq!(result(&output), json!([0.01051, 0.774, 0.161]));
}

#[test]
fn scalar_commands_match_the_shared_reference_fixtures() {
    let reference = include_str!("../../../../tests/fixtures/parity/reference.jsonl");
    for line in reference.lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let id = case["id"].as_str().unwrap();
        // This Python fixture records a promotion outside the CLI's i64 input range.
        if id == "depth_object_sized_int" || case.get("error").is_some() {
            continue;
        }
        // Exercise scalar entry points with selected cells from these array fixtures.
        let cell = match id {
            "height_finney_slope_percent" | "wind_canopy_substitution_IMP" => Some(1),
            "tilt_finney_equal_length_degrees" | "tilt_standard_acos_boundary" => Some(0),
            _ => None,
        };
        let Some(inputs) = case["inputs"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                let value = match cell {
                    Some(index) if value["kind"] == "ndarray" => &value["data"][index],
                    _ => value,
                };
                Some((name.clone(), fixture_scalar(value)?))
            })
            .collect::<Option<serde_json::Map<String, Value>>>()
        else {
            continue;
        };
        let function = case["function"].as_str().unwrap();
        let command = function.strip_prefix("get_").unwrap().replace('_', "-");
        let output = run(
            &[&command, "--json", "-"],
            Some(&serde_json::to_string(&inputs).unwrap()),
        );
        let actual = result(&output);
        assert!(output.stderr.is_empty(), "{id}");
        let index = cell.unwrap_or(0);
        if let Some(tuple) = case["output"]["tuple"].as_array() {
            let actual = actual.as_array().unwrap();
            assert_eq!(actual.len(), tuple.len(), "{id}");
            for (actual, expected) in actual.iter().zip(tuple) {
                assert_eq!(
                    actual.as_f64(),
                    fixture_scalar(expected).unwrap().as_f64(),
                    "{id}"
                );
            }
        } else if case["clamped_mask"][index] == true {
            assert_eq!(actual, Value::Null, "{id}");
        } else {
            let expected = fixture_scalar(&case["output"]["data"][index]).unwrap();
            if expected.is_f64() {
                let tolerance = case["rtol"].as_f64().unwrap_or(0.0);
                let expected = expected.as_f64().unwrap();
                let actual = actual.as_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs(),
                    "{id}: {actual} != {expected}"
                );
            } else {
                assert_eq!(actual, expected, "{id}");
            }
        }
    }
}

fn fixture_scalar(value: &Value) -> Option<Value> {
    if !value.is_object() {
        return Some(value.clone());
    }
    if let Some(integer) = value["int"].as_str() {
        return Some(json!(integer.parse::<i64>().unwrap()));
    }
    let float = value["float"].as_str()?;
    Some(match float {
        "inf" => json!("Infinity"),
        "-inf" => json!("-Infinity"),
        "nan" => json!("NaN"),
        _ => json!(float.parse::<f64>().unwrap()),
    })
}

#[test]
fn invalid_input_goes_to_stderr_and_fails() {
    let args = ["flame-depth", "--json", "-"];
    for (input, message) in [
        ("[]", "object"),
        (r#"{"ros":3,"res_time":[2]}"#, "scalar"),
        (r#"{"ros":"3","res_time":2}"#, "number"),
        (r#"{"ros":1e400,"res_time":2}"#, "finite"),
        (r#"{"ros":3,"res_time":2,"typo":1}"#, "typo"),
        (r#"{"ros":3"#, "JSON"),
    ] {
        assert_error(run(&args, Some(input)), message);
    }
    assert_error(run(&["flame-depth", "--ros", "3"], None), "--res-time");
    assert_error(
        run(
            &["flame-depth", "--ros", "3", "--ros", "4", "--res-time", "2"],
            None,
        ),
        "more than once",
    );
    assert_error(
        run(&["flame-depth", "--ros", "3", "--json", "-"], None),
        "combined",
    );
    assert_error(
        run(
            &[
                "flame-tilt",
                "--model",
                "Butler",
                "--wind-speed",
                "2",
                "--wind-speed-units",
                "mps",
                "--canopy-ht",
                "0",
            ],
            None,
        ),
        "canopy height must be positive",
    );
}

#[test]
fn help_lists_all_calculations() {
    let output = run(&["--help"], None);
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let help = String::from_utf8(output.stdout).unwrap();
    for subcommand in [
        "mid-flame-ws",
        "flame-length",
        "flame-height",
        "flame-tilt",
        "flame-residence-time",
        "flame-depth",
    ] {
        assert!(help.contains(subcommand), "{help}");
    }
}
