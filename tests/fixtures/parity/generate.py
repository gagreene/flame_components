"""Capture the reviewed Python reference from a pinned Git tree."""

import argparse
import importlib
import json
import platform
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

BASELINE = "7f9e0ea2899882f0f805b25bb7e23b842270d520"
ROOT = Path(__file__).resolve().parents[3]
FLOAT_TOLERANCE = 2e-14
NELSON_BRANCHES = (("fire_type", [1, 2]), ("midflame_ws", 0), ("height", "flame_length"))

LENGTH_MODELS = (
    "Fons_NOWIND", "Thomas_NOWIND", "Yuana_NOWIND", "Barbon_NOWIND",
    "Nelson_BACK", "Fernandes_BACK", "Clark_BACK", "Vega_BACK",
    "Byram_HEAD", "Anderson1_HEAD", "Anderson2_HEAD", "Newman_HEAD",
    "Sneewujagt_HEAD", "Nelson1_HEAD", "Clark_HEAD", "Nelson2_HEAD",
    "VanWilgen_HEAD", "Burrows_HEAD", "MarsdenSmedley_HEAD", "Weise1_HEAD",
    "Catchpole_HEAD", "Fernandes1_HEAD", "Butler_HEAD", "Fernandes_HEAD",
    "Nelson3_HEAD", "Nelson4_HEAD", "Weise2_HEAD", "Davies_HEAD", "Finney_HEAD",
)


def encode(value):
    """Encode numbers as round-trippable strings and arrays in logical C order."""
    if isinstance(value, np.ndarray | np.generic):
        array = np.asarray(np.ma.getdata(value))
        mask = np.ma.getmaskarray(value)
        encoded = {
            "kind": "numpy_scalar" if isinstance(value, np.generic) else "ndarray",
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "data": [encode(item) for item in array.ravel().tolist()],
        }
        if isinstance(value, np.ma.MaskedArray):
            encoded["kind"] = "masked_array"
            encoded["mask"] = mask.ravel().tolist()
        if value.ndim and value.strides[0] < 0:
            encoded["layout"] = "reversed_first_axis"
        return encoded
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int | float):
        return {type(value).__name__: repr(value)}
    if isinstance(value, list | tuple):
        return {type(value).__name__: [encode(item) for item in value]}
    raise TypeError(f"Unsupported fixture value: {type(value).__name__}")


def case(name, function, inputs, *, rtol=0, branches=()):
    return name, function, inputs, rtol, branches


def cases():
    depth = "get_flame_depth"
    for name, ros, res_time in (
        ("integers", 3, 2),
        ("bool_and_int", True, 2),
        ("large_exact_int", 2**53 + 1, 1),
        ("object_sized_int", 2**70, 1),
        ("signed_overflow", 2**62, 2),
        ("signed_zero", -0.0, 2.0),
        ("positive_infinity", np.inf, 2.0),
        ("negative_infinity", -np.inf, 2.0),
        ("infinity_times_zero", np.inf, 0.0),
        ("input_nan", np.nan, 2.0),
        ("zero_dimensional_and_scalar", np.array(3), 2),
        ("two_zero_dimensional_arrays", np.array(3), np.array(2)),
        ("masked_zero_dimensional", np.ma.array(3, mask=True), np.array(2)),
        ("masked_integer", np.ma.array([3, 2], dtype="int64", mask=[True, False]), 2),
        ("unsigned_wrap", np.array([200], dtype="uint8"), np.array([2], dtype="uint8")),
        ("uint64_wrap", np.array([2**63 + 1], dtype="uint64"), np.array([2], dtype="uint64")),
        ("mixed_signed_unsigned", np.array([2**53 + 1], dtype="uint64"),
         np.array([1], dtype="int64")),
        ("reversed_nd_broadcast",
         np.array([[3.0, np.nan], [0.0, np.inf]], dtype="float32")[::-1],
         np.array([[2.0, 0.0]], dtype="float32")),
    ):
        yield case(f"depth_{name}", depth, {"ros": ros, "res_time": res_time})

    residence = "get_flame_residence_time"
    for name, ros, units in (
        ("seconds", 3.0, "sec"),
        ("minutes", 3.0, "min"),
        ("infinite_ros", np.inf, "sec"),
        ("zero_ros", 0.0, "sec"),
        ("tiny_ros", 1e-310, "sec"),
        ("mixed_ros", np.array([3.0, np.inf, np.nan, -3.0, 0.0, 1e-310]), "min"),
    ):
        yield case(f"residence_{name}", residence,
                   {"ros": ros, "fuel_consumption": 0.8, "midflame_ws": 1.4, "units": units},
                   rtol=FLOAT_TOLERANCE)
    yield case("residence_float32_ros", residence, {
        "ros": np.array([3.0, 4.0], dtype="float32"),
        "fuel_consumption": np.array([0.8], dtype="float32"),
        "midflame_ws": np.array([1.4], dtype="float32"), "units": "min",
    }, rtol=5e-7)

    length = "get_flame_length"
    for model in LENGTH_MODELS:
        inputs = {"model": model, "fire_intensity": 425.0}
        if model == "Finney_HEAD":
            inputs["flame_depth"] = 1.8
        yield case(f"length_coefficients_{model}", length, dict(inputs, params_only=True))
        yield case(f"length_value_{model}", length, inputs, rtol=FLOAT_TOLERANCE)
    yield case("length_negative_intensity", length, {
        "model": "Byram_HEAD", "fire_intensity": np.array([-1.0, 0.0, np.nan]),
    })
    yield case("length_finney_zero_depth", length, {
        "model": "Finney_HEAD", "fire_intensity": 425.0, "flame_depth": 0.0,
    })

    height = "get_flame_height"
    yield case("height_nelson_zero_wind", height, {
        "model": "Nelson", "flame_length": 2.0, "fire_type": "surface",
        "fire_intensity": np.nan, "midflame_ws": 0.0,
    }, branches=NELSON_BRANCHES)
    yield case("height_nelson_mixed_mask", height, {
        "model": "Nelson", "flame_length": 2.0,
        "fire_type": np.array([1, 2, 3, 1, 2, 3]),
        "fire_intensity": np.array([np.nan, 360, 100, 720, np.nan, 1000]),
        "midflame_ws": np.ma.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                  mask=[False, False, False, False, False, True]),
    }, branches=NELSON_BRANCHES, rtol=FLOAT_TOLERANCE)
    yield case("height_nelson_cap_boundary", height, {
        "model": "Nelson", "flame_length": np.array([np.nextafter(2.0, -np.inf), 2.0,
                                                     np.nextafter(2.0, np.inf)]),
        "fire_type": "surface", "fire_intensity": 720.0, "midflame_ws": 1.0,
    }, branches=NELSON_BRANCHES)
    for units in ("degrees", "percent"):
        yield case(f"height_finney_slope_{units}", height, {
            "model": "Finney", "flame_length": 8.0, "flame_tilt": 25.0, "slope_units": units,
            "slope_angle": np.array([np.nextafter(1.0, -np.inf), 1.0,
                                     np.nextafter(1.0, np.inf), np.nan, 30.0]),
        }, branches=(("slope_angle", 1),), rtol=FLOAT_TOLERANCE)

    tilt = "get_flame_tilt"
    adjacent_heights = np.array([np.nextafter(5.0, -np.inf), 5.0, np.nextafter(5.0, np.inf)])
    yield case("tilt_standard_acos_boundary", tilt, {
        "model": "Standard", "flame_length": 5.0, "flame_height": adjacent_heights,
    }, rtol=FLOAT_TOLERANCE)
    yield case("tilt_standard_float32", tilt, {
        "model": "Standard",
        "flame_length": np.array([2.0, 3.0, 2.0, 2.0], dtype=np.float32),
        "flame_height": np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }, rtol=5e-7)
    for units in ("degrees", "percent"):
        yield case(f"tilt_finney_equal_length_{units}", tilt, {
            "model": "Finney", "flame_length": 5.0,
            "flame_height": adjacent_heights, "slope_angle": -10.0, "slope_units": units,
        }, branches=(("flame_height", "flame_length"),), rtol=FLOAT_TOLERANCE)
    for units, wind_speed in (("kph", 18.0), ("mps", 5.0), ("mph", 11.1847)):
        yield case(f"tilt_butler_{units}", tilt, {
            "model": "Butler", "wind_speed": wind_speed, "wind_speed_units": units,
            "canopy_ht": 10.0,
        }, rtol=FLOAT_TOLERANCE)

    wind = "get_mid_flame_ws"
    for units in ("SI", "IMP"):
        yield case(f"wind_canopy_substitution_{units}", wind, {
            "wind_speed": 18.0, "canopy_cover": 35.0, "canopy_ht": np.array([0.0, 10.0, 0.0]),
            "canopy_baseht": 0.0, "units": units,
        }, branches=(("canopy_ht", 0), ("f", 5)), rtol=FLOAT_TOLERANCE)
    yield case("wind_shelter_boundary", wind, {
        "wind_speed": 20.0, "canopy_ht": 10.0, "canopy_baseht": 0.0, "units": "IMP",
        "canopy_cover": np.array([np.nextafter(1500.0, -np.inf), 1500.0,
                                  np.nextafter(1500.0, np.inf), np.nan]),
    }, branches=(("canopy_ht", 0), ("f", 5)), rtol=FLOAT_TOLERANCE)

    yield case("error_depth_list", depth, {"ros": [1, 2], "res_time": 3})
    yield case("error_depth_broadcast", depth, {"ros": np.ones(2), "res_time": np.ones(3)})
    yield case("error_length_invalid_model_first", length, {
        "model": "invalid", "fire_intensity": [1], "params_only": 1,
    })
    yield case("error_length_params_only_missing_depth", length, {
        "model": "Finney_HEAD", "fire_intensity": 100.0, "params_only": True,
    })
    yield case("error_height_missing_inputs_first", height, {
        "model": "Nelson", "flame_length": [2],
    })
    yield case("error_tilt_butler_canopy", tilt, {
        "model": "Butler", "wind_speed": 5.0, "wind_speed_units": "mps", "canopy_ht": 0.0,
    })
    yield case("error_residence_validation_order", residence, {
        "ros": [1], "fuel_consumption": 1.0, "midflame_ws": 1.0, "units": "bad",
    })
    yield case("error_wind_after_diagnostic", wind, {
        "wind_speed": 20.0, "canopy_ht": np.array([0.0, 1.0]), "canopy_baseht": np.ones(3),
        "canopy_cover": 50.0,
    })


def trace_branch(condition, operands, frame):
    left_name, right_name = operands
    left = frame[left_name]
    right = frame[right_name] if isinstance(right_name, str) else right_name
    trace = {
        "operands": list(operands),
        "left": encode(left),
        "right": encode(right),
        "condition": encode(condition),
    }
    left_data = np.ma.getdata(left)
    right_data = np.ma.getdata(right)
    if left_data.dtype == np.float64 and (np.isscalar(right) or right_data.dtype == np.float64):
        x, y = np.broadcast_arrays(left_data, right_data.astype("float64", copy=False))
        xmask, ymask = np.broadcast_arrays(np.ma.getmaskarray(left), np.ma.getmaskarray(right))
        # Signed distance in representable binary64 values for nonnegative finite operands.
        valid = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                 & ~xmask & ~ymask)
        trace["ulps_from_right"] = [
            str(int(a) - int(b)) if comparable else None
            for a, b, comparable in zip(x.view("int64").flat, y.view("int64").flat, valid.flat)
        ]
    return trace


def capture(package, specification):
    name, function, inputs, rtol, branches = specification
    row = {"id": name, "function": function, "inputs": {k: encode(v) for k, v in inputs.items()}}
    maximum = np.ma.maximum
    where = np.ma.where
    observed_branches = []

    def record_output(value, lower_bound):
        row["kernel_result"] = encode(value)
        clamped = maximum(value, lower_bound)
        row["clamped_mask"] = np.ma.getmaskarray(clamped).ravel().tolist()
        return clamped

    def record_where(condition, yes, no):
        operands = branches[len(observed_branches)]
        observed_branches.append(trace_branch(condition, operands, sys._getframe(1).f_locals))
        return where(condition, yes, no)

    with warnings.catch_warnings(record=True) as captured, \
            patch.object(np.ma, "maximum", record_output), \
            patch.object(np.ma, "where", record_where if branches else where):
        warnings.simplefilter("always")
        try:
            call_line = sys._getframe().f_lineno + 1
            result = getattr(package, function)(**inputs)
        except (TypeError, ValueError) as error:
            row["error"] = {"category": type(error).__name__, "message": str(error)}
        else:
            row["output"] = encode(result)
            if rtol:
                row["rtol"] = rtol
    assert len(observed_branches) == len(branches), name
    if branches:
        row["branches"] = observed_branches

    row["diagnostics"] = []
    arithmetic_warnings = []
    for warning in captured:
        record = {"category": warning.category.__name__, "message": str(warning.message)}
        if warning.category is RuntimeWarning:
            arithmetic_warnings.append(record)
        else:
            record["at_caller"] = (
                Path(warning.filename).resolve() == Path(__file__).resolve()
                and warning.lineno == call_line
            )
            row["diagnostics"].append(record)
    if arithmetic_warnings:
        row["baseline_arithmetic_warnings"] = arithmetic_warnings
    return row


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def generate(output_dir, numpy_version):
    if np.__version__ != numpy_version:
        raise ValueError(f"Expected NumPy {numpy_version}, found {np.__version__}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "reference.jsonl"
    metadata_path = output_dir / "provenance.json"
    for path in (data_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)

    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary)
        files = git("ls-tree", "-r", "--name-only", BASELINE, "--", "src/flame_components")
        for name in files.decode("utf-8").splitlines():
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(git("show", f"{BASELINE}:{name}"))

        source = checkout / "src"
        sys.path.insert(0, str(source))
        package = importlib.import_module("flame_components")
        assert Path(package.__file__).resolve().is_relative_to(source.resolve()), package.__file__
        rows = [capture(package, specification) for specification in cases()]
        sys.path.remove(str(source))

    metadata = {
        "schema_version": 1,
        "reference_commit": BASELINE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "default_integer_dtype": np.dtype(int).name,
        "case_count": len(rows),
    }
    with data_path.open("x", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with metadata_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(f"Captured {len(rows)} cases from {BASELINE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--numpy-version", default="2.5.0")
    args = parser.parse_args()
    generate(args.output_dir, args.numpy_version)
