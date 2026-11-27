"""Compare public calculation backends with results from the pinned reference."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import flame_components

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "parity"
CASES = [
    json.loads(line)
    for line in (FIXTURES / "reference.jsonl").read_text(encoding="utf-8").splitlines()
]
ERROR_TYPES = {"TypeError": TypeError, "ValueError": ValueError}

# NumPy's default integer changed from int32 to int64 on 64-bit Windows in 2.0.
DEFAULT_INT_CASES = {
    "depth_integers",
    "depth_bool_and_int",
    "depth_zero_dimensional_and_scalar",
    "depth_unsigned_wrap",
}

# The masked maximum used by the original clamp preserves these dtypes on NumPy 1.x.
LEGACY_DTYPES = {
    "depth_unsigned_wrap": "uint8",
    "depth_uint64_wrap": "uint64",
    "depth_reversed_nd_broadcast": "float32",
    "residence_float32_ros": "float32",
    "tilt_standard_float32": "float32",
}


def decode(value):
    if not isinstance(value, dict):
        return value
    if "int" in value:
        return int(value["int"])
    if "float" in value:
        return float(value["float"])
    if "tuple" in value:
        return tuple(map(decode, value["tuple"]))
    if "list" in value:
        return list(map(decode, value["list"]))

    data = np.array(list(map(decode, value["data"])), dtype=value["dtype"]).reshape(value["shape"])
    if value.get("layout") == "reversed_first_axis":
        data = np.ascontiguousarray(data[::-1])[::-1]
    if value["kind"] == "masked_array":
        return np.ma.array(data, mask=np.array(value["mask"]).reshape(value["shape"]))
    if value["kind"] == "numpy_scalar":
        return data[()]
    return data


def assert_output(actual, case):
    output = case["output"]
    expected = decode(output)
    if "tuple" in output:
        assert type(actual) is tuple
        assert all(type(value) is float for value in actual)
        assert actual == expected
        return

    dtype = np.dtype(output["dtype"])
    if np.__version__.startswith("1.") and case["id"] in LEGACY_DTYPES:
        dtype = np.dtype(LEGACY_DTYPES[case["id"]])
    elif case["id"] in DEFAULT_INT_CASES:
        dtype = np.dtype(int)

    if output["kind"] == "numpy_scalar":
        assert type(actual) is dtype.type
    else:
        assert type(actual) is np.ndarray
    assert actual.dtype == dtype
    assert actual.shape == tuple(output["shape"])

    actual = np.asarray(actual)
    np.testing.assert_array_equal(
        np.isnan(actual), np.array(case["clamped_mask"]).reshape(actual.shape)
    )
    if case.get("rtol"):
        np.testing.assert_allclose(actual, expected, rtol=case["rtol"], atol=0)
    else:
        np.testing.assert_array_equal(actual, expected.astype(dtype))
    if dtype.kind == "f":
        zero = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_public_backend_matches_pinned_reference(case, calculation_backend):
    function = getattr(flame_components, case["function"])
    inputs = {name: decode(value) for name, value in case["inputs"].items()}

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        if "error" in case:
            error = case["error"]
            with pytest.raises(ERROR_TYPES[error["category"]]) as raised:
                call_line = sys._getframe().f_lineno + 1
                function(**inputs, backend=calculation_backend)
            assert str(raised.value) == error["message"]
        else:
            call_line = sys._getframe().f_lineno + 1
            actual = function(**inputs, backend=calculation_backend)
            assert_output(actual, case)

    observed = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
            "at_caller": (
                Path(warning.filename).resolve() == Path(__file__).resolve()
                and warning.lineno == call_line
            ),
        }
        for warning in recorded
    ]
    assert observed == case["diagnostics"]
