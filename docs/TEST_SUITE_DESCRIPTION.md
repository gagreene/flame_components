# Test Suite Description

Python tests cover both calculation backends, public validation, and
multiprocessing. Build the native extension with `uv sync`, then run each
backend explicitly.

```text
uv run pytest tests/ -v --calculation-backend=python
uv run pytest tests/ -v --calculation-backend=rust
```

The standalone library and CLI suites run with Cargo.

```text
cargo test --locked --manifest-path rust/Cargo.toml -p flame-components -p flame-components-cli
cargo test --locked --manifest-path rust/Cargo.toml -p flame-components --features extended
```

## Overview

`flame_components` computes fire-behavior metrics (mid-flame wind speed, flame length,
height, tilt, residence time, depth) from scalar or `numpy.ndarray` inputs. Every public
function shares two conventions the test suite verifies throughout:

1. **Per-cell masks.** Input NaNs join the existing mask. The selected branch
   determines which masks propagate. Result formatting fills masked positions
   with NaN and retains computed NaNs and signed zero.
2. **Two failure modes for bad input, deliberately different:**
   - **Type/argument errors** (wrong type, missing required argument, unknown enum
     value like an invalid `model` or `units` string) → raise `TypeError` or
     `ValueError` immediately, with a clear message.
   - **Numerically degenerate input** (division by zero, a value outside a formula's
     mathematical domain such as `arccos` of a ratio > 1) → return `NaN`, consistent
     with the NaN-propagation contract. This is intentional, not a gap — see
     "Documented NaN-for-undefined behaviors" below — and is distinct from case 1
     precisely because the input *type* and *shape* were valid; only the numeric
     combination was undefined.
   - Every function also floors negative results at exactly `0` where negative
     values are physically meaningless (e.g. depth, wind speed, residence time),
     rather than propagating a negative number.

This distinction is why the test suite pairs almost every function with both an
"Errors" class (asserting a clean exception) and explicit NaN/clamping tests
(asserting the degenerate-but-valid-type case) — collapsing the two into "should not
crash" would hide real, meaningfully different failure modes.

### Documented NaN-for-undefined behaviors

These are asserted explicitly in the suite so a future change can't silently alter
them without a test failing — they are the *other* half of the two-failure-modes
convention above:

- `get_flame_residence_time(ros=0, ...)` → `NaN` (division by zero; no fire spread
  means residence time is undefined, not simply "missing").
- `get_flame_length('Finney_HEAD', ..., flame_depth=0)` → `NaN` (division by
  `flame_depth**0.161`; zero depth is undefined for this model).
- `get_flame_length(..., fire_intensity=<negative>)` → `NaN` (a fractional power of a
  negative number is undefined in the reals).
- `get_flame_tilt('Standard', flame_length=L, flame_height=H)` with `H > L` → `NaN`
  (`arccos(H/L)` requires `H/L` in `[0, 1]`; a flame can't be taller than its own
  length).

---

## Backend, parity, and packaging coverage

| Tests | What they cover |
|---|---|
| `tests/unit/test_backend.py` and `test_warning_contract.py` | Explicit and automatic backend selection, import failures, NumPy error policies, multiprocessing dispatch, and warning location and order. |
| `tests/unit/test_input_validation.py` | Dtypes, scalar validation, legacy model names, signed zeros, computed NaNs, and partial blocks. |
| `tests/integration/test_reference_fixtures.py` | Both Python-reachable engines against the frozen reference in `tests/fixtures/parity/`. Provenance records the reference commit and runtime. |
| `tests/integration/test_native_extension.py`, `test_native_extended_buffers.py`, and `test_numpy_scalar_parity.py` | Native buffers, broadcasting, scalar division, precision, model behavior, and maximum-rank shapes. Extended cases run on platforms with a wider NumPy format. |
| `rust/crates/flame-components/tests/` | Standalone Rust model and fixture checks, masked arithmetic, and extended storage. |
| `rust/crates/flame-components-cli/tests/` | CLI flags, JSON, diagnostics, and invalid inputs. |

CI builds and installs the native wheels and source distribution. It checks
Python 3.11?3.14 on Linux, Windows, and macOS, including NumPy 1.23.2 on Linux.
It also verifies the single Cargo archive, runs the packaged CLI, and exercises
both automatic backend-selection paths.

## Unit tests (`tests/unit/`)

One file per public function, one for the multiprocessing dispatcher, and one for the
public `__all__` surface. Per-function files have up to three classes:
`*Scalar`/model-named classes (happy-path + edge-case behavior for scalar input),
`*Array` (vectorized behavior, shape, NaN propagation), and `*Errors` (invalid input
raises the right exception type).

### `test_public_api.py` (1 test)

| Test | What & why |
|---|---|
| `test_all_contains_only_snake_case_names` | `flame_components.__all__` contains exactly the 7 snake_case public function names — `from flame_components import *` must not surface implementation details like private helpers. |

### `test_get_mid_flame_ws.py` (14 tests)

Tests Albini & Baughman (1979) mid-flame wind speed under a canopy.

| Test | What & why |
|---|---|
| `test_si_returns_positive_float` | Valid SI-unit inputs produce a positive result — basic happy path. |
| `test_imp_returns_positive_float` | Same, for IMP units — confirms the IMP branch also runs end-to-end. |
| `test_zero_wind_returns_zero` | Zero input wind speed must produce zero output (formula is linear in wind speed; catches an accidental additive constant). |
| `test_returns_float_not_array` | Scalar in → scalar out, not a length-1 `ndarray` — callers doing arithmetic on the result would break if this silently changed. |
| `test_canopy_ht_zero_warns` | `canopy_ht=0` must emit `UserWarning` mentioning `canopy_ht` (documented zero-division guard). |
| `test_canopy_ht_zero_returns_non_negative` | After the warned substitution (0 → 0.5 m equivalent), the result must still be usable (non-negative), not `NaN` or an exception. |
| `test_si_and_imp_agree_for_equivalent_inputs` | A wind speed and canopy geometry expressed in SI vs. IMP units, converted correctly for the documented 10-m/20-ft reference-height difference, must produce the same result. Catches silent unit-conversion drift between the two code paths. |
| `test_negative_wind_speed_clamped_to_zero` | Negative wind speed (non-physical) is floored at exactly `0`, not left negative. |
| `test_array_output_shape` | Array input produces same-shape array output. |
| `test_array_all_non_negative` | Vectorized results respect the same non-negativity floor as the scalar path. |
| `test_nan_propagates` | NaN in one array element masks only that element's output; sibling elements are unaffected. |
| `test_invalid_units_raises` | `units='METRIC'` (not `'SI'`/`'IMP'`) raises `ValueError`. |
| `test_non_string_units_raises` | Non-string `units` (e.g. `42`) raises `TypeError` before the `ValueError` string check even runs. |
| `test_string_wind_speed_raises` | Wrong type for `wind_speed` raises `TypeError`. |

### `test_get_flame_length.py` (16 tests)

Tests the 28 published fire-intensity-to-flame-length correlations (Finney & Grumstrup
2023), keyed by `model`.

| Test | What & why |
|---|---|
| `test_byram_head_known_value` | `Byram_HEAD` model against a hand-computed value (`0.0775 * 100**0.46`) — anchors the whole `model_dict` lookup/power-law mechanism to a known-correct number. |
| `test_zero_intensity_returns_zero` | Zero fire intensity → zero flame length (`0**0.46 = 0`), not `NaN` or an error — zero intensity is a valid, defined input (no fire), unlike zero `flame_depth` below. |
| `test_returns_float_not_array` | Scalar in → scalar out. |
| `test_non_negative` | Basic non-negativity floor. |
| `test_params_only_returns_tuple` | `params_only=True` returns the raw `(a, b)` model-parameter tuple instead of computing a length — an alternate code path worth covering directly since it returns early, before any masked-array math runs. |
| `test_finney_head_without_flame_depth_raises` | `Finney_HEAD` is the one model requiring `flame_depth`; omitting it must raise `ValueError`, not silently divide by `None`. |
| `test_finney_head_known_value` | `Finney_HEAD` (`fl = 0.01051 * I^0.774 / D^0.161`) against a hand-computed value — anchors the coefficient/exponent tuple for this special three-parameter model, the only one requiring `flame_depth`. |
| `test_finney_head_zero_flame_depth_is_undefined_returns_nan` | **Documented NaN case:** dividing by `flame_depth**0.161` at `flame_depth=0` is undefined; asserts `NaN`, not a crash or a silently wrong number. |
| `test_negative_fire_intensity_is_undefined_returns_nan` | **Documented NaN case:** a fractional power of a negative number is undefined in the reals. |
| `test_params_only_non_bool_raises_type_error` | `params_only='yes'` (a truthy non-bool) must raise `TypeError`, not be silently treated as `True`. |
| `test_array_output_shape` | Array input produces same-shape output. |
| `test_monotone_increasing_with_intensity` | Higher intensity → strictly longer flame — a coarse sanity check on the power-law's sign/exponent that a unit test on a single value can't catch (e.g. a flipped exponent could still hit one lucky matching value). |
| `test_nan_propagates` | NaN masking, standard pattern. |
| `test_invalid_model_name_raises` | Unknown `model` string raises `ValueError` listing valid options. |
| `test_non_string_model_raises` | Non-string `model` raises `TypeError`. |
| `test_string_intensity_raises` | Wrong type for `fire_intensity` raises `TypeError`. |

### `test_get_flame_height.py` (21 tests)

Tests two independent models: Nelson & Adkins (1986) (uses fire intensity + wind) and
Finney & Martin (1992) (uses tilt + slope).

| Test | What & why |
|---|---|
| `test_surface_fire_known_value` | Nelson model, `fire_type=1`, against a hand-computed value — anchors the `a` coefficient and the core `a*I/ws` formula. |
| `test_zero_wind_returns_flame_length` | `midflame_ws=0` → height equals `flame_length` (a windless flame stands straight up) — this is a distinct branch (`ma.where(midflame_ws==0, flame_length, ...)`), not just "another number," so it needs its own test. |
| `test_height_capped_at_flame_length` | A computed height exceeding `flame_length` is capped there — physically, flame height can't exceed flame length. |
| `test_string_surface_equals_integer_1` | String (`'surface'`) and integer (`1`) `fire_type` inputs must produce identical results — the `fire_type_dict` translation layer must be a true no-op equivalence, not a numerically-close-but-different path. |
| `test_active_crown_uses_larger_a_coefficient` | Crown fires (`fire_type=3`) use a materially larger coefficient (`0.0175` vs `1/360`) — asserts the *direction* of the difference, catching a swapped-coefficient bug that a single-value test might not. |
| `test_passive_crown_same_coefficient_as_surface` | `fire_type=2` ("passive crown") shares the surface-fire coefficient — a deliberate, easy-to-invert domain rule worth pinning down explicitly. |
| `test_returns_float_not_array` | Scalar in → scalar out. |
| `test_non_negative` | Basic non-negativity floor. |
| `test_invalid_integer_fire_type_raises` | `fire_type=99` (outside the valid `{1,2,3}` domain) raises `ValueError` rather than silently computing a meaningless crown-coefficient result. |
| `test_zero_fire_type_raises` | `fire_type=0` is likewise outside the valid `{1,2,3}` domain and must raise. |
| `test_zero_tilt_vertical_flame_equals_flame_length` | Finney model, `flame_tilt=0` on flat ground → height equals `flame_length` (vertical flame) — anchors the trigonometric identity at its simplest case. |
| `test_45_degree_tilt_flat_ground` | 45° tilt against the exact `sin(π/4)` value — a second, non-trivial anchor point for the trig formula. |
| `test_percent_slope_units_accepted` | `slope_units='percent'` is accepted and produces a sane (non-negative) result — exercises the alternate slope-unit conversion branch. |
| `test_non_negative_on_steep_slope` | Non-negativity floor holds even on a steep slope, where the more complex (non-flat-ground) formula branch is active. |
| `test_array_output_shape` | Array input produces same-shape output. |
| `test_array_all_non_negative` | Vectorized non-negativity floor. |
| `test_nan_propagates` | NaN masking, standard pattern. |
| `test_invalid_model_raises` | Unknown `model` raises `ValueError`. |
| `test_nelson_without_fire_type_raises` | Nelson model's three required arguments are enforced together — omitting any one raises `ValueError`. |
| `test_finney_without_slope_raises` | Finney model's required arguments are enforced together. |
| `test_non_string_model_raises` | Non-string `model` raises `TypeError`. |

### `test_get_flame_tilt.py` (21 tests)

Tests three independent models: Standard geometry (flat ground), Finney & Martin
(1992) (sloped ground), and Butler et al. (2004) (crown fires, wind-driven).

| Test | What & why |
|---|---|
| `test_vertical_flame_zero_tilt` | Standard model, `height == length` → `arccos(1) = 0°` — anchors the vertical-flame boundary. |
| `test_horizontal_flame_90_degrees` | `height = 0` → `arccos(0) = 90°` — anchors the horizontal-flame boundary, the opposite extreme from the previous test. |
| `test_arccos_half_equals_60_degrees` | A non-boundary value (`height/length = 0.5`) against the exact `arccos(0.5) = 60°` — confirms the formula isn't just correct at the two easy extremes. |
| `test_returns_float_not_array` | Scalar in → scalar out. |
| `test_non_negative` | Basic non-negativity floor. |
| `test_height_exceeds_length_is_undefined_returns_nan` | **Documented NaN case:** `height > length` pushes `arccos`'s argument outside `[-1, 1]`; a flame can't be taller than its own length, so this is genuinely undefined, not just unusual. |
| `test_flat_ground_returns_non_negative` | Finney model on flat ground (`slope_angle=0`) is a distinct formula branch (`slope_angle <= 1`) from the sloped case; verifies it at least produces a sane result. |
| `test_percent_slope_matches_equivalent_degree_slope` | A 20% slope (`slope_units='percent'`) and its equivalent `arctan(0.20)` degree value (`slope_units='degrees'`) must produce the same tilt — confirms the percent-to-radians conversion is numerically correct, not just branch-reachable. |
| `test_butler_returns_positive_tilt` | Butler model (crown fire + wind) produces a positive tilt — basic happy path for the third, independent model. |
| `test_higher_wind_produces_more_tilt` | Stronger wind → greater tilt (monotonicity) — a directional sanity check the exact-value tests below can't provide on their own. |
| `test_mph_units_matches_equivalent_kph` | `36 km/h` and its exact `22.3694 mph` equivalent must produce the same tilt — confirms the `mph`→m/s conversion factor is correct, not just that the branch runs. |
| `test_mps_units_matches_equivalent_kph` | `36 km/h` and its exact `10 m/s` equivalent must produce the same tilt — confirms the "no conversion needed" `mps` branch is truly a no-op, not just reachable. |
| `test_zero_canopy_ht_raises` | `canopy_ht=0` (Butler model divides by it) raises a clean `ValueError` rather than leaking a raw `ZeroDivisionError`. |
| `test_negative_canopy_ht_raises` | Negative `canopy_ht` is equally non-physical and must raise the same `ValueError`. |
| `test_string_canopy_ht_raises_type_error` | Wrong type for `canopy_ht` raises `TypeError` before reaching any arithmetic. |
| `test_array_output_shape` | Array input produces same-shape output. |
| `test_nan_propagates` | NaN masking via `flame_length`, standard pattern. |
| `test_nan_propagates_via_canopy_ht_butler_model` | NaN in a `canopy_ht` array (Butler model) masks only the affected element — verifies the masking contract holds for `canopy_ht` specifically, not just the scalar zero/negative domain checks above. |
| `test_invalid_model_raises` | Unknown `model` raises `ValueError`. |
| `test_standard_missing_flame_height_raises` | Standard model's required arguments enforced. |
| `test_butler_missing_canopy_ht_raises` | Butler model's required arguments enforced. |

### `test_get_flame_residence_time.py` (11 tests)

Tests Nelson & Adkins (1988) flame residence time.

| Test | What & why |
|---|---|
| `test_known_value_seconds` | Hand-computed value (`0.39 * 1**0.25 * 1**1.51 / (1/60) = 23.4`) anchors the formula in its default output unit. |
| `test_known_value_minutes` | Same inputs with `units='min'` — confirms the `/60` output-unit conversion is applied correctly and consistently with the seconds case. |
| `test_returns_float_not_array` | Scalar in → scalar out. |
| `test_negative_ros_clamped_to_zero` | Negative `ros` flips the formula's sign; result is floored at exactly `0`. |
| `test_zero_ros_is_undefined_returns_nan` | **Documented NaN case:** `ros=0` (no fire spread) divides by zero in the `ros/60` denominator; residence time is genuinely undefined, not just zero or missing. |
| `test_array_output_shape` | Array input produces same-shape output. |
| `test_array_all_non_negative` | Vectorized non-negativity floor. |
| `test_nan_propagates` | NaN masking, standard pattern. |
| `test_string_ros_raises` | Wrong type for `ros` raises `TypeError`. |
| `test_invalid_units_raises` | `units='seconds'` (a plausible typo for `'sec'`) raises `ValueError` rather than silently behaving like `'sec'`. |
| `test_non_string_units_raises` | Non-string `units` raises `TypeError`. |

### `test_get_flame_depth.py` (11 tests)

Tests Fons et al. (1963) flame depth: `fd = ros * res_time` — the simplest function in
the library (a single multiplication), so its tests focus on the clamping and
NaN-propagation contract rather than formula correctness.

| Test | What & why |
|---|---|
| `test_known_value` | `2.0 * 3.0 = 6.0` exactly — trivial but confirms no unexpected transformation (e.g. an accidental unit conversion) is applied. |
| `test_zero_ros` | Zero rate of spread → zero depth (valid, defined input; contrast with `get_flame_residence_time`'s `ros=0`, which is a *denominator* there and produces `NaN` instead). |
| `test_zero_res_time` | Zero residence time → zero depth, by symmetry with the previous test. |
| `test_returns_float_not_array` | Scalar in → scalar out. |
| `test_negative_ros_clamped_to_zero` | Negative `ros` makes the product negative; asserts the *exact* clamped value (`0.0`), not just non-negativity. |
| `test_negative_res_time_clamped_to_zero` | Symmetric case for negative `res_time` — both multiplicands' negative-clamping paths are covered independently. |
| `test_array_output_shape` | Array input produces same-shape output. |
| `test_array_known_values` | Element-wise multiplication against hand-computed values `[2, 6, 12]`. |
| `test_nan_propagates` | NaN masking, standard pattern. |
| `test_string_ros_raises_type_error` | Wrong type for `ros` raises `TypeError`. |
| `test_string_res_time_raises_type_error` | Wrong type for `res_time` raises `TypeError`. |

### `test_flame_component_array_multiprocessing.py` (7 tests)

Tests the block-splitting multiprocessing dispatcher.

| Test | What & why |
|---|---|
| `test_matches_direct_call_when_concatenated` | The dispatcher returns a **list of per-block arrays**, not a single concatenated array. Concatenating the blocks must reproduce exactly what calling `get_flame_depth` directly on the full array would give — the core correctness guarantee of the whole dispatcher. |
| `test_explicit_block_size_produces_expected_block_count` | An explicit `block_size` produces the expected number of correctly-sized blocks — verifies the block-splitting logic itself, independent of the underlying calculation. |
| `test_num_processors_less_than_two_warns_and_still_runs` | `num_processors=1` emits a `UserWarning` and completes successfully with the documented `num_processors=2` fallback, rather than raising. |
| `test_num_processors_zero_warns_and_still_runs` | Same fallback path exercised at the `num_processors=0` boundary. |
| `test_unknown_flame_function_raises` | An unrecognized `flame_function` key raises `ValueError` listing the valid options. |
| `test_no_array_inputs_raises` | All-scalar `kwargs` gives the dispatcher nothing to split into blocks; raises `ValueError` rather than silently doing no work. |
| `test_mismatched_array_shapes_raises` | Two input arrays of different shapes can't be split into matching blocks; raises `ValueError`. |

---

## Integration tests (`tests/integration/test_flame_pipeline.py`) (7 tests)

Unlike the unit tests (one function, isolated), these chain multiple functions the way
a real fire-behavior workflow would, verifying that one function's output is valid
input to the next — a class of bug unit tests can't catch (e.g. a unit mismatch between
two functions that are each individually correct).

| Test | What & why |
|---|---|
| `test_scalar_pipeline_produces_positive_depth` | `get_flame_residence_time` → `get_flame_depth`: residence time output feeds into flame depth without type errors and produces a physically sane (positive) result. |
| `test_residence_depth_pipeline_known_value` | Same pipeline against a hand-computed exact value (`ros=fuel_consumption=midflame_ws=1.0` → `res_time=flame_depth=0.39`) — pins a fixed numeric result through both functions, catching unit-conversion drift that a positivity-only check can't. |
| `test_array_pipeline_preserves_shape` | Same pipeline, vectorized: shape is preserved end-to-end and all values stay non-negative through both functions' clamping. |
| `test_scalar_height_bounded_by_flame_length` | `get_mid_flame_ws` → `get_flame_length` → `get_flame_height`: chains three functions and asserts the Nelson model's height-capped-at-length invariant still holds when the inputs come from upstream calculations rather than hand-picked values. |
| `test_array_pipeline_shape_and_bounds` | Same three-function pipeline, vectorized. |
| `test_roundtrip_recovers_original_height` | `get_flame_tilt` (Standard) → `get_flame_height` (Finney, flat ground) is a mathematical identity on flat ground (`height → tilt = arccos(h/L) → height = L·cos(tilt) = h`); recovering the original height confirms the two independently-implemented models (Standard tilt geometry and Finney height geometry) are consistent with each other, not just each internally correct. |
| `test_nan_in_ros_propagates_to_depth` | NaN in one element of a pipeline's array input must contaminate only that element all the way through two chained functions, without affecting sibling elements — the multi-function analogue of each unit test's single-function `test_nan_propagates`. |

---

## Regression tests (`tests/regression/test_gotchas.py`) (5 tests)

Pinned reproductions of specific historical bugs, kept permanently so they can never
silently reappear. Unlike the edge-case tests above (which test *behavior contracts*),
these test *specific past incidents*.

| Test | What & why |
|---|---|
| `test_string_fire_type_surface_does_not_raise` | Historical Bug: `fire_type`'s `isinstance` guard didn't include `str`, so `fire_type='surface'` raised `TypeError` even though string fire types are documented as valid. |
| `test_string_fire_type_active_crown_does_not_raise` | Same bug, second valid string value — confirms the fix isn't value-specific. |
| `test_string_fire_type_with_array_midflame_ws` | A second historical bug compounded the first: the `isinstance` check for the *numeric* fire_type branch mistakenly checked `midflame_ws` instead of `fire_type`, so this combination (string `fire_type` + array `midflame_ws`) raised `AttributeError`. Both fixes are required simultaneously for this case to pass. |
| `test_kwargs_signature_accepts_keyword_arguments` | Historical bug: `flame_component_array_multiprocessing` accepted `*kwargs` (a positional tuple) instead of `**kwargs` (a keyword dict), so calling it with keyword arguments (the only way its API is documented to be used) raised `TypeError`. Also asserts the concatenated block results are numerically identical to a direct `get_flame_residence_time` call — not just that no exception occurs. |
| `test_flame_residence_key_resolves_to_valid_function` | Historical bug: the dispatcher's function-name lookup string was `'getFlameResidence'` (missing `'Time'`), so `globals().get(...)` returned `None` and the dispatcher raised "function does not exist" for a documented, valid key. Also asserts the concatenated block results are numerically identical to a direct `get_flame_residence_time` call — the resolved function must be correct, not merely resolvable. |
