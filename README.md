
# Flame Components

This repository provides several functions for calculating various flame characteristics such as mid-flame wind speed, flame length, flame height, flame tilt, flame residence time, and flame depth. It also includes utility functions for multiprocessing these calculations across blocks of data, which can be useful for high-performance fire modeling.

## Features

- **Mid-Flame Wind Speed Calculation** (`get_mid_flame_ws`): Calculates mid-flame wind speed based on parameters such as wind speed, canopy cover, and canopy height.
- **Flame Length Estimation** (`get_flame_length`): Estimates flame length using different published models.
- **Flame Height Calculation** (`get_flame_height`): Calculates flame height based on flame length and model-specific parameters.
- **Flame Tilt Angle Calculation** (`get_flame_tilt`): Calculates the angle of flame tilt relative to vertical.
- **Flame Residence Time** (`get_flame_residence_time`): Computes flame residence time based on rate of spread, fuel consumption, and wind speed.
- **Flame Depth Calculation** (`get_flame_depth`): Calculates flame depth using flame residence time and rate of spread.
- **Array Multiprocessing** (`flame_component_array_multiprocessing`): Enables multiprocessing of flame component calculations across blocks of data, making large-scale processing efficient.

## Requirements

- Python 3.11+
- **Libraries**: `numpy`

## Installation

```bash
pip install flame-components
```

**Distribution name vs. import name:** the package is published to PyPI as
`flame-components` (hyphen, used with `pip install`), but imported in Python as
`flame_components` (underscore, a valid identifier):

```python
import flame_components
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
packaging.

```bash
uv sync              # create .venv and install runtime + dev dependencies
uv run pytest tests/ -v   # run the test suite
uv build             # build the wheel and sdist into dist/
```

## Rust library

The [standalone Rust crate](rust/crates/flame-components/README.md) provides
scalar and array APIs for all six calculations. Rust 1.78 or later can build
and test it without a Python environment.

```text
cargo test --locked --manifest-path rust/Cargo.toml
```

## Standalone command line

Build the executable with `cargo build --release --locked --manifest-path
rust/Cargo.toml -p flame-components-cli`. For example,
`flame-components flame-depth --ros 3 --res-time 2` prints `6`.
The CLI accepts scalar flags and scalar JSON from a file or standard input.
[CLI reference](rust/crates/flame-components-cli/README.md)

## Usage

### Key Functions

- **`get_mid_flame_ws`**: Calculates mid-flame wind speed.
- **`get_flame_length`**: Estimates flame length based on specified models.
- **`get_flame_height`**: Calculates flame height for a given flame length.
- **`get_flame_tilt`**: Computes flame tilt angle using various models.
- **`get_flame_residence_time`**: Estimates flame residence time for a given rate of spread.
- **`get_flame_depth`**: Computes flame depth from flame residence time and rate of spread.

### Example

```python
from flame_components import get_mid_flame_ws, get_flame_length

# Example calculation for mid-flame wind speed
mid_flame_ws = get_mid_flame_ws(
    wind_speed=15,
    canopy_cover=50,
    canopy_ht=10,
    canopy_baseht=2,
    units='SI'
)

# Example calculation for flame length
flame_length = get_flame_length(
    model='Byram_HEAD',
    fire_intensity=500
)
```

### Array Multiprocessing Example

To perform calculations across blocks of data using multiple processors:

```python
import numpy as np
from flame_components import flame_component_array_multiprocessing

wind_speed = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
canopy_cover = np.array([40, 50, 60, 40, 50, 60])
canopy_ht = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
canopy_baseht = np.array([2.0, 2.0, 3.0, 2.0, 2.0, 3.0])

# Returns a list of per-block result arrays, one per worker, in block order --
# concatenate them yourself if you need a single array back.
blocks = flame_component_array_multiprocessing(
    flame_function='midflame_ws',
    num_processors=2,
    wind_speed=wind_speed,
    canopy_cover=canopy_cover,
    canopy_ht=canopy_ht,
    canopy_baseht=canopy_baseht,
    units='SI'
)
mid_flame_ws = np.concatenate(blocks)
```

`kwargs` passed to `flame_component_array_multiprocessing` must match the parameter names
of the target function (here, `get_mid_flame_ws`'s `wind_speed`/`canopy_cover`/`canopy_ht`/
`canopy_baseht`/`units`). See the API Reference below for `flame_function`'s valid values
and each target function's required parameters.

## API Reference

All functions accept `int`/`float` scalars or `numpy.ndarray` (NaN cells are masked and
propagate as NaN in the output). Passing at least one array returns an array; all-scalar
input returns a Python `float`. Every function raises `TypeError` for a wrong input type
and `ValueError` for an invalid string option (e.g. an unknown `model` or `units`) or a
missing model-specific required parameter.

Every calculation and the multiprocessing helper accepts a keyword-only
`backend` argument. The default `backend="auto"` selects Rust when the compiled
extension is installed and uses NumPy when it is absent. An error loading an
installed extension propagates to the caller. Use `backend="python"` or
`backend="rust"` to select an engine explicitly. Rust supports extended NumPy
`longdouble` inputs when the extension and NumPy use the same native format.

The NumPy backend suppresses incidental arithmetic warnings. Explicit NumPy
error modes such as `over="raise"` still apply. The package reports
`UserWarning` for conditions such as zero canopy height and a processor count
below two.

#### `get_mid_flame_ws(wind_speed, canopy_cover, canopy_ht, canopy_baseht, units='SI', *, backend="auto")`
Mid-flame wind speed (m/s) under a canopy — Albini & Baughman (1979).
- `units`: `'SI'` (10-m wind in km/h, heights in m) or `'IMP'` (20-ft wind in mi/h, heights in ft).
- `canopy_ht == 0` is substituted with a 0.5 m equivalent and raises `UserWarning`.
- Output is always m/s regardless of input units.

#### `get_flame_length(model, fire_intensity, flame_depth=None, params_only=False, *, backend="auto")`
Flame length (m) from head fire intensity (kW/m), or the raw model coefficients if
`params_only=True`. `model` is one of 29 published models (see `model_dict` in
`src/flame_components/core.py` for the full list and per-model citation); `flame_depth`
is required only for `model='Finney_HEAD'`.

#### `get_flame_height(model, flame_length, fire_type=None, fire_intensity=None, midflame_ws=None, flame_tilt=None, slope_angle=None, slope_units=None, *, backend="auto")`
Flame height (m). `model='Nelson'` requires `fire_type` (`1`/`'surface'`, `2`/`'passive
crown'`, `3`/`'active crown'`), `fire_intensity`, and `midflame_ws`. `model='Finney'`
requires `flame_tilt`, `slope_angle`, and `slope_units` (`'degrees'` or `'percent'`).

#### `get_flame_tilt(model, flame_length=None, flame_height=None, slope_angle=None, slope_units=None, wind_speed=None, wind_speed_units=None, canopy_ht=None, *, backend="auto")`
Flame tilt angle from vertical (degrees). `model='Standard'` (flat ground) requires
`flame_length`/`flame_height`. `model='Finney'` (sloped ground) additionally requires
`slope_angle`/`slope_units`. `model='Butler'` (crown fires only) requires `wind_speed`,
`wind_speed_units` (`'kph'`/`'mps'`/`'mph'`), and `canopy_ht` (must be `> 0`, or raises
`ValueError`).

#### `get_flame_residence_time(ros, fuel_consumption, midflame_ws, units, *, backend="auto")`
Flame residence time — Nelson & Adkins (1988). `ros` must always be in m/min; `units`
(`'sec'` or `'min'`) controls the output unit only, not the expected input unit.

#### `get_flame_depth(ros, res_time, *, backend="auto")`
Flame depth (m) — Fons et al. (1963). `ros` in m/min, `res_time` in minutes.

#### `flame_component_array_multiprocessing(flame_function, num_processors=2, block_size=None, *, backend="auto", **kwargs)`
Splits array inputs into blocks and runs one of the six functions above across multiple
processes. `flame_function` is one of `'midflame_ws'`, `'flame_length'`, `'flame_height'`,
`'flame_tilt'`, `'flame_residence'`, `'flame_depth'`. `kwargs` must match the target
function's parameter names, with at least one `numpy.ndarray` value (all arrays must share
the same shape). `num_processors < 2` is coerced to `2` with a `UserWarning`. Returns a
**list of per-block result arrays, not a single concatenated array** — call
`numpy.concatenate()` on the result yourself if you need one.

See `docs/CODEBASE.md` for internal architecture and implementation-level gotchas, and
`docs/TEST_SUITE_DESCRIPTION.md` for the full test-suite catalog.

## Scientific References

The equations implemented here are drawn from the published fire-behavior literature.
Citations below are as referenced in each function's docstring in
`src/flame_components/core.py`; consult the original publications for full bibliographic
detail (journal, volume, DOI) before citing this package's underlying science formally.

- **`get_mid_flame_ws`**: Albini, F.A. & Baughman, R.G. (1979).
- **`get_flame_length`**: 29 models catalogued in Finney, M.A. & Grumstrup, T.P. (2023);
  each model's original source (Byram, Fons, Nelson, Anderson, Fernandes, Butler, and
  others) is cited inline next to its coefficients in `core.py`'s `model_dict`.
- **`get_flame_height`**: Nelson, R.M. & Adkins, C.W. (1986) (`model='Nelson'`), or
  Finney, M.A. & Martin, S.W. (1992) (`model='Finney'`).
- **`get_flame_tilt`**: Standard geometry (`model='Standard'`), Finney, M.A. & Martin, S.W.
  (1992) (`model='Finney'`), Butler, B.W. et al. (2004) (`model='Butler'`).
- **`get_flame_residence_time`**: Nelson, R.M. & Adkins, C.W. (1988).
- **`get_flame_depth`**: Fons, W.L. et al. (1963).

If you use this package in published research, please cite both the package itself (see
`CITATION.cff`) and the original publication(s) for the specific model(s) you used.

## Support and Issue Reporting

Please open a [GitHub issue](https://github.com/gagreene/flame_components/issues) for
bugs, questions, or feature requests. To reproduce a numerical discrepancy quickly,
please include:

- the installed `flame-components` version (`flame_components.__version__`) and Python/
  numpy versions;
- the exact function call, including whether inputs were scalars or arrays;
- the expected result and the actual result you got.

## Contributing

Contributions are welcome. If you have ideas for new features or improvements, please submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the
[LICENSE](https://github.com/gagreene/flame_components/blob/master/LICENSE) file for details.
