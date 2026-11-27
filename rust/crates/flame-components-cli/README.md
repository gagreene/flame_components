# Flame Components CLI

The `flame-components` executable runs the six scalar calculations with the
standalone Rust library. It requires no Python, NumPy, or ArcGIS runtime.

Build and test from the repository's `rust/` directory.

```sh
cargo build --locked --release -p flame-components-cli
cargo test --locked -p flame-components -p flame-components-cli
```

The binary is `target/release/flame-components` (`flame-components.exe` on
Windows). Run `flame-components --help` to list the commands and
`flame-components <command> --help` for command-specific arguments.

## Scalar flags

```sh
flame-components mid-flame-ws --wind-speed 18 --canopy-cover 35 --canopy-ht 10 --canopy-baseht 0
flame-components flame-length --model Byram_HEAD --fire-intensity 425
flame-components flame-height --model Finney --flame-length 8 --flame-tilt 25 --slope-angle 1 --slope-units percent
flame-components flame-tilt --model Standard --flame-length 5 --flame-height 3
flame-components flame-residence-time --ros 3 --fuel-consumption 0.8 --midflame-ws 1.4 --units min
flame-components flame-depth --ros 3 --res-time 2
flame-components flame-length --model Finney_HEAD --params-only
```

The depth example prints `6` metres. The lookup prints `[0.01051,0.774,0.161]`
in coefficient, intensity exponent, and depth exponent order. Lookup requires
only the model selector. See the [library's model table](../flame-components/src/length.rs)
for all flame-length selectors. `Barbon_iNOWIND` is an alias for `Barbon_NOWIND`.

Wind speed outputs use metres per second. The default `SI` wind inputs use
kilometres per hour at 10 m and canopy heights in metres. `IMP` inputs use miles
per hour at 20 ft and canopy heights in feet. Canopy cover uses percent in
either system. Nelson height accepts fire-type codes `1`, `2`, and `3` or
the names `surface`, `passive crown`, and `active crown`. Quote names with spaces.

## JSON input and output

Pass a single JSON object from a file or pipe it to standard input. Use
underscores or hyphens in keys that correspond to multiword flags. Supply each
argument once, including when using different spellings of the same key.

```sh
flame-components flame-depth --json measurements.json
printf '%s\n' '{"ros":3,"res_time":2}' | flame-components flame-depth --json -
printf '%s\n' '{"model":"Byram_HEAD","params_only":true}' | flame-components flame-length --json -
```

PowerShell can pipe a string directly to `flame-components flame-depth --json -`.
Supply the object or scalar flags for each calculation. Input objects accept
numbers, booleans, strings for named options, and `null` for a missing
measurement. Use `"Infinity"`, `"-Infinity"`, or `"NaN"` in a numeric field for
non-finite input. The same tokens without JSON quoting work with scalar flags.
Boolean measurements act as `1` and `0`. An input NaN is masked.

Integer tokens must fit in signed 64 bits, from `-9223372036854775808` to
`9223372036854775807`. Depth preserves integral inputs, including values above
`2^53`. The Rust depth equation wraps products at 64 bits and then clamps
negative values to zero. Decimal or exponent notation requests binary64
floating-point input. With an integer and a floating-point depth input, both
use binary64. The other equations use binary64 for measurements.

Successful calculations write one JSON value and a newline to stdout. The
clamp changes negative results, including negative infinity, to zero. It
preserves signed zero and unmasked NaN. Masked results use `null`. Positive
infinity and unmasked NaN use the strings `"Infinity"` and `"NaN"`.
CLI diagnostics go to stderr. Invalid input exits with status 2 and leaves
stdout empty.
