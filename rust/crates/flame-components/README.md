# Flame Components for Rust

The library implements mid-flame wind speed, flame length, flame height, flame
tilt, flame residence time, and flame depth. Its build and runtime dependencies
are Rust crates. The minimum supported Rust compiler for the library is 1.78.

```rust
use flame_components::{depth::flame_depth, Cell};

let depth = flame_depth(Cell::present(3_i64), Cell::present(2_i64));
assert_eq!(depth, Cell::present(6));
```

Every calculation has a scalar and a broadcast array entry point. Array
functions accept read-only multidimensional views from the re-exported
`ndarray` crate, including 0-D and strided views. The model APIs use `f64`. Standard tilt also offers a typed `f32` path.
Depth multiplies in the caller's chosen common arithmetic type, preserving
integer width and wrapping at overflow.

Use `Cell::input` to normalize a floating-point NaN to a masked cell at input, or
`Cell::missing` to supply an explicit mask. Equation results retain the
mask separately from the numeric payload. Apply `Cell::nonnegative` to
match the Python facade's nonnegative clamp while preserving the Rust result
type. Callers can handle the missing cells when converting output to their
preferred format.

Model and unit enums describe the input fields for each calculation.
Mid-flame wind reports the zero-canopy substitution as a diagnostic in both
successful results and shape errors. Other shape or invalid model requests
return a typed error.

From the repository root, run the Rust unit tests, pipelines, and frozen
Python-reference comparisons with Cargo:

```text
cargo test --locked --manifest-path rust/Cargo.toml
```

The standalone package contains the unit tests and the depth/residence
integration tests. The repository checkout also supplies the shared frozen
fixtures for cross-language comparisons.
