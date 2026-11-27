# Flame Components for Rust

The library implements mid-flame wind speed, flame length, flame height, flame
tilt, flame residence time, and flame depth. The default build uses Rust crates
and supports Rust 1.78 and later.

Use a Cargo path dependency on `rust/crates/flame-components` in a source
checkout. For example, if the checkout is next to your Cargo project, add this
to your `Cargo.toml`.

```toml
[dependencies]
flame-components = { path = "../flame_components/rust/crates/flame-components" }
```

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

To build and verify the source archives, use a current stable Cargo from the
repository's `rust/` directory.

```text
cargo +stable package --locked -p flame-components-wide -p flame-components --features flame-components/extended
```

The library archive contains the unit tests and the depth/residence integration
tests. The repository checkout also supplies the shared frozen fixtures for
cross-language comparisons. Packaging the library alone for crates.io requires
the exact helper version declared in its manifest to be available in the
registry, even for a default-feature build.

The optional `extended` feature adds the host's C `long double` representation
and arithmetic to the Rust kernels. Building with this feature requires a C
compiler. The Python extension enables it to preserve compatible NumPy
`longdouble` values and masks through the Rust models.
