# Frozen Python reference

Each line of reference.jsonl is one public call against the Git commit in
provenance.json. Python, Rust, and CLI tests can read the JSON without running
the generator. The Python integration tests select the Python backend explicitly.

- Inputs are named arguments. Numeric scalars use an object with an int or float
  field. The contents are decimal strings so integers retain full precision and
  floating-point numbers round-trip to binary64. The strings nan, inf, -inf,
  and -0.0 preserve special values and signed zero. Booleans and model names use
  JSON booleans and strings.
- Arrays have a kind, dtype, shape, and flat data in logical row-major order.
  An empty shape means a 0-D array or a NumPy scalar, depending on kind. A masked
  array has a flat Boolean mask in the same order as data. The reversed_first_axis
  layout marks an input with a negative stride.
- A successful case has output. Tuple outputs are coefficient lookups and require
  exact comparison. For arrays and NumPy scalars, kernel_result records the masked
  value supplied to the original public clamp. Payloads in masked cells are
  unspecified. The clamped_mask records missing cells before public NaN filling.
  Compare result shape, dtype, masks, integers, signed zeros, and non-finite
  classifications exactly. Cases with rtol permit that relative tolerance for
  finite values, with zero absolute tolerance.
- Branch records capture the actual masked condition passed to NumPy and its
  comparison operands. When available, ulps_from_right gives the signed distance
  between positive, finite binary64 operands. Null means the distance is
  undefined for that cell.
- An error has a category and message. Diagnostics preserve count, category,
  message, and whether the warning points to the caller. The observed NumPy
  arithmetic warnings are recorded separately because the public facade
  suppresses them.

Hand checks include integer depth 3 × 2 = 6, unsigned 8-bit depth
(200 × 2) mod 256 = 144 before the clamp, and Nelson height selecting the
2 m flame length at zero wind with missing intensity. The equality and
adjacent-float cases exercise distinct paths even when a numerical tolerance
could hide small output differences.

The checked-in data records the Windows NumPy 2.5.0 run. The Python tests account
for NumPy's platform default integer and arithmetic promotion differences
while keeping the stored reference intact. The clamp retains the result dtype,
unmasked NaNs, and signed zero. To reproduce the
capture with the recorded runtime, write it to a separate directory and compare
the result with the checked-in data.

    python tests/fixtures/parity/generate.py --output-dir <new-directory>

The generator extracts only package source from the pinned Git commit and checks
that the package it imports comes from that extraction.
