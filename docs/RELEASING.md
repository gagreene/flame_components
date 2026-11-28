# Release Procedure

This describes how to cut a release of `flame-components` once Trusted Publishing is
configured (one-time setup below). It does not need updating for each release — it
describes the current process, not a specific release's history.

## One-time setup (do this once, before the first automated publish)

These steps require the PyPI/TestPyPI web UI and GitHub repository settings — they can't
be automated from a local session.

1. **GitHub environments.** In the repo's Settings → Environments, create `pypi` and
   `testpypi`. Add a required reviewer to `pypi` (manual approval before every production
   publish); `testpypi` can be left unprotected since nothing it touches is a real release.
2. **Trusted Publishers.** On both [pypi.org](https://pypi.org/manage/account/publishing/)
   and [test.pypi.org](https://test.pypi.org/manage/account/publishing/), register a
   *pending* trusted publisher (the project doesn't need to exist yet) with:
   - Owner: `gagreene`, Repository: `flame_components`
   - Workflow name: `release.yml`
   - Environment name: `pypi` (on pypi.org) / `testpypi` (on test.pypi.org) — must match
     the `environment:` name in `.github/workflows/release.yml` exactly.
3. Configure branch protection for `master` and `pre_release` to require the `Tests`
   workflow's checks before merge.

## Cutting a release

1. Make sure `master` is green (the `Tests` workflow passing on the latest commit) and the
   worktree is clean.
2. Set the release SemVer in `[workspace.package]` in `rust/Cargo.toml`, refresh
   `rust/Cargo.lock` with `cargo update --workspace --manifest-path rust/Cargo.toml`,
   and get those changes onto `master` before tagging. The library, CLI, and Python
   extension inherit this version. Maturin derives the Python version from the
   extension crate and converts it to PEP 440. Cargo `0.2.0-rc.1` becomes Python
   `0.2.0rc1`.
3. Create and push an annotated tag with the matching Python version.

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

   For a TestPyPI rehearsal, tag the release candidate as `v0.2.0rc1`.
   For a final release, check that the versioned README and Python metadata
   links point at the published tag. Verify the release-notes link once the
   GitHub Release is available.
4. Create a GitHub Release from that tag (Releases → Draft a new release), with curated
   user-facing notes (see "What goes in release notes" below).
   - **Mark it as a pre-release** for an `rcN` tag to publish to TestPyPI.
   - **Leave it as a full release** for a final version tag to publish to PyPI.
5. Publishing the GitHub Release runs the distribution workflow. It builds and tests
   CPython 3.11–3.14 wheels for Linux x86-64 (manylinux 2.28), Windows x86-64, and
   macOS Apple Silicon. It also builds and installs the source distribution with
   Rust 1.78. Each installed artifact is checked against the Cargo version and
   release tag. Standalone CLI binaries are built and tested on all three platforms
   without Python. The workflow runs `twine check` on the Python distributions and
   attaches them and the CLI binaries to the GitHub Release before publishing.
6. Approve the environment deployment if GitHub prompts for it. Production publishing
   requires the reviewer configured in the one-time setup.
7. After publishing, install `flame-components==<version>` in a clean environment and
   run `python scripts/smoke_test.py --backend rust` from a checkout. For a TestPyPI
   rehearsal, use the TestPyPI release with NumPy installed from PyPI.

Wheels include the Rust extension. Installing the source distribution requires a
Rust toolchain and a C compiler for the extended-precision arithmetic shim. The standalone CLI runs independently of the Python package.

## Rust source archive

The standalone library can be used as a Cargo path dependency from the checkout.
From `rust/`, verify the single library archive with its optional extended
arithmetic enabled.

```bash
cargo package --locked -p flame-components --features extended
```

GitHub's standalone Rust workflow verifies this archive. The source for the
optional `extended` feature ships inside `flame-components`. A crates.io release
publishes that library crate.

## What goes in release notes

Only externally meaningful information: new features, behavior changes, compatibility
changes, deprecations/removals, and fixes that could affect users. Internal
implementation cleanup, test-suite restructuring, and detailed bug-discovery narrative
belong in the private Obsidian vault's Decision Log, not here.

## If something goes wrong

PyPI files and version numbers **cannot be replaced or re-uploaded** — once a version is
public, that exact set of bytes is permanent (even if deleted, the version number can
never be reused). Options for a broken release:

- **Yank it** (PyPI project page → Manage → yank a release) if it's broken enough that
  people shouldn't install it, but a small number of people already have working pins to
  it. A yanked release stays downloadable by exact version pin but is hidden from normal
  resolution.
- **Ship a new patch version** with the fix. This is the normal path for almost anything —
  there is no "fix in place."
- If the *tag* was wrong (pointed at the wrong commit) but nothing was published yet,
  it's fine to delete and recreate the tag — see the Decision Log's `v0.1.0` re-pointing
  entry for a precedent. Once something has actually been published to PyPI under that
  version, don't re-point the tag; cut a new version instead.
