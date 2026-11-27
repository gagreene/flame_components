"""
Standalone package smoke test for flame-components.

Run this against an installed package (wheel, sdist, or PyPI/TestPyPI install) in an
environment without the repository on PYTHONPATH. To smoke-test an unbuilt checkout
with NumPy installed, add src/ to PYTHONPATH and select the Python backend.

Usage: python scripts/smoke_test.py [--backend rust|python|auto]
"""
import argparse
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np

import flame_components as fc


def check_installed_version(cargo_manifest, release_tag, release_prerelease):
    from packaging.version import InvalidVersion, Version

    with cargo_manifest.open("rb") as manifest:
        cargo_version = tomllib.load(manifest)["workspace"]["package"]["version"]
    expected = Version(cargo_version)
    if release_tag:
        try:
            tag_version = Version(release_tag)
        except InvalidVersion as exc:
            raise AssertionError(f"invalid release tag: {release_tag!r}") from exc
        assert release_tag.startswith("v") and str(tag_version) == str(expected), (
            f"release tag {release_tag!r} does not match Cargo version {cargo_version!r}"
        )
        assert expected.is_prerelease == release_prerelease, (
            f"GitHub pre-release setting does not match version {expected}"
        )

    assert Path(fc.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
        f"flame_components was not imported from the test environment: {fc.__file__}"
    )
    installed = version("flame-components")
    assert installed == str(expected) and fc.__version__ == installed, (
        f"expected {expected} from Cargo version {cargo_version!r}, got "
        f"metadata {installed!r} and __version__ {fc.__version__!r}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "python", "rust"), default="auto")
    parser.add_argument("--cargo-manifest", type=Path, help="workspace Cargo.toml for version checks")
    parser.add_argument("--release-tag", default="", help="GitHub release tag to verify")
    parser.add_argument("--release-prerelease", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if args.release_tag and args.cargo_manifest is None:
        parser.error("--release-tag requires --cargo-manifest")
    if args.cargo_manifest is not None:
        check_installed_version(
            args.cargo_manifest, args.release_tag, args.release_prerelease == "true"
        )

    assert isinstance(fc.__version__, str) and fc.__version__, (
        f"__version__ missing or empty: {fc.__version__!r}"
    )

    expected_names = {
        'get_mid_flame_ws',
        'get_flame_length',
        'get_flame_height',
        'get_flame_tilt',
        'get_flame_residence_time',
        'get_flame_depth',
        'flame_component_array_multiprocessing',
    }
    assert set(fc.__all__) == expected_names, f"unexpected __all__: {fc.__all__!r}"
    for name in fc.__all__:
        assert callable(getattr(fc, name)), f"{name} in __all__ is not callable"

    # Scalar call
    mid_flame_ws = fc.get_mid_flame_ws(
        wind_speed=15, canopy_cover=50, canopy_ht=10, canopy_baseht=2, units='SI',
        backend=args.backend,
    )
    assert isinstance(mid_flame_ws, float) and mid_flame_ws > 0, (
        f"unexpected scalar result: {mid_flame_ws!r}"
    )

    # Array call
    fire_intensity = np.array([100.0, 500.0, 1000.0])
    flame_length = fc.get_flame_length(
        model='Byram_HEAD', fire_intensity=fire_intensity, backend=args.backend
    )
    assert isinstance(flame_length, np.ndarray) and flame_length.shape == (3,), (
        f"unexpected array result: {flame_length!r}"
    )

    # Multiprocessing call
    wind_speed = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    canopy_cover = np.array([40, 50, 60, 40, 50, 60])
    canopy_ht = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    canopy_baseht = np.array([2.0, 2.0, 3.0, 2.0, 2.0, 3.0])
    blocks = fc.flame_component_array_multiprocessing(
        flame_function='midflame_ws',
        num_processors=2,
        wind_speed=wind_speed,
        canopy_cover=canopy_cover,
        canopy_ht=canopy_ht,
        canopy_baseht=canopy_baseht,
        units='SI',
        backend=args.backend,
    )
    result = np.concatenate(blocks)
    assert result.shape == wind_speed.shape, f"unexpected multiprocessing result shape: {result.shape!r}"

    print(f"Smoke test passed: flame_components {fc.__version__} (backend={args.backend})")


if __name__ == "__main__":
    # Guarded per Python's multiprocessing requirements -- flame_component_array_
    # multiprocessing spawns worker processes and must not run at import time.
    try:
        main()
    except AssertionError as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
