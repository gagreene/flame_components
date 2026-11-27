"""Tests for backend selection and dispatch to multiprocessing workers."""

import multiprocessing as mp
import sys
from types import ModuleType

import numpy as np
import pytest

from flame_components import (
    _backend,
    _blocks,
    core,
    flame_component_array_multiprocessing,
)


def test_python_selection_leaves_native_unimported(monkeypatch):
    def unexpected_import(name):
        pytest.fail(f"Unexpected native import: {name}")

    monkeypatch.setattr(_backend, "import_module", unexpected_import)
    assert _backend.resolve_backend("python") == "python"
    assert core.get_flame_depth(3, 2, backend="python") == 6


@pytest.mark.parametrize("backend", ["Rust", "", None, []])
def test_invalid_backend_raises_value_error(backend):
    with pytest.raises(ValueError, match="backend must be one of auto, rust, python"):
        _backend.resolve_backend(backend)


@pytest.mark.parametrize("backend", ["auto", "rust"])
def test_native_resolves_when_present(monkeypatch, backend):
    extension = ModuleType("flame_components._native")
    monkeypatch.setitem(sys.modules, extension.__name__, extension)

    assert _backend.resolve_backend(backend) == "rust"
    assert _backend.load_native() is extension


def test_auto_uses_python_when_extension_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "flame_components._native", None)

    assert _backend.resolve_backend("auto") == "python"
    assert core.get_flame_depth(3, 2) == 6


def test_explicit_rust_gives_install_guidance_when_native_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "flame_components._native", None)

    with pytest.raises(ImportError, match="Install .*Rust.*backend") as caught:
        _backend.resolve_backend("rust")

    assert type(caught.value) is ImportError
    assert caught.value.__cause__.name == "flame_components._native"


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'native_dependency'", name="native_dependency"),
        ImportError("DLL load failed", name="flame_components._native"),
        RuntimeError("native initialization failed"),
    ],
)
@pytest.mark.parametrize("backend", ["auto", "rust"])
def test_native_selection_propagates_import_failures(monkeypatch, error, backend):
    def broken_import(name):
        raise error

    monkeypatch.setattr(_backend, "import_module", broken_import)
    with pytest.raises(type(error)) as caught:
        _backend.resolve_backend(backend)

    assert caught.value is error


def test_worker_passes_concrete_backend_to_calculation(monkeypatch):
    def calculate_block(ros, res_time, *, backend):
        return backend, ros, res_time

    monkeypatch.setattr(core, "get_flame_depth", calculate_block)
    result = _blocks._multiprocessing_worker(
        "get_flame_depth", "rust", {"ros": 1, "res_time": 2}
    )

    assert result == ("rust", 1, 2)


def test_python_blocks_keep_mask_order_and_partial_block_under_spawn(monkeypatch):
    monkeypatch.setattr(_blocks.mp, "Pool", mp.get_context("spawn").Pool)
    ros = np.ma.array(
        [[1, 2], [3, 4], [5, 6]],
        mask=[[False, True], [False, False], [False, False]],
    )
    blocks = flame_component_array_multiprocessing(
        "flame_depth", 2, 2, backend="python", ros=ros, res_time=2
    )

    assert len(blocks) == 2
    np.testing.assert_array_equal(blocks[0], [[2, np.nan], [6, 8]])
    np.testing.assert_array_equal(blocks[1], [[10, 12]])


def test_public_call_validates_backend_before_equation_input():
    with pytest.raises(ValueError, match="backend must be one of auto, rust, python"):
        core.get_flame_depth([], 2, backend="wrong")


@pytest.mark.parametrize("backend", ["auto", "rust"])
def test_rust_dispatches_normalized_data_masks_and_raw_results(monkeypatch, backend):
    calls = []

    def flame_depth(ros, res_time):
        data, mask = ros
        times, time_mask = res_time
        calls.append((data, mask, times, time_mask))
        return np.array([4, 16], dtype=np.uint8), mask, False

    extension = ModuleType("flame_components._native")
    extension.flame_depth = flame_depth
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    result = core.get_flame_depth(
        np.ma.array([2, 8], dtype=np.uint8, mask=[False, True]),
        np.array([2, 2], dtype=np.uint8),
        backend=backend,
    )
    assert len(calls) == 1
    assert calls[0][0].dtype == np.uint8
    np.testing.assert_array_equal(calls[0][1], [False, True])
    np.testing.assert_array_equal(result, [4, np.nan])
