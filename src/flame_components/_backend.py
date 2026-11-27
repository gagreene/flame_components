"""Resolve the Python backend and load the optional native extension on demand."""

from importlib import import_module
from types import ModuleType
from typing import Literal

_NATIVE_MODULE = "flame_components._native"


def load_native() -> ModuleType:
    """Load the optional extension and report how to install it if absent."""
    try:
        return import_module(_NATIVE_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _NATIVE_MODULE:
            raise
        raise ImportError(
            "Install a Flame Components wheel with the Rust backend, or use backend='python'."
        ) from error


def resolve_backend(backend: str) -> Literal["python", "rust"]:
    """Validate the requested backend and select a concrete implementation."""
    if not isinstance(backend, str) or backend not in ("auto", "rust", "python"):
        raise ValueError("backend must be one of auto, rust, python")
    if backend == "rust":
        load_native()
        return "rust"
    if backend == "auto":
        try:
            import_module(_NATIVE_MODULE)
        except ModuleNotFoundError as error:
            if error.name != _NATIVE_MODULE:
                raise
            return "python"
        return "rust"
    return "python"
