"""Select the engine for full public-API runs."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--calculation-backend", choices=("python", "rust"), default="python")


@pytest.fixture(scope="session")
def calculation_backend(pytestconfig):
    return pytestconfig.getoption("--calculation-backend")


@pytest.fixture(autouse=True)
def route_default_calculations_to_test_backend(calculation_backend, monkeypatch, request):
    if request.node.path.name == "test_backend.py":
        return

    from flame_components import _backend

    original = _backend.resolve_backend

    def resolve(backend):
        return original(calculation_backend if backend == "auto" else backend)

    monkeypatch.setattr(_backend, "resolve_backend", resolve)
