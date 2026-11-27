"""Check where zero-dimensional masked-array arithmetic returns NumPy scalars."""

import numpy as np
import pytest

from flame_components import get_flame_height, get_mid_flame_ws


@pytest.fixture(params=["python", "rust"])
def backend(request):
    if request.param == "rust":
        pytest.importorskip("flame_components._native")
    return request.param


@pytest.mark.parametrize("shape", [(), (1,)])
@pytest.mark.parametrize("mask_length", [False, True])
def test_finney_height_at_vertical_slope(backend, shape, mask_length):
    length = np.ma.array(np.full(shape, 8.0), mask=mask_length)
    angle = np.full(shape, 90.0)
    result = get_flame_height(
        "Finney", length, flame_tilt=angle, slope_angle=angle, slope_units="degrees",
        backend=backend,
    )
    assert result.shape == shape
    np.testing.assert_equal(result, np.full(shape, np.nan if shape or mask_length else 0.0))


@pytest.mark.parametrize(
    ("wind_shape", "cover_shape", "expected"),
    [((), (), 1.1318895505795175), ((1,), (), 1.1318895505795175), ((), (1,), np.nan)],
)
def test_wind_shelter_comparison_with_infinite_cover(backend, wind_shape, cover_shape, expected):
    result = get_mid_flame_ws(
        np.full(wind_shape, 10.0), np.full(cover_shape, -np.inf),
        np.array(1.0), np.array(0.0), backend=backend,
    )
    shape = np.broadcast_shapes(wind_shape, cover_shape)
    assert result.shape == shape
    np.testing.assert_allclose(result, np.full(shape, expected), rtol=2e-14)
