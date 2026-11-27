"""Partition array inputs and collect calculation results in row order."""

import multiprocessing as mp

from numpy import ndarray

from . import _utils

_FUNCTION_NAMES = {
    "midflame_ws": "get_mid_flame_ws",
    "flame_length": "get_flame_length",
    "flame_height": "get_flame_height",
    "flame_tilt": "get_flame_tilt",
    "flame_residence": "get_flame_residence_time",
    "flame_depth": "get_flame_depth",
}


def prepare_blocks(flame_function: str, num_processors: int, kwargs: dict) -> tuple[str, int, int]:
    """Validate the function and inputs before the facade handles the processor count."""
    _utils.check_choice("flame_function", flame_function, _FUNCTION_NAMES)
    row_count = _utils.common_row_count(kwargs)
    num_processors = _utils.check_int("num_processors", num_processors)
    return _FUNCTION_NAMES[flame_function], row_count, num_processors


def _multiprocessing_worker(function_name: str, backend: str, kwargs: dict):
    """Calculate one block with the backend selected by the parent process."""
    from . import core

    function = getattr(core, function_name)
    return function(backend=backend, **kwargs)


def calculate_blocks(
    function_name: str,
    row_count: int,
    num_processors: int,
    block_size: int | None,
    backend: str,
    kwargs: dict,
) -> list:
    """Split the arrays on their first axis and dispatch the ordered blocks."""
    if block_size is None:
        block_size = max(1, row_count // num_processors)
    else:
        block_size = _utils.check_int("block_size", block_size, minimum=1)

    tasks = []
    for start in range(0, row_count, block_size):
        rows = slice(start, start + block_size)
        block = {
            key: value[rows] if isinstance(value, ndarray) else value
            for key, value in kwargs.items()
        }
        tasks.append((function_name, backend, block))

    with mp.Pool(processes=min(num_processors, len(tasks))) as pool:
        return pool.starmap(_multiprocessing_worker, tasks)
