from typing import List

import numpy as np


def weights_to_lists(weights: List[np.ndarray]) -> tuple[list[list[float]], list[tuple[int, ...]], int]:
    lists: list[list[float]] = []
    shapes: list[tuple[int, ...]] = []
    byte_size = 0
    for w in weights:
        arr = w.astype(np.float32, copy=False)
        lists.append(arr.ravel().tolist())
        shapes.append(tuple(arr.shape))
        byte_size += int(arr.nbytes)
    return lists, shapes, byte_size


def lists_to_weights(lists: list[list[float]], shapes: list[tuple[int, ...]]) -> List[np.ndarray]:
    return [np.asarray(vals, dtype=np.float32).reshape(shape) for vals, shape in zip(lists, shapes)]


def subtract_weights(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [x - y for x, y in zip(a, b)]


def add_weights(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [x + y for x, y in zip(a, b)]

