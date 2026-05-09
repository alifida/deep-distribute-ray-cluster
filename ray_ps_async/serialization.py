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


def zeros_like_weights(w: List[np.ndarray]) -> List[np.ndarray]:
    return [np.zeros_like(x, dtype=np.float32) for x in w]


def encode_update(
    arrays: List[np.ndarray],
    method: str = "fp32",
    topk_pct: float = 1.0,
) -> tuple[list[list[float]], list[tuple[int, ...]], dict, int]:
    m = (method or "fp32").lower()
    lists: list[list[float]] = []
    shapes: list[tuple[int, ...]] = []
    meta: dict = {"method": m, "tensor_meta": []}
    tx_bytes = 0

    for arr in arrays:
        x = arr.astype(np.float32, copy=False).ravel()
        shp = tuple(arr.shape)
        shapes.append(shp)

        if m == "fp16":
            q = x.astype(np.float16)
            lists.append(q.astype(np.float32).tolist())
            meta["tensor_meta"].append({"dtype": "fp16"})
            tx_bytes += int(q.nbytes)
        elif m == "q8":
            max_abs = float(np.max(np.abs(x))) if x.size else 1.0
            scale = max(max_abs / 127.0, 1e-12)
            q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
            lists.append(q.astype(np.float32).tolist())
            meta["tensor_meta"].append({"dtype": "q8", "scale": scale})
            tx_bytes += int(q.nbytes)
        elif m == "topk":
            k = max(1, int(np.ceil((float(topk_pct) / 100.0) * max(1, x.size))))
            if x.size <= k:
                idx = np.arange(x.size, dtype=np.int32)
                vals = x.astype(np.float32)
            else:
                idx = np.argpartition(np.abs(x), -k)[-k:].astype(np.int32)
                vals = x[idx].astype(np.float32)
            # flattened sparse payload: [n, k, idx..., vals...]
            flat = [float(x.size), float(k)] + idx.astype(np.float32).tolist() + vals.tolist()
            lists.append(flat)
            meta["tensor_meta"].append({"dtype": "topk"})
            tx_bytes += int(idx.nbytes + vals.nbytes + 8)
        else:  # fp32
            q = x.astype(np.float32)
            lists.append(q.tolist())
            meta["tensor_meta"].append({"dtype": "fp32"})
            tx_bytes += int(q.nbytes)

    return lists, shapes, meta, tx_bytes


def decode_update(
    lists: list[list[float]],
    shapes: list[tuple[int, ...]],
    meta: dict | None,
) -> List[np.ndarray]:
    m = ((meta or {}).get("method") or "fp32").lower()
    tmeta = (meta or {}).get("tensor_meta") or []
    out: List[np.ndarray] = []
    for i, (vals, shp) in enumerate(zip(lists, shapes)):
        tm = tmeta[i] if i < len(tmeta) else {}
        if m == "q8":
            scale = float(tm.get("scale", 1.0))
            q = np.asarray(vals, dtype=np.int8).astype(np.float32)
            arr = (q * scale).reshape(shp)
        elif m == "topk":
            f = np.asarray(vals, dtype=np.float32)
            n = int(f[0]) if f.size else int(np.prod(shp))
            k = int(f[1]) if f.size > 1 else 0
            idx = f[2 : 2 + k].astype(np.int32)
            v = f[2 + k : 2 + (2 * k)].astype(np.float32)
            dense = np.zeros((n,), dtype=np.float32)
            if k > 0:
                dense[idx] = v
            arr = dense.reshape(shp)
        else:
            arr = np.asarray(vals, dtype=np.float32).reshape(shp)
        out.append(arr)
    return out

