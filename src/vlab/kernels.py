"""Pure-numpy kernels behind the vlab ops — no torch, no dispatch.

Keeping the math here (instead of inline in `ops.py`) is what makes the backend
testable: the dispatcher wrappers are trivial argument plumbing, and the kernels
can be fed bad variants to prove the test suite actually catches broken math
(tests/mutation_gate.py patches these functions and requires every injection to
be caught).
"""

from __future__ import annotations

import numpy as np


def softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable softmax along `axis` (max-shift, then normalise)."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def mm_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def bmm_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)


def sum_dim_np(x: np.ndarray, dims, keepdim: bool) -> np.ndarray:
    return x.sum(axis=tuple(dims), keepdims=bool(keepdim))


def transpose_np(x: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
    return np.swapaxes(x, dim0, dim1)


def expand_np(x: np.ndarray, size) -> np.ndarray:
    """Broadcast view (alias, not a copy) — matters: in-place writes must be seen."""
    return np.broadcast_to(x, tuple(size))


def as_strided_np(x: np.ndarray, size, stride, offset: int = 0) -> np.ndarray:
    return np.lib.stride_tricks.as_strided(x, tuple(size), tuple(stride), offset=offset)


def view_np(x: np.ndarray, size) -> np.ndarray:
    return x.reshape(tuple(size))


def relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def attention_block_np(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float):
    """One scaled-dot-product attention block, expressed only through kernels.

    q [B, H, N, D], k/v [B, H, M, D]. Returns (out, probs) so tests can compare
    both the intermediate probabilities and the final result.
    """
    scores = mm_np(q, transpose_np(k, -1, -2)) * scale
    probs = softmax_np(scores, axis=-1)
    return mm_np(probs, v), probs
