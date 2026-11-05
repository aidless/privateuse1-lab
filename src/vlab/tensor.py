"""A numpy-backed vlab tensor: the storage half of a PyTorch PrivateUse1 backend.

`torch._C._acc.create_empty_tensor(size, dtype)` mints a tensor that belongs to
the PrivateUse1 device type but has no storage of its own. The numpy array in
`_raw` is the stand-in for device memory. Ops in this package read and write
that array.
"""

from __future__ import annotations

import numpy as np
import torch

# the only dtypes this lab implements, and their numpy equivalents
_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


class VlabTensor(torch.Tensor):
    """PrivateUse1 tensor whose storage is a numpy array."""

    @staticmethod
    def __new__(cls, size, dtype, raw=None):
        # create_empty_tensor mints the PrivateUse1 tensor; __class__ swap makes
        # it ours so we can hang the backing store off it.
        res = torch._C._acc.create_empty_tensor(tuple(size), dtype)
        res.__class__ = cls
        return res

    def __init__(self, size, dtype, raw=None):
        self._raw = raw

    @property
    def raw(self) -> np.ndarray:
        raw = self.__dict__.get("_raw", None)
        if raw is None:
            raise RuntimeError("vlab tensor has no backing store (uninitialised?)")
        return raw

    @raw.setter
    def raw(self, value) -> None:
        object.__setattr__(self, "_raw", value)

    def __repr__(self) -> str:  # keep it readable in test failures
        try:
            return f"VlabTensor({self.raw!r})"
        except RuntimeError:
            return "VlabTensor(<uninitialised>)"

    __str__ = __repr__


def wrap(raw: np.ndarray, dtype: torch.dtype | None = None) -> VlabTensor:
    """Attach a numpy array to a fresh PrivateUse1 tensor.

    The array is kept as-is, *not* made contiguous: views (`expand`,
    `as_strided`, `transpose`) must alias the base buffer, and
    `np.ascontiguousarray` would silently copy them into independent memory.
    """
    arr = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
    if dtype is None:
        dtype = _torch_dtype(arr.dtype)
    return VlabTensor(arr.shape, dtype, arr)


def unwrap(t) -> np.ndarray:
    if not isinstance(t, VlabTensor):
        raise TypeError(f"expected a vlab tensor, got {type(t).__name__}")
    return t.raw


def _torch_dtype(np_dtype) -> torch.dtype:
    for tdt, ndt in _NP_DTYPE.items():
        if np.dtype(ndt) == np.dtype(np_dtype):
            return tdt
    raise TypeError(f"vlab does not implement numpy dtype {np_dtype}")


def _numpy_dtype(dtype: torch.dtype):
    try:
        return _NP_DTYPE[dtype]
    except KeyError:
        raise TypeError(f"vlab does not implement torch dtype {dtype}") from None
