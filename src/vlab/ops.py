"""Op registrations for the vlab PrivateUse1 dispatch key.

Each function is a ``torch.library.impl(..., "privateuseone")`` kernel. The
attention test drives a subset of them through torch and compares to CPU torch
within a float32 tolerance (``tests/test_attention_e2e.py``). It does not cover
every function in this file. Known holes are listed in ``docs/FAILURE_MODES.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _state
from . import kernels
from .tensor import VlabTensor, _numpy_dtype, unwrap, wrap


def _val(x):
    """Operand -> numpy value.

    Torch passes wrapped python scalars to the *Tensor* overloads (verified:
    `t + 2.5` lands on aten::add.Tensor with a float), and CPU 0-dim tensors show
    up as wrapped numbers in mixed expressions. A backend kernel must accept all
    three shapes of operand or the operator silently rejects the device.
    """
    if isinstance(x, VlabTensor):
        return x.raw
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return x


def _alloc(size, dtype, fill=None):
    np_dtype = _numpy_dtype(dtype if dtype is not None else torch.float32)
    arr = np.empty(size, dtype=np_dtype)
    if fill is not None:
        arr.fill(fill)
    _state.record_alloc(arr.nbytes)
    return wrap(arr)


# --------------------------------------------------------------------- factory


@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    return _alloc(tuple(size), dtype)


@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    # `stride` is ignored. The buffer is a contiguous allocation of `size`.
    return _alloc(tuple(size), dtype)


@torch.library.impl("aten::zeros", "privateuseone")
def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _alloc(tuple(size), dtype, fill=0)


@torch.library.impl("aten::ones", "privateuseone")
def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _alloc(tuple(size), dtype, fill=1)


@torch.library.impl("aten::randn", "privateuseone")
def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    # Reseeds numpy's legacy global RNG from the stored seed on every call.
    # `manual_seed_all` therefore repeats the same draw; it does not advance a
    # generator. See docs/FAILURE_MODES.md.
    np.random.seed(_state.get_seed())
    arr = np.random.randn(*tuple(size)).astype(_numpy_dtype(dtype or torch.float32))
    _state.record_alloc(arr.nbytes)
    return wrap(arr)


# ------------------------------------------------------------ data movement


@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(a, b, non_blocking=False):
    if isinstance(a, VlabTensor):
        # host -> device: take the host bytes
        src = b.numpy() if not isinstance(b, VlabTensor) else b.raw
        a.raw = np.array(src, copy=True)
        return a
    if isinstance(b, VlabTensor):
        # device -> host: write into the caller's host tensor
        a.copy_(torch.from_numpy(np.asarray(b.raw).copy()))
        return a
    raise TypeError("vlab::_copy_from needs at least one vlab tensor")


@torch.library.impl("aten::detach", "privateuseone")
def detach(self):
    # shares the buffer: a detached view of device memory, not a copy
    out = wrap(self.raw)
    return out


@torch.library.impl("aten::clone", "privateuseone")
def clone(self, *, memory_format=None):
    out = wrap(np.array(self.raw, copy=True))
    _state.record_alloc(self.raw.nbytes)
    return out


# --------------------------------------------------------------------- views


@torch.library.impl("aten::view", "privateuseone")
def view(self, size):
    return wrap(kernels.view_np(self.raw, size))


@torch.library.impl("aten::_unsafe_view", "privateuseone")
def _unsafe_view(self, size):
    return wrap(kernels.view_np(self.raw, size))


@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(self, size, stride, storage_offset=None):
    off = int(storage_offset) if storage_offset is not None else 0
    return wrap(kernels.as_strided_np(self.raw, size, stride, off))


@torch.library.impl("aten::expand", "privateuseone")
def expand(self, size, *, implicit=False):
    return wrap(kernels.expand_np(self.raw, size))


@torch.library.impl("aten::transpose.int", "privateuseone")
def transpose(self, dim0, dim1):
    return wrap(kernels.transpose_np(self.raw, int(dim0), int(dim1)))


@torch.library.impl("aten::t", "privateuseone")
def t(self):
    return wrap(self.raw.T)


# ------------------------------------------------------------------- elementwise


@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(a, b, alpha=1):
    return wrap(_val(a) + _val(b) * alpha)


@torch.library.impl("aten::add.Scalar", "privateuseone")
def add_scalar(a, b, alpha=1):
    return wrap(unwrap(a) + b * alpha)


@torch.library.impl("aten::sub.Tensor", "privateuseone")
def sub_tensor(a, b, alpha=1):
    return wrap(_val(a) - _val(b) * alpha)


@torch.library.impl("aten::sub.Scalar", "privateuseone")
def sub_scalar(a, b, alpha=1):
    return wrap(unwrap(a) - b * alpha)


@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul_tensor(a, b):
    return wrap(_val(a) * _val(b))


@torch.library.impl("aten::mul.Scalar", "privateuseone")
def mul_scalar(a, b):
    return wrap(unwrap(a) * b)


@torch.library.impl("aten::div.Tensor", "privateuseone")
def div_tensor(a, b):
    return wrap(_val(a) / _val(b))


@torch.library.impl("aten::div.Scalar", "privateuseone")
def div_scalar(a, b):
    return wrap(unwrap(a) / b)


@torch.library.impl("aten::neg", "privateuseone")
def neg(self):
    return wrap(-self.raw)


@torch.library.impl("aten::exp", "privateuseone")
def exp(self):
    return wrap(np.exp(self.raw))


@torch.library.impl("aten::relu", "privateuseone")
def relu(self):
    return wrap(kernels.relu_np(self.raw))


@torch.library.impl("aten::pow.Scalar", "privateuseone")
def pow_scalar(a, b):
    return wrap(np.power(unwrap(a), b))


@torch.library.impl("aten::pow.Tensor_Scalar", "privateuseone")
def pow_tensor_scalar(a, b):
    return wrap(np.power(unwrap(a), b))


@torch.library.impl("aten::pow.Tensor_Scalar_out", "privateuseone")
def pow_tensor_scalar_out(a, b, out=None):
    # `out` is ignored. This returns a new tensor.
    return wrap(np.power(unwrap(a), b))


@torch.library.impl("aten::pow.Tensor_Tensor", "privateuseone")
def pow_tensor_tensor(a, b):
    return wrap(np.power(unwrap(a), unwrap(b)))


# --------------------------------------------------------------- reductions


@torch.library.impl("aten::sum", "privateuseone")
def sum_all(self, *, dtype=None):
    out = np.asarray(self.raw.sum(), dtype=_numpy_dtype(dtype or torch.float32))
    return wrap(out)


@torch.library.impl("aten::sum.dim_IntList", "privateuseone")
def sum_dim(self, dim, keepdim=False, *, dtype=None):
    dims = tuple(int(d) for d in dim)
    out = kernels.sum_dim_np(self.raw, dims, keepdim)
    if dtype is not None:
        out = out.astype(_numpy_dtype(dtype))
    return wrap(out)


@torch.library.impl("aten::max.dim", "privateuseone")
def max_dim(self, dim, keepdim=False):
    idx = self.raw.argmax(axis=int(dim))
    vals = self.raw.max(axis=int(dim), keepdims=bool(keepdim))
    # `keepdim` changes the value rank only. Index rank is unchanged.
    return wrap(np.asarray(vals)), wrap(np.asarray(idx, dtype=np.int64))


# ------------------------------------------------------------- softmax / gemm


@torch.library.impl("aten::softmax.int", "privateuseone")
def softmax(self, dim, *, dtype=None):
    return wrap(kernels.softmax_np(self.raw, int(dim)))


@torch.library.impl("aten::mm", "privateuseone")
def mm(a, b):
    return wrap(kernels.mm_np(unwrap(a), unwrap(b)))


@torch.library.impl("aten::matmul", "privateuseone")
def matmul(a, b):
    return wrap(kernels.bmm_np(unwrap(a), unwrap(b)))


@torch.library.impl("aten::bmm", "privateuseone")
def bmm(a, b):
    return wrap(kernels.bmm_np(unwrap(a), unwrap(b)))


# ------------------------------------------------------------------ in-place


@torch.library.impl("aten::fill_.Scalar", "privateuseone")
def fill_scalar(self, value):
    self.raw.fill(value)
    return self


@torch.library.impl("aten::zero_", "privateuseone")
def zero_(self):
    self.raw.fill(0)
    return self


@torch.library.impl("aten::copy_", "privateuseone")
def copy_(self, src, non_blocking=False):
    # dispatched here whenever either side is a vlab tensor, so both directions
    # must be handled: device<-host (write our buffer) and host<-device (fill the
    # caller's CPU tensor with torch.from_numpy of our buffer).
    if isinstance(self, VlabTensor):
        self.raw[...] = src.raw if isinstance(src, VlabTensor) else src.numpy()
        return self
    if isinstance(src, VlabTensor):
        # np.asarray (not ascontiguousarray): the latter turns a 0-dim result
        # into shape (1,), which then cannot be copied into a 0-dim tensor
        self.copy_(torch.from_numpy(np.asarray(src.raw).copy()))
        return self
    raise TypeError("vlab::copy_ needs at least one vlab tensor")
