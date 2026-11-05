"""vlab — numpy storage behind PyTorch's PrivateUse1 Python registration API.

Call ``register()`` before using ``device="vlab"``. Registration is explicit
and idempotent. This package is a CPU-only teaching stand-in, not a hardware
backend.

Usage:
    import vlab
    vlab.register()
    t = torch.empty(2, 3, device="vlab")
"""

from __future__ import annotations

import torch

from . import _state
from .device import CALLS, VlabBackendModule, VlabGuard, VlabHook
from .tensor import VlabTensor, wrap

__all__ = [
    "CALLS",
    "VlabBackendModule",
    "VlabGuard",
    "VlabHook",
    "VlabTensor",
    "register",
    "to_numpy",
    "wrap",
]

_DEVICE_NAME = "vlab"
_REGISTERED = False


def register(name: str = _DEVICE_NAME) -> str:
    """Attach the vlab backend to the PrivateUse1 dispatch key (idempotent).

    ``_setup_privateuseone_for_python_backend`` renames, generates methods, then
    registers the module, hook, and guard. Calling the pieces out of that order
    is not the same failure every time; see docs/FAILURE_MODES.md.
    """
    global _REGISTERED, _DEVICE_NAME
    if _REGISTERED:
        return _DEVICE_NAME

    from . import ops  # noqa: F401  registers the op set as a side effect

    torch.utils.backend_registration._setup_privateuseone_for_python_backend(
        name,
        backend_module=VlabBackendModule(),
        hook=VlabHook(),
        device_guard=VlabGuard(),
    )
    _DEVICE_NAME = name
    _REGISTERED = True
    return name


def to_numpy(t) -> "object":
    """Return the numpy array behind a vlab tensor (copy-free view)."""
    if not isinstance(t, VlabTensor):
        raise TypeError(f"expected a vlab tensor, got {type(t).__name__}")
    return t.raw


def device_count() -> int:
    return _state.DEVICE_COUNT
