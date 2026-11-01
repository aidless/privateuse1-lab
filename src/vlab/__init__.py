"""vlab — a hardware-free PyTorch PrivateUse1 backend.

Why this exists: adapting PyTorch to a new accelerator (Ascend NPU, MUSA, MLU,
or any domestic GPGPU) means implementing one backend contract — device rename,
three hook objects, and a `privateuseone` op set. The contract is exercised here
with numpy storage, so the whole checklist runs on a laptop in ~2 seconds, with
no GPU, no driver, and no vendor SDK.

Usage:
    import vlab
    vlab.register()                       # idempotent
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

    Order matters and is fixed by torch/utils/backend_registration.py:542:
    rename -> generate methods -> device module -> hook -> device guard.
    Calling `_setup_privateuseone_for_python_backend` performs all five in that
    order; doing it by hand in another order leaves `torch.vlab.*` unmapped.
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
