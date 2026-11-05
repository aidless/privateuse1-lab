"""The three non-op hook points PyTorch requires of a PrivateUse1 backend.

`_setup_privateuseone_for_python_backend` registers exactly three things besides
the device rename:

  1. a backend module  -> `torch._register_device_module(name, module)`
  2. a PrivateUse1Hooks object -> `torch._C._acc.register_python_privateuseone_hook`
  3. a DeviceGuard     -> `torch._C._acc.register_python_privateuseone_device_guard`

PyTorch ships dummy versions of all three. The dummy backend module reports a
single device. This lab supplies its own three objects. Storage stays numpy.
"""

from __future__ import annotations

import torch

from . import _state
from .tensor import VlabTensor, _numpy_dtype

#: every hook call is recorded here, so tests can assert the hook was *reached*
#: instead of merely registered (a registered hook that never fires proves nothing)
CALLS: list[tuple[str, tuple]] = []


def _record(name: str, *args) -> None:
    CALLS.append((name, args))


class VlabBackendModule:
    """What `torch.vlab.*` forwards to (device lifecycle + memory introspection)."""

    def is_available(self) -> bool:
        _record("backend.is_available")
        return True

    def is_initialized(self) -> bool:
        _record("backend.is_initialized")
        return True

    def device_count(self) -> int:
        _record("backend.device_count")
        return _state.DEVICE_COUNT

    def current_device(self) -> int:
        _record("backend.current_device")
        return _state.current_device()

    def set_device(self, device) -> None:
        _record("backend.set_device", device)
        _state.set_device(device)

    def manual_seed_all(self, seed: int) -> None:
        _record("backend.manual_seed_all", seed)
        _state.manual_seed_all(seed)

    def _is_in_bad_fork(self) -> bool:
        _record("backend._is_in_bad_fork")
        return False

    # vendor-flavoured extras: a real plugin exposes its allocator's counters
    def memory_allocated(self, device=None) -> int:
        _record("backend.memory_allocated", device)
        return _state.memory_allocated()

    def max_memory_allocated(self, device=None) -> int:
        _record("backend.max_memory_allocated", device)
        return _state.max_memory_allocated()

    def reset_peak_memory_stats(self, device=None) -> None:
        _record("backend.reset_peak_memory_stats", device)
        _state.reset_peak_memory_stats()


class VlabHook(torch._C._acc.PrivateUse1Hooks):
    """Runtime-capability hook consulted by the dispatcher/lazy-init paths."""

    def is_available(self) -> bool:
        _record("hook.is_available")
        return True

    def is_built(self) -> bool:
        _record("hook.is_built")
        return True

    def has_primary_context(self, device_index) -> bool:
        _record("hook.has_primary_context", device_index)
        # a device index outside the simulated fleet has no context
        return 0 <= int(device_index) < _state.DEVICE_COUNT


class VlabGuard(torch._C._acc.DeviceGuard):
    """Device guard: the object the dispatcher queries for 'which device type'."""

    def type_(self):
        _record("guard.type_")
        return torch._C._autograd.DeviceType.PrivateUse1

    def set_device(self, device) -> None:
        _record("guard.set_device", device)
        _state.set_device(device)

    def exchange_device(self, device):
        _record("guard.exchange_device", device)
        prev = _state.current_device()
        _state.set_device(device)
        return prev

    def device(self) -> int:
        _record("guard.device")
        return _state.current_device()


class VlabGenerator(torch._C._acc.PrivateUse1Hooks):
    """Placeholder kept out of the registration path.

    A vendor backend also implements a Generator (Philox state) so that
    `manual_seed` is reproducible on-device. numpy's Generator can stand in, but
    only if every random op reads it — that is a separate, larger surface than
    the three hooks registered here, and this lab does not claim it.
    """


def to_numpy(t) -> "object":
    """Unwrap any vlab tensor; used by tests and the public helpers."""
    if not isinstance(t, VlabTensor):
        raise TypeError(f"expected a vlab tensor, got {type(t).__name__}")
    return t.raw


__all__ = [
    "CALLS",
    "VlabBackendModule",
    "VlabGuard",
    "VlabHook",
    "_numpy_dtype",
    "to_numpy",
]
