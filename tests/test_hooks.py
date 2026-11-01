"""Which of the three registered hooks PyTorch actually calls from python.

Registration requires a backend module, a PrivateUse1Hooks object and a
DeviceGuard. Registration succeeding says nothing about *reachability*: each hook
can be a no-op that never fires. These tests record real calls (`vlab.CALLS`) so
the claim is measured, and they encode the gap found in torch 2.14.1: the device
guard is reached by `.to(device)`, the backend module by `torch.vlab.*`, and the
PrivateUse1Hooks object is **not reached at all** from python-level paths.
"""

import pytest
import torch

import vlab


def calls_during(fn):
    vlab.CALLS.clear()
    fn()
    return [name for name, _ in vlab.CALLS]


def test_device_guard_is_reached_by_a_device_transfer():
    names = calls_during(lambda: torch.randn(2, 2).to("vlab"))
    assert "guard.type_" in names, f"guard never consulted by .to(device): {names}"


def test_backend_module_is_reached_by_generated_device_methods():
    names = calls_during(lambda: torch.vlab.is_available())
    assert "backend.is_available" in names, names
    names = calls_during(lambda: torch.vlab.device_count())
    assert "backend.device_count" in names, names
    names = calls_during(lambda: torch.vlab.manual_seed_all(11))
    assert "backend.manual_seed_all" in names, names


@pytest.mark.parametrize(
    "label,fn",
    [
        ("is_available", lambda: torch.vlab.is_available()),
        ("to(device)", lambda: torch.randn(2, 2).to("vlab")),
        ("empty", lambda: torch.empty(2, 2, device="vlab")),
        ("elementwise", lambda: torch.ones(2, 2, device="vlab") + 1),
        ("set_device", lambda: torch.vlab.set_device(0)),
    ],
)
def test_privateuse1_hooks_object_is_not_reached_from_python(label, fn):
    """Documented gap, asserted so it cannot rot silently.

    If a future torch starts consulting the hooks from these paths, this test
    fails and the README's "unexercised" claim must be updated — which is the
    point: the gap is recorded as a measurement, not a guess.
    """
    names = calls_during(fn)
    assert not [n for n in names if n.startswith("hook.")], (
        f"{label} now reaches PrivateUse1Hooks ({names}); update README's "
        "reachability table instead of deleting this test"
    )


def test_hook_object_conforms_when_called_directly():
    """Direct conformance (not integration): the hook answers like a real device."""
    torch.vlab.set_device(0)
    hook = vlab.VlabHook()
    assert hook.is_available() is True
    assert hook.is_built() is True
    assert hook.has_primary_context(0) is True
    assert hook.has_primary_context(1) is True
    assert hook.has_primary_context(2) is False  # device_count == 2
    assert hook.has_primary_context(-1) is False


def test_guard_type_is_privateuse1():
    assert vlab.VlabGuard().type_() == torch._C._autograd.DeviceType.PrivateUse1
