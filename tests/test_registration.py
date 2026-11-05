"""The backend contract: rename, three hooks, generated methods, op dispatch."""

import torch

import vlab


def test_device_name_is_renamed_not_privateuseone():
    """`rename_privateuse1_backend` renames the *device*, not the dispatch key.

    The string in ``device="vlab"`` is the renamed device. The dispatcher table
    still labels the key ``PrivateUse1``. On current torch, ``torch.library.impl``
    accepts ``privateuseone``, ``PrivateUse1``, and the renamed name as that same
    key after ``rename_privateuse1_backend``. Before the rename, the name ``vlab``
    is rejected. See ``docs/FAILURE_MODES.md``.
    """
    assert torch.vlab is not None
    assert torch.empty(1, device="vlab").device.type == "vlab"
    # the dispatch key keeps its canonical name
    assert torch._C._dispatch_dump_table("aten::empty.memory_format").count("PrivateUse1") >= 1


def test_generated_device_methods_exist():
    """`generate_methods_for_privateuse1_backend` maps torch.vlab.* -> module."""
    for name in (
        "is_available",
        "device_count",
        "current_device",
        "set_device",
        "manual_seed_all",
        "memory_allocated",
        "max_memory_allocated",
    ):
        assert hasattr(torch.vlab, name), f"torch.vlab.{name} missing after registration"


def test_device_count_comes_from_the_backend_module():
    """Default fleet is 2. The torch dummy module reports 1; see FAILURE_MODES.md."""
    assert torch.vlab.device_count() == vlab.device_count() == 2


def test_set_device_is_bounds_checked_and_observable():
    torch.vlab.set_device(1)
    assert torch.vlab.current_device() == 1
    try:
        torch.vlab.set_device(2)
    except RuntimeError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover - the check must exist
        raise AssertionError("set_device(2) must fail with device_count=2")
    torch.vlab.set_device(0)
    assert torch.vlab.current_device() == 0


def test_unimplemented_op_fails_loudly_instead_of_silently_using_cpu():
    """A partial op set is normal; what matters is that the gap is visible.

    There is no CPU fallback registered for this device. A missing kernel raises
    ``NotImplementedError`` whose text starts with ``Could not run`` and names
    the ``vlab`` backend. It must not return a CPU result.
    """
    t = torch.ones(4, device="vlab")
    try:
        torch.sort(t)
    except (RuntimeError, NotImplementedError) as exc:
        message = str(exc)
        assert "Could not run" in message
        assert "vlab" in message
    else:  # pragma: no cover
        raise AssertionError("sort has no vlab kernel; it must not silently succeed")
