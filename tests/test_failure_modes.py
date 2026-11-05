"""What breaks when the PrivateUse1 Python registration contract is wrong.

Registration is process-global, so each case that calls it uses a fresh
interpreter. The symptoms asserted here are the ones described in
``docs/FAILURE_MODES.md``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _py(code: str) -> str:
    env = os.environ.copy()
    src = str(ROOT / "src")
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not prev else src + os.pathsep + prev
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise AssertionError(combined)
    return proc.stdout


def test_unregistered_device_name_is_rejected():
    out = _py(
        """
import torch
try:
    torch.empty(1, device="vlab")
    print("EMPTY_OK")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
"""
    )
    assert "EMPTY_OK" not in out
    assert "RuntimeError" in out
    assert "vlab" in out
    assert "privateuseone" in out


def test_registering_the_module_before_rename_raises():
    out = _py(
        """
import torch
class Module:
    def device_count(self):
        return 2
try:
    torch._register_device_module("vlab", Module())
    print("REGISTERED")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
"""
    )
    assert "REGISTERED" not in out
    assert "RuntimeError" in out
    assert "vlab" in out


def test_generating_methods_before_rename_leaves_torch_vlab_missing():
    out = _py(
        """
import torch
torch.utils.generate_methods_for_privateuse1_backend()
torch.utils.rename_privateuse1_backend("vlab")
print("HAS_VLAB", hasattr(torch, "vlab"))
print("IS_PRIVATEUSEONE", hasattr(torch.Tensor, "is_privateuseone"))
print("IS_VLAB", hasattr(torch.Tensor, "is_vlab"))
"""
    )
    assert "HAS_VLAB False" in out
    assert "IS_PRIVATEUSEONE True" in out
    assert "IS_VLAB False" in out


def test_setup_without_kernels_uses_the_dummy_module_and_refuses_empty():
    out = _py(
        """
import torch
torch.utils.backend_registration._setup_privateuseone_for_python_backend("vlab")
print("COUNT", torch.vlab.device_count())
try:
    torch.empty(2, device="vlab")
    print("EMPTY_OK")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
"""
    )
    assert "COUNT 1" in out
    assert "EMPTY_OK" not in out
    assert "NotImplementedError" in out
    assert "Could not run" in out
    assert "empty.memory_format" in out


def test_guard_that_reports_cpu_makes_to_vlab_stay_on_cpu():
    out = _py(
        """
import torch
import vlab.ops  # kernels exist; the guard is the thing under test
class Guard(torch._C._acc.DeviceGuard):
    def type_(self):
        return torch._C._autograd.DeviceType.CPU
torch.utils.backend_registration._setup_privateuseone_for_python_backend(
    "vlab", device_guard=Guard()
)
print(torch.ones(2).to("vlab").device.type)
"""
    )
    assert out.strip().splitlines()[-1] == "cpu"


def test_hook_that_returns_false_is_not_consulted_from_python():
    out = _py(
        """
import torch
import vlab.ops
called = []
class Hook(torch._C._acc.PrivateUse1Hooks):
    def is_available(self):
        called.append("is_available")
        return False
    def is_built(self):
        called.append("is_built")
        return False
    def has_primary_context(self, device_index):
        called.append("has_primary_context")
        return False
class Guard(torch._C._acc.DeviceGuard):
    def type_(self):
        return torch._C._autograd.DeviceType.PrivateUse1
torch.utils.backend_registration._setup_privateuseone_for_python_backend(
    "vlab", hook=Hook(), device_guard=Guard()
)
empty = torch.empty(2, device="vlab")
moved = torch.ones(2).to("vlab")
print("CALLED", called)
print("EMPTY", empty.device.type)
print("MOVED", moved.device.type)
"""
    )
    assert "CALLED []" in out
    assert "EMPTY vlab" in out
    assert "MOVED vlab" in out


def test_device_count_env_is_read_at_import():
    out = _py(
        """
import os
os.environ["VLAB_DEVICE_COUNT"] = "4"
import vlab
print(vlab.device_count())
"""
    )
    assert out.strip() == "4"


def test_device_count_env_is_ignored_after_import():
    out = _py(
        """
import os
os.environ.pop("VLAB_DEVICE_COUNT", None)
import vlab
os.environ["VLAB_DEVICE_COUNT"] = "4"
print(vlab.device_count())
"""
    )
    assert out.strip() == "2"
