"""Process-wide vlab state: simulated devices and allocation accounting.

A real vendor plugin asks its runtime how many devices exist, which one is
current, and how much memory is live. None of that can be inferred from tensor
attributes, so it lives here — this is the part of a backend that `torch.library`
op registration cannot express, and the reason `torch.vlab.*` methods exist.
"""

from __future__ import annotations

import os
import threading

_lock = threading.Lock()

#: number of simulated devices; overridable so multi-device paths can be tested
DEVICE_COUNT = max(1, int(os.environ.get("VLAB_DEVICE_COUNT", "2")))

_current_device = 0
_seed = 0
_live_bytes = 0
_peak_bytes = 0


def current_device() -> int:
    return _current_device


def set_device(device) -> None:
    global _current_device
    idx = device if isinstance(device, int) else int(str(device).split(":")[-1] or 0)
    if not 0 <= idx < DEVICE_COUNT:
        raise RuntimeError(
            f"vlab: device index {idx} out of range (device_count={DEVICE_COUNT})"
        )
    _current_device = idx


def record_alloc(nbytes: int) -> None:
    global _live_bytes, _peak_bytes
    with _lock:
        _live_bytes += nbytes
        _peak_bytes = max(_peak_bytes, _live_bytes)


def record_free(nbytes: int) -> None:
    global _live_bytes
    with _lock:
        _live_bytes -= nbytes


def memory_allocated() -> int:
    return _live_bytes


def max_memory_allocated() -> int:
    return _peak_bytes


def reset_peak_memory_stats() -> None:
    global _peak_bytes
    with _lock:
        _peak_bytes = _live_bytes


def manual_seed_all(seed: int) -> int:
    global _seed
    _seed = int(seed) & 0xFFFFFFFF
    return _seed


def get_seed() -> int:
    return _seed
