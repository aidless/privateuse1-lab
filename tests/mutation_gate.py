"""Mutation gate: inject known-bad kernels and require the oracle to catch them.

Unit tests that pass prove the happy path, not that the tests have teeth. This
gate swaps `vlab.kernels.*` for deliberately broken variants, drives the same
public operations, and fails unless every injection is detected (by a value
mismatch against CPU torch, or by an exception). The control run at the top
proves the gate can also *pass* — a gate that only ever goes red is decoration.

Usage: python tests/mutation_gate.py   (exit 0 = all injections caught)
"""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import torch

import vlab
from vlab import kernels

vlab.register()

ATOL = 1e-6


# Each case is (label, run_on_device, run_on_cpu) closing over the *same* input,
# so a mutation is judged by "does any case separate it from CPU torch".
def _cases():
    torch.manual_seed(0)
    a, b, c = torch.randn(4, 8), torch.randn(8, 6), torch.randn(4, 6)
    big = torch.full((1, 4), 200.0, dtype=torch.float32)

    def case_mm():
        return vlab.to_numpy(torch.mm(a.to("vlab"), b.to("vlab"))), (a @ b).numpy()

    def case_softmax():
        return (
            vlab.to_numpy(torch.softmax(c.to("vlab"), dim=-1)),
            torch.softmax(c, dim=-1).numpy(),
        )

    def case_softmax_large_logits():
        # 200 >> 88: exp() overflows float32 unless the max-shift happens
        return (
            vlab.to_numpy(torch.softmax(big.to("vlab"), dim=-1)),
            torch.softmax(big, dim=-1).numpy(),
        )

    def case_sum_dim():
        return (
            vlab.to_numpy(torch.sum(c.to("vlab"), dim=[-1], keepdim=True)),
            c.sum(dim=-1, keepdim=True).numpy(),
        )

    return {
        "mm": case_mm,
        "softmax": case_softmax,
        "softmax_large_logits": case_softmax_large_logits,
        "sum_dim": case_sum_dim,
    }


def _differs(labels) -> dict[str, bool]:
    """Run every case; True where the device result disagrees with CPU torch."""
    out = {}
    for label in labels:
        fn = _cases()[label]
        try:
            got, want = fn()
        except Exception:
            out[label] = True  # raising counts as caught
            continue
        out[label] = not (
            got.shape == want.shape and np.allclose(got, want, rtol=1e-5, atol=ATOL)
        )
    return out


# --------------------------------------------------------------------- variants


def _softmax_no_maxshift(x, axis):
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)  # no max-shift: silently wrong once logits are large
        return e / e.sum(axis=axis, keepdims=True)


def _softmax_wrong_axis(x, axis):
    shifted = x - x.max(axis=0, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=0, keepdims=True)


def _mm_transposed_operand(a, b):
    return a @ b.T


def _sum_wrong_axis(x, dims, keepdim):
    return x.sum(axis=0, keepdims=bool(keepdim))


MUTATIONS = {
    "SOFTMAX_NO_MAXSHIFT": ("softmax_np", _softmax_no_maxshift),
    "SOFTMAX_WRONG_AXIS": ("softmax_np", _softmax_wrong_axis),
    "MM_TRANSPOSED_OPERAND": ("mm_np", _mm_transposed_operand),
    "SUM_WRONG_AXIS": ("sum_dim_np", _sum_wrong_axis),
}


def main() -> int:
    all_labels = list(_cases())

    # 1. control: unpatched kernels must MATCH on every case (gate can pass)
    control = _differs(all_labels)
    if any(control.values()):
        print(f"CONTROL FAILED (gate is not trustworthy): {control}")
        return 2
    print(f"control (no bug): all {len(all_labels)} cases match CPU oracle -> PASSED")

    # 2. every injection must be caught by at least one case
    failures = []
    for name, (attr, impl) in MUTATIONS.items():
        with mock.patch.object(kernels, attr, impl):
            result = _differs(all_labels)
        caught = [k for k, bad in result.items() if bad]
        missed = [k for k, bad in result.items() if not bad]
        if caught:
            note = f"  (NOT visible in {missed})" if missed else ""
            print(f"{name:24s} -> CAUGHT by {caught}{note}")
        else:
            print(f"{name:24s} -> MISSED by every case")
            failures.append(name)

    if failures:
        print(f"MUTATION GATE FAILED: {len(failures)} injection(s) escaped: {failures}")
        return 1
    print(
        f"MUTATION GATE PASSED: {len(MUTATIONS)} injections caught, control matches "
        f"on all {len(all_labels)} cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
