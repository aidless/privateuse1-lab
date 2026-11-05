"""Scaled-dot-product attention on the vlab device, compared with CPU torch.

The block is matmul, scale, softmax, matmul, all through dispatched vlab kernels.
Agreement is ``rtol=1e-4, atol=1e-5`` because the gemm is numpy, not libtorch.
A wrong kernel or a missing op in this chain fails the test. Ops outside the
chain are not covered here.
"""

import numpy as np
import torch

import vlab


def _attention_cpu(q, k, v, scale):
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return probs @ v, probs


def _attention_device(q, k, v, scale):
    scores = torch.mm(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.mm(probs, v), probs


def test_softmax_attention_block_matches_cpu():
    torch.manual_seed(0)
    for shape in [(8, 16), (48, 32)]:  # (N, D)
        N, D = shape
        q_cpu, k_cpu, v_cpu = (torch.randn(N, D) for _ in range(3))
        scale = 1.0 / (D ** 0.5)
        out_cpu, probs_cpu = _attention_cpu(q_cpu, k_cpu, v_cpu, scale)
        out_dev, probs_dev = _attention_device(
            q_cpu.to("vlab"), k_cpu.to("vlab"), v_cpu.to("vlab"), scale
        )
        got_probs = vlab.to_numpy(probs_dev)
        got_out = vlab.to_numpy(out_dev)
        assert np.allclose(got_probs, probs_cpu.numpy(), rtol=1e-4, atol=1e-5)
        assert np.allclose(got_out, out_cpu.numpy(), rtol=1e-4, atol=1e-5)
        # attention rows are distributions — a shape-correct but unnormalised
        # kernel would still pass an allclose against itself, not against CPU
        assert np.allclose(got_probs.sum(axis=-1), 1.0, rtol=1e-5)


def test_batched_multi_head_attention_matches_cpu():
    torch.manual_seed(1)
    B, H, N, D = 2, 3, 24, 16
    q_cpu, k_cpu, v_cpu = (torch.randn(B, H, N, D) for _ in range(3))
    scale = 1.0 / (D ** 0.5)
    out_cpu, _ = _attention_cpu(q_cpu, k_cpu, v_cpu, scale)

    q_d, k_d, v_d = (t.to("vlab") for t in (q_cpu, k_cpu, v_cpu))
    scores = torch.matmul(q_d, k_d.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    out_d = torch.matmul(probs, v_d)
    assert np.allclose(vlab.to_numpy(out_d), out_cpu.numpy(), rtol=1e-4, atol=1e-5)


def test_value_only_check_is_not_enough_for_softmax(tmp_path):
    """Why the numerical-stability property is checked separately.

    Softmax(x) and softmax(x + c) are identical, so an implementation that skips
    the max-shift still matches CPU on small logits and only differs when exp()
    overflows. This test pins that: the *unshifted* formula overflows float32 at
    logits ~200 while the kernel's output stays finite and correct.
    """
    big = np.array([[200.0, 199.0, 198.0]], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        unshifted = np.exp(big) / np.exp(big).sum()
    assert not np.isfinite(unshifted).all(), "float32 exp(200) should overflow"

    dev = torch.from_numpy(big).to("vlab")
    got = vlab.to_numpy(torch.softmax(dev, dim=-1))
    assert np.isfinite(got).all()
    assert np.allclose(got, torch.softmax(torch.from_numpy(big), -1).numpy(), rtol=1e-6)
