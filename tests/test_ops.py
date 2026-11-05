"""Numeric parity: every vlab kernel against the CPU implementation.

The oracle is CPU torch — not a second numpy expression — so a shared mistake in
the lab's numpy code cannot validate itself.
"""

import numpy as np
import pytest
import torch

import vlab


def on_vlab(cpu_t):
    return cpu_t.to("vlab")


def host(t):
    return t.to("cpu").detach()


@pytest.mark.parametrize(
    "name,make",
    [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
        ("add_scalar", lambda a, b: a + 2.5),
        ("mul_scalar", lambda a, b: a * -1.5),
        ("neg", lambda a, b: -a),
        ("exp", lambda a, b: a.exp()),
        ("relu", lambda a, b: a.relu()),
        ("pow", lambda a, b: a.pow(3)),
    ],
)
def test_elementwise_matches_cpu(name, make):
    torch.manual_seed(0)
    a_cpu, b_cpu = torch.randn(3, 4), torch.rand(3, 4) + 0.5
    got = host(make(on_vlab(a_cpu), on_vlab(b_cpu)))
    want = make(a_cpu, b_cpu)
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-7), f"{name} mismatch"


def test_mm_and_bmm_match_cpu():
    torch.manual_seed(1)
    for shape_a, shape_b in [((2, 3), (3, 5)), ((1, 4), (4, 1))]:
        a_cpu, b_cpu = torch.randn(*shape_a), torch.randn(*shape_b)
        # numpy matmul and CPU torch disagree in the last few ulps. This is a
        # tolerance check, not a bit-exact one. rtol=1e-4, atol=1e-5.
        assert torch.allclose(host(torch.mm(on_vlab(a_cpu), on_vlab(b_cpu))),
                              torch.mm(a_cpu, b_cpu), rtol=1e-4, atol=1e-5)
    batched_a, batched_b = torch.randn(3, 4, 5), torch.randn(3, 5, 2)
    assert torch.allclose(host(torch.bmm(on_vlab(batched_a), on_vlab(batched_b))),
                          torch.bmm(batched_a, batched_b), rtol=1e-4, atol=1e-5)


def test_sum_matches_cpu_including_axis_and_keepdim():
    torch.manual_seed(2)
    x_cpu = torch.randn(2, 3, 4)
    for dims, keepdim in [([2], False), ([0, 1], True), ([-1], True)]:
        got = host(torch.sum(on_vlab(x_cpu), dim=dims, keepdim=keepdim))
        want = torch.sum(x_cpu, dim=dims, keepdim=keepdim)
        assert got.shape == want.shape and torch.allclose(got, want, rtol=1e-6)
    assert torch.allclose(host(torch.sum(on_vlab(x_cpu))), torch.sum(x_cpu), rtol=1e-6)


@pytest.mark.parametrize("tau", [1.0, 20.0])
def test_softmax_matches_cpu_and_is_max_shifted(tau):
    """The max-shift is what keeps `exp` inside float32 — check it with a large scale.

    Softmax is invariant to adding a constant, so a *value-only* test cannot see
    whether the shift happened; the second half of the test therefore looks at
    the intermediate that overflows without it.
    """
    torch.manual_seed(3)
    x_cpu = torch.randn(4, 32) * tau
    got = host(torch.softmax(on_vlab(x_cpu), dim=-1))
    want = torch.softmax(x_cpu, dim=-1)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-7)

    # without the shift, exp(200) overflows float32; with it, nothing does
    big = torch.full((1, 4), 200.0)
    shifted = host(torch.softmax(on_vlab(big), dim=-1))
    assert torch.isfinite(shifted).all(), "softmax produced inf/nan on large logits"
    assert torch.allclose(shifted, torch.full((1, 4), 0.25), rtol=1e-6)


def test_copy_directions_roundtrip_exactly():
    torch.manual_seed(4)
    cpu = torch.randn(5, 6, dtype=torch.float32)
    back = on_vlab(cpu).to("cpu")
    assert torch.equal(back.detach(), cpu), "H2D->D2H must be bit-exact"


def test_device_to_device_copy_does_not_alias():
    torch.manual_seed(5)
    src = torch.randn(3, 3)
    a = on_vlab(src)
    b = torch.empty(3, 3, device="vlab")
    b.copy_(a)
    b.fill_(0.0)
    assert not torch.allclose(host(a), torch.zeros(3, 3)), "copy_ then fill_ must not touch the source"


def test_writable_views_alias_the_base_tensor():
    """view: writing through it is visible in the base (it is the same buffer)."""
    base = torch.ones(1, 3, device="vlab")
    flat = base.view(3)
    flat.fill_(2.0)
    assert np.all(vlab.to_numpy(base) == 2.0), "view must alias, not copy"


def test_write_through_expanded_view_is_rejected_not_silently_applied():
    """PyTorch refuses in-place writes through an expanded (overlapping) view.

    Our kernel inherits that from numpy's read-only broadcast view. A write that
    silently landed somewhere else would be worse than an exception: the test
    therefore asserts *raising*, not merely "base unchanged".
    """
    base = torch.ones(1, 3, device="vlab")
    expanded = base.expand(4, 3)
    try:
        expanded.fill_(7.0)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("in-place write through an expanded view must not succeed")
    assert np.all(vlab.to_numpy(base) == 1.0)


def test_clone_and_detach_semantics():
    src = torch.ones(2, 2, device="vlab")
    cloned = src.clone()
    cloned.fill_(9.0)
    assert np.all(vlab.to_numpy(src) == 1.0), "clone must copy"
    detached = src.detach()
    detached.fill_(5.0)
    assert np.all(vlab.to_numpy(src) == 5.0), "detach is a view of the same buffer"


def test_random_is_reproducible_through_the_backend_seed():
    torch.vlab.manual_seed_all(123)
    first = vlab.to_numpy(torch.randn(4, device="vlab")).copy()
    torch.vlab.manual_seed_all(123)
    second = vlab.to_numpy(torch.randn(4, device="vlab")).copy()
    assert np.array_equal(first, second), "manual_seed_all must make randn reproducible"
