# When the PrivateUse1 Python contract breaks

`tests/test_failure_modes.py` starts a fresh interpreter for each registration
case, because `rename_privateuse1_backend` can run once per process. The
symptoms below are what that file asserts. CI installs `torch>=2.6` from the
CPU wheel index and does not pin a patch version, so a future torch can turn a
green test red. Update this page in the same change.

`tests/test_hooks.py` records calls on the objects this repo registers. That is
a separate measurement from the fresh-interpreter cases.

## Registration

| Break | What happens | Locked by |
|---|---|---|
| `device="vlab"` before any registration | `RuntimeError`. The message lists `privateuseone` among known device types and does not accept `vlab`. | `test_unregistered_device_name_is_rejected` |
| `torch._register_device_module("vlab", ...)` before `rename_privateuse1_backend` | `RuntimeError` on the register call. `torch.vlab` is not created by that call. | `test_registering_the_module_before_rename_raises` |
| `generate_methods_for_privateuse1_backend()` before the rename | No exception. `torch.Tensor.is_privateuseone` exists. `torch.vlab` does not, and `torch.Tensor.is_vlab` does not. | `test_generating_methods_before_rename_leaves_torch_vlab_missing` |
| `_setup_privateuseone_for_python_backend("vlab")` with no module and no kernels | `torch.vlab.device_count()` is `1` (torch's dummy module). `torch.empty(..., device="vlab")` raises `NotImplementedError` containing `Could not run` and `empty.memory_format`. | `test_setup_without_kernels_uses_the_dummy_module_and_refuses_empty` |
| Kernels registered, device guard `type_()` returns `CPU` | `tensor.to("vlab")` returns a tensor whose `device.type` is `cpu`. No exception. | `test_guard_that_reports_cpu_makes_to_vlab_stay_on_cpu` |
| Kernels registered, `PrivateUse1Hooks` methods return `False` | `torch.empty` and `.to("vlab")` still produce `vlab` tensors. The hook methods are not called. | `test_hook_that_returns_false_is_not_consulted_from_python` |
| A kernel is missing (`torch.sort` on a vlab tensor) | `NotImplementedError` containing `Could not run` and `vlab`. The call does not return a CPU tensor. | `test_unimplemented_op_fails_loudly_instead_of_silently_using_cpu` |

The helper `_setup_privateuseone_for_python_backend` itself calls rename, then
`generate_methods_for_privateuse1_backend`, then device-module registration,
then the hook, then the guard. Its docstring in the installed torch marks that
order as significant. The two rows above are the mis-orders this repo actually
ran; they are not a claim about every other interleaving.

On the torch used to write these tests, after a successful rename,
`torch.library.impl` accepts the key strings `privateuseone`, `PrivateUse1`,
and the renamed device name as one dispatcher key. Before the rename, the
string `vlab` is rejected. The in-repo kernels are registered as
`privateuseone`. The dump table still prints the key as `PrivateUse1`
(`test_device_name_is_renamed_not_privateuseone`).

## Device count

`VLAB_DEVICE_COUNT` is read when `vlab` is imported (`src/vlab/_state.py`).
Setting it to `4` before import makes `vlab.device_count()` return `4`. Setting
it after import leaves the already-read value, which is `2` when the variable
was unset. Both are subprocess tests in `test_failure_modes.py`.

## Hooks the Python surface does not call

With this repo's own objects, `.to("vlab")` records `guard.type_`.
`torch.vlab.is_available`, `device_count`, and `manual_seed_all` record the
backend module. `torch.empty`, an elementwise op, and `set_device` do not
record `hook.*`. `test_privateuse1_hooks_object_is_not_reached_from_python`
fails if one of those paths starts calling the hook, which is the signal to
edit the reachability section of the README.

Calling `VlabHook` directly checks `has_primary_context` for indices inside and
outside the simulated fleet. That is a direct call, not a dispatcher call.

## Holes in this tree

These are behaviors of the current code. No test requires them to stay wrong.
Changing `src/vlab/ops.py` is the time to re-check this list.

- `aten::empty_strided` ignores `stride` and allocates a contiguous buffer of `size`.
- `aten::max.dim` applies `keepdim` to the values and not to the indices.
- `aten::pow.Tensor_Scalar_out` ignores `out` and returns a new tensor.
- `aten::randn` calls `np.random.seed` with the stored seed on every invocation, so a second `randn` without another `manual_seed_all` repeats the first draw. `test_random_is_reproducible_through_the_backend_seed` checks the reseed, not that the generator advances.
- `memory_allocated` only increases. `record_free` is never called, including when a tensor is deleted.
- `.to("cpu")` can return a `VlabTensor` whose `_raw` slot is empty while the CPU storage holds the values. `repr` then prints `VlabTensor(<uninitialised>)`. Numeric tests read the torch storage, not `_raw`.
- `kernels.attention_block_np` is not called by the test suite. The attention tests go through torch dispatch.
- Matmul agreement is `rtol=1e-4, atol=1e-5`. Numpy and CPU torch float32 gemm are not bit-exact; a tighter tolerance failed on a `(1, 4) @ (4, 1)` result while this page was being written.

## What a Python failure will not show

A hook method that raises, or a C++ lazy-init path that consults
`PrivateUse1Hooks`, is outside the calls above. This suite does not build a
C++ extension and does not run `torch.testing._internal.optests.opcheck`.
Autograd is not registered. `torch.compile` is not exercised.
