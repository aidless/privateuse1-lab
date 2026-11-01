# privateuse1-lab

[![ci](https://github.com/aidless/privateuse1-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/privateuse1-lab/actions/workflows/ci.yml)

**A hardware-free PyTorch `PrivateUse1` backend, built to answer one question: what does "port PyTorch to a new accelerator" actually require?**

Every vendor plugin — `torch_npu` (Ascend), `torch_musa` (Moore Threads), `torch_mlu` (Cambricon), and any domestic GPGPU effort — implements the same backend contract. That contract is small enough to exercise on numpy storage, with no GPU, no driver and no vendor SDK: registration, three hook objects, and a `privateuseone` op set. This repo implements all of it and pins each piece with a test that can fail.

```python
import torch, vlab
vlab.register()
a = torch.ones(2, 3, device="vlab")        # a tensor on the fake device
b = (a @ torch.ones(3, 4, device="vlab"))  # dispatched vlab kernels
q = torch.randn(8, 16).to("vlab")
torch.softmax(torch.mm(q, q.transpose(-1, -2)) * 0.25, dim=-1).to("cpu")
```

37 tests, ~0.2 s, plus a mutation gate.

## Quick start

```bash
pip install -e .                  # installs the `vlab` package (src layout)
pytest tests/ -q                  # 37 conformance tests
python tests/mutation_gate.py     # 4-broken-kernel injection gate
```

`pip install -e .` is what CI runs (CPU-only torch wheel); alternatively
`PYTHONPATH=src pytest tests/ -q` works without installing. Requires
`torch>=2.6` and `numpy>=1.24`.

## The contract, and what each part costs

`torch/utils/backend_registration.py:542` fixes the order: **rename → generate methods → device module → hook → device guard.** Getting it wrong is silent — `torch.<name>.*` just stays missing.

| Piece | What it is | Where it lives here | Reached by PyTorch? |
|---|---|---|---|
| `rename_privateuse1_backend(name)` | the *device* name (`vlab`), not the dispatch key | `vlab/__init__.py:register` | — |
| `BackendModule` | device count, current/set device, seeds, memory counters | `vlab/device.py:VlabBackendModule` | ✅ via every `torch.vlab.*` call (measured) |
| `PrivateUse1Hooks` | `is_available` / `is_built` / `has_primary_context` | `vlab/device.py:VlabHook` | ❌ **not reached from any python-level path** (measured, see below) |
| `DeviceGuard` | answers "which device type am I" during dispatch | `vlab/device.py:VlabGuard` | ✅ on `.to(device)` (measured) |
| op set | `aten::*` kernels registered under `privateuseone` | `vlab/ops.py` (24 ops) | ✅ via dispatch |

### The reachability finding

Registration succeeding proves nothing about whether a hook is ever *called*. `tests/test_hooks.py` records real calls (`vlab.CALLS`) and asserts what actually happens on torch 2.14.1:

```
.to(device)            -> guard.type_
torch.vlab.is_available / device_count / set_device / manual_seed_all -> backend.*
torch.empty / elementwise ops / create_empty_tensor -> (nothing)
PrivateUse1Hooks.*     -> never reached from python
```

So: a backend can ship a hook object whose every method returns garbage and no Python-level test would notice. That is exactly the class of bug that only shows up as a runtime failure deep inside C++ lazy-initialisation paths. The test for it is written as a *documented gap* — it fails if a future torch starts calling the hooks, forcing the table above to be corrected instead of quietly rotting.

The device *guard*, by contrast, is real: `.to("vlab")` reaches it, and `guard.type_()` returning `PrivateUse1` is what makes the transfer land on our kernels.

## Op set: small, but closed

24 kernels: factories (`empty.memory_format`, `empty_strided`, `zeros`, `ones`, `randn`), movement (`_copy_from`, `copy_`, `detach`, `clone`), views (`view`, `_unsafe_view`, `as_strided`, `expand`, `transpose.int`, `t`), elementwise (`add`/`sub`/`mul`/`div` .Tensor+.Scalar, `neg`, `exp`, `relu`, `pow` ×3), reductions (`sum`, `sum.dim_IntList`, `max.dim`), and `softmax.int`, `mm`, `matmul`, `bmm`.

"Small but closed" means the set is exactly what a **scaled-dot-product attention block** needs, so `tests/test_attention_e2e.py` runs that block end-to-end on the fake device and compares against CPU torch — including a 2×3×24×16 multi-head batch. A hole in the op set fails that test rather than a user's model.

Three things this lab learned by being run, all of which are ordinary vendor-plugin bugs:

1. **Wrapped python scalars land on the `Tensor` overload.** `t + 2.5` dispatches `aten::add.Tensor` with `b` a bare `float`, not `aten::add.Scalar`. A kernel that calls `unwrap(b)` raises `TypeError: unsupported operand type(s) for +: 'VlabTensor' and 'float'` — the operator silently refuses the device. `vlab/ops.py:_val` accepts tensor, 0-dim CPU tensor and plain number.
2. **Views must alias, not copy.** `np.ascontiguousarray` inside the tensor wrapper turns `expand`/`view`/`transpose` results into independent copies, so writes through a view disappear. After the fix, writing through `view` is visible in the base (tested), and writing through an `expand`ed view raises — matching PyTorch, which refuses in-place writes on overlapping memory.
3. **0-dim results break host copies.** `sum(x)` returns a 0-dim tensor; `torch.from_numpy(np.ascontiguousarray(arr))` turns the 0-dim array into shape `(1,)` and the copy fails with `output with shape [] doesn't match the broadcast shape [1]`.

## The mutation gate

`tests/mutation_gate.py` replaces `vlab.kernels.*` with four broken variants and requires the oracle (CPU torch) to catch each one. It also prints which case caught it, because a mutation visible in only one case tells you where a test suite is blind:

```
control (no bug): all 4 cases match CPU oracle -> PASSED
SOFTMAX_NO_MAXSHIFT      -> CAUGHT by ['softmax_large_logits']  (NOT visible in ['mm', 'softmax', 'sum_dim'])
SOFTMAX_WRONG_AXIS       -> CAUGHT by ['softmax', 'softmax_large_logits']  (NOT visible in ['mm', 'sum_dim'])
MM_TRANSPOSED_OPERAND    -> CAUGHT by ['mm']
SUM_WRONG_AXIS           -> CAUGHT by ['sum_dim']
```

The first line is the one worth reading. Softmax is shift-invariant, so **no value comparison at ordinary logits can see a missing max-shift** — the mutation survives a plain `allclose(softmax(x), cpu_softmax(x))` at any sane input scale, and only separates at `exp`-overflow scale (logits ≈ 200 > float32's 88). A softmax test that checks values and stops there is decorative.

## What this does *not* claim

- **No autograd.** PyTorch's own pure-python backend test skips `test_backend_simple` (autograd + `backward()`) with *"Disabled due to CI failures; see pytorch/pytorch#190232"*. The autograd path for a python backend is not upstream-guaranteed, so this lab does not assert it. A real plugin (torch_npu/torch_musa) implements autograd through C++ fallback kernels instead.
- **No `faketensor`/`torch.compile` support.** `torch.library.opcheck` meta-kernel tests are out of scope; Dynamo tracing on this backend is untested. PyTorch's own test disables Dynamo around its assertions for the same reason.
- **No generator (Philox) object.** `manual_seed_all` reseeds numpy's legacy RNG so `torch.randn(device="vlab")` is reproducible (tested), but that is not a device generator and does not reduce/advance like one.
- **Not a performance claim.** numpy storage means every "kernel" is a host call; this repo measures conformance, not throughput.
- **Not the C++ path.** A production backend registers kernels via `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` in C++ and builds a `torch` stub library. The registration *duties* are identical; the mechanisms differ. `docs/vendor-hooks.md` maps each duty to a file:line in the real vendor plugins.

## Layout

```
src/vlab/__init__.py    registration in the mandated order (idempotent)
src/vlab/device.py      BackendModule + PrivateUse1Hooks + DeviceGuard, all call-recorded
src/vlab/tensor.py      numpy-backed tensor (create_empty_tensor + __class__ swap)
src/vlab/kernels.py     pure numpy math, no torch — the mutation targets
src/vlab/ops.py         24 aten kernels, thin wrappers over kernels
tests/test_registration.py  rename vs dispatch key, generated methods, missing-kernel loudness
tests/test_ops.py           numeric parity vs CPU, view aliasing, copy directions
tests/test_attention_e2e.py attention block on-device vs CPU
tests/test_hooks.py         which hooks are actually reached (measured)
tests/mutation_gate.py      4 injections, records which case catches them
```

## Reference for a real port

`docs/vendor-hooks.md` maps each duty above to where a shipping plugin does it — `torch_npu/_init/registry/backend.py:47`, `torch_musa/__init__.py:43`, `torch_mlu/__init__.py:48` all call `rename_privateuse1_backend` and keep their own `BackendModule`; `torch_npu` additionally monkey-patches through `patch_manager.py`, and `torch_mlu` generates stubs from `mlu_functions.yaml`. The interesting comparison is not the naming but *which duties each one leaves to C++*.

Companions: [`aidless/kv-block-mass`](https://github.com/aidless/kv-block-mass) (a Triton kernel for KV-eviction scoring on CUDA **and** ROCm) and [`aidless/paged-kv`](https://github.com/aidless/paged-kv) (C++ PagedAttention block manager).

MIT.
