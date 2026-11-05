# privateuse1-lab

PyTorch's Python entry for a new device is the PrivateUse1 registration helper: rename the device, generate methods, pass a backend module, a hooks object, and a device guard, then register `aten` kernels.
This repo does that on numpy arrays, so the same steps run on a CPU with no GPU and no vendor SDK.
It is a small teaching tree for that Python contract, with tests for a few ways the contract fails.

There is no PyPI release and no status badge. The workflow file is [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Fifteen-minute CPU path

The first install downloads a CPU torch wheel and needs a network. After that, pytest and the mutation gate do not use the network, a GPU, or a vendor SDK. On a fast link the download is the long step; the tests themselves finish in seconds. Python 3.10 or 3.12 matches CI.

From a fresh clone, `src/` is not on the import path, so `import vlab` raises `ModuleNotFoundError` until you install it or set `PYTHONPATH`.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
pip install pytest numpy
pip install -e .
pytest tests/ -q
python tests/mutation_gate.py
```

CI runs these five commands after `python -m pip install --upgrade pip`, on Python 3.10 and Python 3.12. Use `python3 -m pip` and `python3 -m pytest` when the console scripts are not on `PATH`. A passing mutation gate prints `MUTATION GATE PASSED` and exits 0.

`mutation_gate.py` is not named `test_*.py`, so pytest does not collect it. CI runs it in the next step. The workflow then runs one more snippet: import `vlab`, assert `_REGISTERED` is false, call `register()` twice, and build a `vlab` tensor. Local pytest does not run that snippet.

If you skip the editable install:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 tests/mutation_gate.py
```

Torch, numpy, and pytest still have to be installed. `VLAB_DEVICE_COUNT` is read when `vlab` is imported (default `2`). Set it before the import: `VLAB_DEVICE_COUNT=4 PYTHONPATH=src python3 -c "import vlab; print(vlab.device_count())"`.

Then read [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md). That is the list of contract breaks the suite catches, and the holes it does not.

## What pytest collects, and what CI adds

| | `pytest tests/` locally | CI |
|---|---|---|
| Files | every `tests/test_*.py` | the same, on CPython 3.10 and 3.12 |
| `tests/mutation_gate.py` | not collected | `python tests/mutation_gate.py` |
| Import must not register | not run | `python -c` checks `vlab._REGISTERED` |
| Torch | whatever already satisfies `torch>=2.6` | unpinned `torch>=2.6` from `https://download.pytorch.org/whl/cpu` |
| GPU | not used | not used |

The CPU index is only in the install step. `pip install -e .` keeps that torch when it already satisfies `>=2.6`. A later pip that upgrades torch from PyPI can replace it with a different build. CI does not pin the patch version. Hook reachability is whatever the installed torch does, and `tests/test_hooks.py` fails if the recorded calls change.

## Counts checked by pytest

`tests/test_readme_counts.py` recomputes these three lines and fails if the README drifts. It does not rewrite the file. The numbers are not a speed or a torch-version claim.

```
Collected pytest tests: 48
Registered aten impls: 39
Mutation injections: 4
```

`Registered aten impls` is the number of `@torch.library.impl` decorators in `src/vlab/ops.py`. Recount with `grep -c '@torch.library.impl' src/vlab/ops.py`.

## What this does not prove

- No hardware. Storage is a numpy array. Nothing here shows a driver, an allocator, or a copy to a device.
- No C++ registration (`TORCH_LIBRARY_IMPL`) and no torch extension library.
- No autograd. Upstream's Python backend test skips `test_backend_simple` and cites [pytorch/pytorch#190232](https://github.com/pytorch/pytorch/issues/190232). This repo does not register a backward.
- No `torch.compile`, no faketensor meta kernels, and no `torch.testing._internal.optests.opcheck`.
- No device generator. `manual_seed_all` stores an integer. `randn` reseeds numpy from that integer on every call, so two draws in a row match. That is not Philox state advancing.
- Not every registered op is checked. The attention tests cover matmul, scale, softmax, and matmul. `empty_strided` ignores `stride`, `max.dim` does not apply `keepdim` to indices, and `pow.Tensor_Scalar_out` ignores `out`. Details are in [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md).
- Gemm is not bit-exact. `mm` / `bmm` / attention comparisons use `rtol=1e-4, atol=1e-5` because numpy and CPU torch float32 matmul differ in the last few ulps.
- `PrivateUse1Hooks` can return the wrong answer and the Python paths in `tests/test_hooks.py` still pass. A direct call checks the object. A C++ lazy-init path is not run.
- [`docs/vendor-hooks.md`](docs/vendor-hooks.md) records a file fetch from 2026-11-05. CI does not re-fetch those repos. This tree does not track their versions or star counts.
- Wall-clock time is not a claim. Do not treat a local timing as part of the contract.

## When registration is wrong

Full table: [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md). The short version, from tests:

- No `register()`: `torch.empty(..., device="vlab")` raises `RuntimeError` and names `privateuseone`.
- Device module registered before the rename: `RuntimeError`, and that call does not create `torch.vlab`.
- `generate_methods_for_privateuse1_backend()` before the rename: no exception, `Tensor.is_privateuseone` exists, `torch.vlab` does not.
- Setup with torch's dummy module and no kernels: `torch.vlab.device_count()` is `1`, and `empty` raises `NotImplementedError` (`Could not run`, `empty.memory_format`).
- Guard `type_()` returns CPU while kernels exist: `.to("vlab")` stays on `cpu` and does not raise.
- Hooks object returns false while kernels exist: `empty` and `.to("vlab")` still yield `vlab` tensors, and the hook methods are not called.
- Missing op (`torch.sort`): `NotImplementedError` containing `Could not run` and `vlab`, not a silent CPU result.

`_setup_privateuseone_for_python_backend` performs rename, then method generation, then the module, the hook, and the guard. The rows above are the mis-orders and substitutions the tests run. They are not every possible order.

## What the numerical tests cover

`tests/test_ops.py` compares a fixed elementwise list, sum, softmax, copy, view aliasing, clone versus detach, and a reseeded `randn` to CPU torch. `tests/test_attention_e2e.py` runs scaled-dot-product attention, including a `2×3×24×16` batch, through dispatch.

Three kernel constraints the tests are aimed at:

1. Wrapped Python scalars hit the Tensor overload. `t + 2.5` dispatches `aten::add.Tensor` with a float. `_val` accepts a vlab tensor, a CPU tensor, or a plain number.
2. `view` aliases the base buffer (`test_writable_views_alias_the_base_tensor`). An in-place write through `expand` raises (`test_write_through_expanded_view_is_rejected_not_silently_applied`).
3. A 0-dim `sum` is copied back with `np.asarray`. `np.ascontiguousarray` on a 0-dim array has shape `(1,)`, which does not copy into a 0-dim tensor. `test_sum_matches_cpu_including_axis_and_keepdim` covers that host copy.

`tests/mutation_gate.py` patches four functions in `vlab.kernels` and requires CPU torch to disagree, or the call to raise. The control run with the real kernels must agree. One injection drops the softmax max-shift. Softmax is shift-invariant, so ordinary logits can still match; the gate includes a case with logits `200`. The script prints which case caught which injection. That caught-by list is output, not a second count frozen here.

## Layout

```
src/vlab/__init__.py        register(), explicit and idempotent
src/vlab/device.py          backend module, PrivateUse1Hooks, device guard
src/vlab/tensor.py          numpy array on a PrivateUse1 tensor
src/vlab/kernels.py         numpy math the mutation gate patches
src/vlab/ops.py             aten impls (count checked above)
src/vlab/_state.py          simulated device count, seed, byte counter
tests/test_registration.py  rename, generated methods, missing-kernel error
tests/test_ops.py           parity, views, copies
tests/test_attention_e2e.py attention block versus CPU torch
tests/test_hooks.py         which registered objects are called
tests/test_failure_modes.py contract breaks in fresh interpreters
tests/test_readme_counts.py README counts and CI command lines
tests/mutation_gate.py      four kernel injections; not collected by pytest
docs/FAILURE_MODES.md       symptoms and known holes
docs/vendor-hooks.md        dated file fetch of three vendor rename sites
```

MIT.
