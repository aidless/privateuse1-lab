# Where the real plugins do each of these things

Facts below were read from the repositories (file:line), not from documentation
summaries. Star counts are as measured on 2026-10-31 and are context, not merit.

| Duty in `privateuse1-lab` | torch_npu (Ascend) | torch_musa (Moore Threads) | torch_mlu (Cambricon) |
|---|---|---|---|
| rename the device | `torch_npu/_init/registry/backend.py:47` | `torch_musa/__init__.py:43` | `torch_mlu/__init__.py:48` |
| backend module (count / current / seed) | `torch_npu/npu/__init__.py` + `_init/` registry | `torch_musa/musa/__init__.py` | `torch_mlu/mlu/__init__.py` |
| op registration mechanism | C++ `TORCH_LIBRARY_IMPL` + `patch_manager.py` monkey-patch layer | C++ + a source-level CUDA→MUSA port (`torch_musa/utils/mapping/*.json`) | codegen: `mlu_functions.yaml` → `gen_mlu_stubs.py` |
| version coupling | matrix: TorchNPU 26.2.0 ↔ PyTorch 2.9–2.14 ↔ CANN 9.2.0 | follows upstream torch minor | default branch pinned to `r1.22_pt2.1.0`, no commits since 2025-03 |
| repo / stars | `Ascend/pytorch`, 598★ (GitHub is a mirror; GitCode is the main host) | `MooreThreads/torch_musa`, 519★ | `Cambricon/torch_mlu`, 61★ |

## The three duties this lab can and cannot mirror

**Mirrored (same duties, different mechanism).** All three call
`rename_privateuse1_backend`, all three supply a device module answering
count/current/seed, all three register `aten` kernels under the PrivateUse1
dispatch key. What `vlab` proves is that the *set of duties* is small and
enumerable; what it cannot prove is anything about a real driver.

**Not mirrored: the C++ side.** A production plugin builds a torch stub library
and registers kernels with `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`. The
official starting point is the PrivateUse1 tutorial
(`docs.pytorch.org/tutorials/advanced/privateuseone.html`, source
`pytorch/tutorials:advanced_source/privateuseone.rst`); the C++ extension
plumbing is `pytorch/extension-cpp` (1203★). Note that a commonly cited doc path
`docs/source/notes/privateuse1_backend.rst` **does not exist** (404) — several
blog posts cite it.

**Not mirrored: the acceptance standard.** `torch.testing._internal.optests.opcheck`
is what upstream uses to bless an op implementation (schema, autograd
registration, faketensor, AOT dispatch). This lab skips it: faketensor support
would require a meta kernel per op, which is a genuine, listed gap rather than a
silent one.

## Why the hooks object matters and what to check first

`PrivateUse1Hooks` is registered but never reached from Python (measured —
`tests/test_hooks.py`). In C++ the same hook is consulted on lazy-init paths, so
its methods *are* load-bearing there; the Python mirror just cannot observe it.
Practical consequence for a porting team: do not validate hook behaviour from a
Python test suite — validate it from a C++ integration test or a real workload,
because nothing at the Python level will tell you it is broken.

## Reading order for someone porting a backend

1. `pytorch/tutorials:advanced_source/privateuseone.rst` — the C++ minimal path.
2. `pytorch/test/test_privateuseone_python_backend.py` — the Python mirror used here (173 lines; note its autograd test is skipped upstream, `pytorch/pytorch#190232`).
3. `pytorch/extension-cpp` — stub library + `TORCH_LIBRARY_IMPL` plumbing.
4. `torch_musa` — cleanest small plugin; then its `utils/mapping/*.json` for what a CUDA→other-source port actually touches.
5. `torch_npu` — the version matrix + `patch_manager.py`, i.e. what a mature port accumulates as debt.
6. `torch_mlu/codegen` — YAML→stub generation, i.e. how to avoid maintaining thousands of stubs by hand.
