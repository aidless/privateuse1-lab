# Vendor plugins, and what this lab does not copy

This page names files in other repositories. It is not checked by CI. The
rename-call check below is a one-time fetch, not a pin.

## Rename call sites fetched on 2026-11-05

Each default-branch file below contained a call to
`torch.utils.rename_privateuse1_backend`. Line numbers are where that call sat
on that fetch. They will move.

| Repo | File fetched | Call at that fetch |
|---|---|---|
| `Ascend/pytorch` (`master`) | `torch_npu/_init/registry/backend.py` | `rename_privateuse1_backend("npu")` at line 47 |
| `MooreThreads/torch_musa` (`main`) | `torch_musa/__init__.py` | `rename_privateuse1_backend("musa")` at line 44 |
| `Cambricon/torch_mlu` (`master`; `main` returned 404) | `torch_mlu/__init__.py` | `rename_privateuse1_backend("mlu")` at line 48 |

Star counts and version-coupling tables were removed from this doc. They were
dated snapshots and this repo does not refresh them.

## Same duty, different mechanism

Those three plugins, and this lab, both rename the PrivateUse1 device and
supply a Python module for device count, current device, and seeding. The lab
then registers `aten` kernels with `torch.library.impl(..., "privateuseone")`
on numpy arrays.

A C++ plugin registers kernels with `TORCH_LIBRARY_IMPL` and ships a torch
extension library. This repository does not do that, and nothing here shows
that a driver or an allocator works.

## Docs that were checked the same day

- `pytorch/tutorials` `advanced_source/privateuseone.rst` on `main` returned
  the tutorial source (HTTP 200 from raw.githubusercontent.com).
- `pytorch/pytorch` `docs/source/notes/privateuse1_backend.rst` on `main`
  returned HTTP 404.
- `pytorch/pytorch` `test/test_privateuseone_python_backend.py` on `main`
  skips `test_backend_simple` and cites
  https://github.com/pytorch/pytorch/issues/190232
  (`test_backend_simple (PrivateUse1BackendTest) fails on main`, open at the
  time of the fetch). This lab does not register autograd.

`torch.testing._internal.optests.opcheck` is the upstream op checker (schema,
autograd, faketensor). This lab does not run it. A meta kernel per op would be
required before that checker could apply, and those kernels are not here.

## What to read next to this repo

1. The tutorial source above, for the C++ shape of the same registration idea.
2. `test/test_privateuseone_python_backend.py` in pytorch, for the Python helper this lab follows.
3. `docs/FAILURE_MODES.md` in this repo, for which Python mistakes this suite actually catches.
