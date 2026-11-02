# PREREG — privateuse1-lab 预注册规格

> 本文件是 privateuse1-lab 的**预注册规格**：把研究问题、评测方案、成功标准与
> 明确边界写在分析之前，防止事后按结果挑边界。所有数字可从仓库确定性代码复算。

## 1. 研究问题

把 PyTorch 移植到一个新加速器，**实际上需要实现什么**？这是每一个国产 GPGPU 厂商
（昇腾 `torch_npu`、摩尔线程 `torch_musa`、寒武纪 `torch_mlu`）都要回答的 vendor
plugin 契约问题。本仓用**无硬件**方式回答它：在 numpy 存储上把整个
`PrivateUse1` backend 契约完整实现一遍，并**逐件钉死**（每件配一个能失败的测试）。

**核心假设（H1）**：`PrivateUse1` backend 契约足够小，可完全在 numpy 存储上、
无 GPU/无驱动/无 vendor SDK 的情况下实现并验证（37 条一致性测试）。

**辅助假设（H2，关键发现）**：注册成功**不证明** hook 真的被调用。实测
`torch.utils.backend_registration` 的固定顺序（rename → generate methods → device
module → hook → device guard）里，`PrivateUse1Hooks.*` 在 torch 2.14.1 上**从未被
任何 Python 级路径触达**——一个返回垃圾的 hook 对象不会被任何 Python 测试发现。

**辅助假设（H3）**：算子集"小而封闭"——恰好覆盖 scaled-dot-product attention 块，
因此算子集有洞会让端到端测试失败，而不是用户的模型。

## 2. 评测方案（可复算）

### 2.1 一致性测试（37 条）

```bash
pip install -e .            # 或 PYTHONPATH=src
pytest tests/ -q            # 37 passed, ~0.2s
```

覆盖：registration（rename vs dispatch key、生成的 methods、缺失 kernel 响亮度）、
ops（数值 parity vs CPU、view 别名、拷贝方向）、attention 端到端（2×3×24×16
multi-head batch，vs CPU torch）、hooks（哪些 hook 真的被触达，实测）。

### 2.2 突变门（mutation gate）

```bash
python tests/mutation_gate.py    # 4 个坏内核注入，oracle（CPU torch）必须各抓一个
```

**成功标准**：控制组（无 bug）在全部 4 case 上与 CPU oracle 一致 → PASSED；
每个注入的坏内核**至少被一个 case 抓到** → 突变门通过。**反证**：softmax 是平移
不变算子，普通 logits 值比较**看不见**缺失的 max-shift——只有 `softmax_large_logits`
（logits ≈ 200 > float32 的 88）能分离。一个只做值比较就停的 softmax 测试是装饰。

### 2.3 复算命令

```bash
PYTHONPATH=src python3 -m pytest tests/ -q        # 37 passed
PYTHONPATH=src python3 tests/mutation_gate.py     # MUTATION GATE PASSED
```

## 3. 明确边界（不证明什么）

本仓**不证明**以下内容——真实方法边界，非隐藏缺陷：

1. **无 autograd**：PyTorch 自身对纯 Python backend 跳过 `test_backend_simple`
   （autograd + `backward()`，"Disabled due to CI failures; see pytorch/pytorch#190232"）。
   真实插件（torch_npu/torch_musa）用 C++ fallback kernel 实现 autograd。
2. **无 `faketensor`/`torch.compile` 支持**：`torch.library.opcheck` meta-kernel 测试
   与 Dynamo 追踪超出范围；PyTorch 自身也为此禁用 Dynamo。
3. **无 generator（Philox）对象**：`manual_seed_all` 只 reseed numpy 的 legacy RNG，
   不是设备 generator，不 reduce/advance。
4. **不是性能声明**：numpy 存储意味着每个"kernel"都是宿主调用；本仓测**一致性**，
   不测吞吐。
5. **不是 C++ 路径**：生产 backend 用 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`；
   注册**职责**相同，**机制**不同。`docs/vendor-hooks.md` 把每个职责映射到真实
   vendor plugin 的 file:line。

## 4. 成功标准（判据）

本预注册声明**成功**当且仅当：
1. `pytest tests/ -q` 全绿（37 passed）；
2. `tests/mutation_gate.py` 报告 `MUTATION GATE PASSED`（4 注入全抓 + 控制组匹配）；
3. 每一条"不证明"边界都在 README 或本文件有对应记录（不静默）。
