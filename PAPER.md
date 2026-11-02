# privateuse1-lab：无硬件 PyTorch PrivateUse1 Backend 契约的完整实现与验证

> 本文档是 privateuse1-lab 的**论文摘要草稿**，内容全部来自仓库真实实现与真实测试
> 结果，无虚构数字。

## 摘要（Abstract）

把 PyTorch 移植到新加速器需要实现一个 vendor plugin。所有国产 GPGPU 厂商
（昇腾 `torch_npu`、摩尔线程 `torch_musa`、寒武纪 `torch_mlu`）实现的是**同一个**
backend 契约：注册、三个 hook 对象、以及 `privateuseone` 算子集。本文用**无硬件**
方式把该契约完整实现一遍——在 numpy 存储上，无 GPU、无驱动、无 vendor SDK——并
逐件钉死（每件配一个能失败的测试）。

**关键发现（可复现）**：注册成功**不证明** hook 真的被调用。PyTorch 的
`torch.utils.backend_registration.py:542` 固定了顺序（rename → generate methods →
device module → hook → device guard），实测 `PrivateUse1Hooks.*` 在 torch 2.14.1 上
**从未被任何 Python 级路径触达**——一个返回垃圾的 hook 对象不会被任何 Python 测试
发现。这正是只能在 C++ 惰性初始化路径深处以运行时失败呈现的那类 bug。该测试被写
成**文档化缺口**：未来 torch 开始调用 hook 时它会失败，强制纠正表格而非悄悄腐烂。

**第二个发现**：算子集"小而封闭"——恰好覆盖 scaled-dot-product attention 块，
因此端到端测试（2×3×24×16 multi-head batch）在算子集有洞时失败，而不是用户的模型。

**第三个发现**（突变门量化）：softmax 是平移不变算子，普通 logits 值比较**看不见**
缺失的 max-shift——只有 `softmax_large_logits`（logits ≈ 200 > float32 的 88）能分离。
一个只做值比较就停的 softmax 测试是装饰。

**评测（可复算）**：37 条一致性测试全绿（~0.2s）；突变门 4 个坏内核注入全被 CPU
oracle 抓住，控制组在全部 4 case 上匹配。

**结论**：本仓回答"移植到新加速器实际上需要什么"——答案是**比注册成功少得多**：
registration 证明不了 hook 被调用，而真正被触达的是 device guard（`.to(device)`）
与 dispatch 路径。这不是性能工作（"Not a performance claim"）。

## 关键词

PyTorch backend；PrivateUse1；加速器移植；vendor plugin；一致性测试；
突变测试；GPGPU；国产算力

## 与真实代码的对应

| 摘要声明 | 对应真实实现 |
|---|---|
| 契约在 numpy 上实现 | `src/vlab/`（无 GPU、无驱动） |
| 固定注册顺序 | `src/vlab/__init__.py:register` |
| hook 未被触达（实测） | `tests/test_hooks.py` 记录 `vlab.CALLS` |
| 24 算子集 | `src/vlab/ops.py` |
| attention 端到端 | `tests/test_attention_e2e.py` |
| 突变门 4 注入 | `tests/mutation_gate.py`（实测 PASSED） |
| 37 条测试 | `pytest tests/ -q` 实测 37 passed |
| vendor 职责映射 | `docs/vendor-hooks.md`（torch_npu/torch_musa/torch_mlu） |
