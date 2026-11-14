# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: ci.yml（conformance suite + mutation gate）
# 原命令 `python3 -m pytest tests/ -q` 本机 rc=2（pytest exit 4）：tests/conftest.py:7
# `import vlab` 失败 —— 包在 src/ 下，未 pip install 时需 PYTHONPATH=src。
# 归一: 补 PYTHONPATH=src；按 ci.yml 取回 mutation gate；再加仓内 eval 题库
# （eval/run_cases.py，20 条 expr/raises/doc 断言，期望值都是真跑出来的）。
.PHONY: eval

eval:
	PYTHONPATH=src python3 -m pytest tests/
	PYTHONPATH=src python3 tests/mutation_gate.py
	python3 eval/run_cases.py
