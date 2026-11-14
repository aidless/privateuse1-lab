#!/usr/bin/env python3
"""eval/calibrate_scorer.py — 给本仓唯一的“评委”（CPU torch oracle）做校准测量。

为什么需要这一份：本仓**没有 LLM 评委**，唯一的打分器是 CPU torch 的确定性数值
oracle（`tests/mutation_gate.py` 用它判 mutate 是否被抓）。拿模型打分的项目要报
一致率/κ 是因为模型会漂；本仓要报的是**同一批评委在换阈值时判定会不会翻**——
即「判定对容差有多敏感」的实测，而不是口头声明。

测法（全部现算，无任何手抄数字）：
  * 项集 = `tests/mutation_gate.py` 的 5 个变体（control + 4 个注入）× 4 个 case
    = n 项；每项现算 `max|Δ|`（device 结果 vs CPU torch，float64 提升后取绝对值最大差）。
  * 参照判定 D_ref = 该 gate 自己用的判定：shape 相等 且 `np.allclose(rtol=1e-5, atol=1e-6)`；
    不等/抛异常记 True（=与 CPU 不一致）。
  * 待校准判定 D_abs(thr) = `max|Δ| > thr`；扫 thr，逐档算 TP/FP/TN/FN、
    一致率、Cohen's κ（对 D_ref）。
  * 关键量：control（无 bug）上的假阳数 —— 它给出 float32 噪声底，说明 thr 的下界
    不是拍出来的，而是量出来的。

用法：
    python3 eval/calibrate_scorer.py            # 人类可读 + exit code
    python3 eval/calibrate_scorer.py --json     # 机器可读
    python3 eval/calibrate_scorer.py --save     # 写 eval/calibration-report.json
    python3 eval/calibrate_scorer.py --selftest # 用一个已知过紧的阈值，必须报假阳（exit 3）

退出码：0 = 默认阈值 κ≥0.90 且 control 零假阳；1 = 校准失败；2 = 环境问题；
       3 = --selftest 按预期变红（过紧阈值出现假阳）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import mutation_gate as mg  # noqa: E402 — 只读复用它的 case/变体定义，不修改
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] 无法导入 tests/mutation_gate.py：{exc}", file=sys.stderr)
    raise SystemExit(2)

from unittest import mock  # noqa: E402

from vlab import kernels  # noqa: E402

# 阈值扫描档位：跨越 float32 噪声底（≈1e-7）上下两侧
THRESHOLDS = (1e-9, 1e-6, 1e-5, 1e-3)
DEFAULT_THR = 1e-6


def _max_delta(fn) -> float:
    """现算 device 结果与 CPU torch 的最大绝对差；nan/inf 或抛异常一律记 inf（=明显不一致）。"""
    try:
        got, want = fn()
    except Exception:  # noqa: BLE001 — 抛异常在 gate 语义里也算被抓
        return math.inf
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return math.inf
    d = np.abs(got - want)
    if np.isnan(d).any() or np.isinf(d).any():
        return math.inf
    return float(d.max())


def _ref_verdict(fn) -> bool:
    """D_ref：与 tests/mutation_gate.py 完全相同的判定（shape + allclose）。"""
    try:
        got, want = fn()
    except Exception:  # noqa: BLE001
        return True
    return not (
        got.shape == want.shape
        and np.allclose(got, want, rtol=1e-5, atol=1e-6)
    )


def collect() -> list[dict]:
    vlab = sys.modules["vlab"]
    vlab.register()
    cases = mg._cases()
    items: list[dict] = []

    def push(variant: str, broken: bool):
        for label, fn in cases.items():
            items.append({
                "variant": variant,
                "case": label,
                # broken=True 表示“这个变体在该 case 上原则上可能被看出来”，
                # 但并非每个 case 都该看出（例如 softmax 的 max-shift 在普通 logits 不可见），
                # 所以真值列只用于分组统计，不参与 κ 计算。
                "variant_is_broken": broken,
                "max_abs_delta": _max_delta(fn),
                "ref_differs": _ref_verdict(fn),
            })

    push("CONTROL", False)
    for name, (attr, impl) in mg.MUTATIONS.items():
        with mock.patch.object(kernels, attr, impl):
            push(name, True)
    return items


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def sweep(items: list[dict]) -> list[dict]:
    ref = [it["ref_differs"] for it in items]
    out = []
    for thr in THRESHOLDS:
        pred = [it["max_abs_delta"] > thr for it in items]
        tp = sum(p and r for p, r in zip(pred, ref))
        fp = sum(p and not r for p, r in zip(pred, ref))
        fn = sum((not p) and r for p, r in zip(pred, ref))
        tn = sum((not p) and (not r) for p, r in zip(pred, ref))
        ctrl_fp = sum(
            p for p, it in zip(pred, items) if it["variant"] == "CONTROL")
        n_ctrl = sum(1 for it in items if it["variant"] == "CONTROL")
        out.append({
            "threshold": thr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "agreement": (tp + tn) / len(items),
            "cohen_kappa": cohen_kappa(pred, ref),
            "control_false_positives": ctrl_fp,
            "control_n": n_ctrl,
        })
    return out


def summarize(items: list[dict], rows: list[dict]) -> dict:
    ctrl = [it["max_abs_delta"] for it in items if it["variant"] == "CONTROL"]
    row = next(r for r in rows if r["threshold"] == DEFAULT_THR)
    return {
        "n_items": len(items),
        "n_variants": len({it["variant"] for it in items}),
        "n_cases": len({it["case"] for it in items}),
        "noise_floor_max_abs_delta_control": max(ctrl),
        "noise_floor_median_abs_delta_control": float(np.median(ctrl)),
        "default_threshold": DEFAULT_THR,
        "sweep": rows,
        "default_agreement": row["agreement"],
        "default_cohen_kappa": row["cohen_kappa"],
        "default_control_false_positives": row["control_false_positives"],
        "disagreements_at_default": [
            {"variant": it["variant"], "case": it["case"],
             "max_abs_delta": it["max_abs_delta"], "ref_differs": it["ref_differs"]}
            for it in items
            if (it["max_abs_delta"] > DEFAULT_THR) != it["ref_differs"]
        ],
        "blind_spots": [
            {"variant": it["variant"], "case": it["case"],
             "max_abs_delta": it["max_abs_delta"]}
            for it in items
            if it["variant_is_broken"] and it["max_abs_delta"] <= DEFAULT_THR
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="用 1e-12 这个已知过紧的阈值跑一遍：必须出现 control 假阳")
    args = ap.parse_args()

    items = collect()
    rows = sweep(items)
    if args.selftest:
        # 反证：门必须能红。过紧阈值在 control 上必然假阳（float32 噪声底 > 1e-12）。
        bad = [dict(r) for r in rows]
        thr = 1e-12
        ref = [it["ref_differs"] for it in items]
        pred = [it["max_abs_delta"] > thr for it in items]
        ctrl_fp = sum(p for p, it in zip(pred, items) if it["variant"] == "CONTROL")
        n_ctrl = sum(1 for it in items if it["variant"] == "CONTROL")
        report = {"selftest": True, "threshold": thr,
                  "control_false_positives": ctrl_fp, "control_n": n_ctrl,
                  "agreement": sum(p == r for p, r in zip(pred, ref)) / len(items),
                  "cohen_kappa": cohen_kappa(pred, ref)}
        print(json.dumps(report, ensure_ascii=False, indent=1) if args.json
              else f"SELFTEST thr={thr:g}: control 假阳 {ctrl_fp}/{n_ctrl}，"
                   f"一致率 {report['agreement']:.3f}，κ={report['cohen_kappa']:.3f}"
                   "\n（预期：假阳 > 0 且 κ 下降 —— 收紧阈值会开始误报，说明门是有效而非装饰）")
        return 3 if ctrl_fp > 0 else 1

    rep = summarize(items, rows)
    rep["n_items_detail"] = items

    if args.save:
        out = REPO_ROOT / "eval" / "calibration-report.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[saved] {out.relative_to(REPO_ROOT)}", file=sys.stderr)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        print(f"项集：{rep['n_variants']} 变体 × {rep['n_cases']} case = n = {rep['n_items']}")
        print(f"control（无 bug）噪声底：max|Δ| = {rep['noise_floor_max_abs_delta_control']:.3e} "
              f"（中位 {rep['noise_floor_median_abs_delta_control']:.3e}）")
        print(f"{'thr':>10s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s} "
              f"{'agree':>7s} {'kappa':>7s} {'ctrl_FP':>8s}")
        for r in rows:
            print(f"{r['threshold']:10.0e} {r['tp']:3d} {r['fp']:3d} {r['fn']:3d} {r['tn']:3d} "
                  f"{r['agreement']:7.3f} {r['cohen_kappa']:7.3f} "
                  f"{r['control_false_positives']}/{r['control_n']:<5d}")
        print(f"\n默认阈值 {DEFAULT_THR:g}：一致率 {rep['default_agreement']:.3f}，"
              f"κ = {rep['default_cohen_kappa']:.3f}，"
              f"control 假阳 {rep['default_control_false_positives']}/{rep['n_cases']}")
        if rep["blind_spots"]:
            print("盲区（注入在该 case 上数值不可见，属设计内）：")
            for b in rep["blind_spots"]:
                print(f"  {b['variant']:24s} {b['case']:22s} max|Δ|={b['max_abs_delta']:.3e}")
        if rep["disagreements_at_default"]:
            print("默认阈值下与 D_ref 分歧：")
            for d in rep["disagreements_at_default"]:
                print(f"  {d['variant']} / {d['case']}: max|Δ|={d['max_abs_delta']:.3e} "
                      f"ref_differs={d['ref_differs']}")

    ok = (rep["default_control_false_positives"] == 0
          and rep["default_cohen_kappa"] >= 0.90)
    if not ok:
        print("CALIBRATION FAILED：默认阈值零假阳且 κ≥0.90 不成立", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
