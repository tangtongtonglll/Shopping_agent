#!/usr/bin/env python3
"""
evaluate_agent_perf.py
======================
LangGraph ShoppingAgent 整体编排性能评测。

直接调用项目的节点函数（router_node / retriever_node /
comparator_node / generator_node），精确记录每个节点耗时。

测试指标
--------
1. 路由准确率 (Routing Accuracy)
   router_node 识别的 intent 与 expected_intent 匹配比例。
   意图映射规则:
       CHITCHAT       → "chat"
       PRODUCT_SEARCH → "search"
       FUZZY_SEARCH   → "search"  (都是搜索意图)
       COMPARISON     → "compare"

2. 节点耗时 (Node Latency Profiling)
   使用 time.perf_counter()，记录每个节点平均 / 中位数 / P95 耗时。
   节点: router_node / retriever_node / comparator_node / generator_node

3. 端到端延迟 (End-to-End Latency)
   router_node 开始 → generator_node 结束的完整时间。

4. 性能瓶颈
   自动识别耗时最长的节点并给出优化建议。

运行:
    conda run -n Shopping_agent python tests/evaluation/evaluate_agent_perf.py
"""

import asyncio
import contextlib
import io
import json
import sys
import os
import time
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from app.core.database import SessionLocal
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.WARNING)

# ── 导入项目节点和状态 ────────────────────────────────────────────────────────
from app.graph.nodes import (
    router_node,
    retriever_node,
    comparator_node,
    generator_node,
)
from app.graph.graph import _route_from_generator
from app.graph.state import DEFAULT_STATE

# ── 意图映射：eval 数据集意图 → LangGraph router 输出 ─────────────────────────
INTENT_MAP = {
    "CHITCHAT":       "chat",
    "PRODUCT_SEARCH": "search",
    "FUZZY_SEARCH":   "search",
    "COMPARISON":     "compare",
}

ALL_NODES = ["router_node", "retriever_node", "comparator_node", "generator_node"]


# ══════════════════════════════════════════════════════════════════════════════
# 单次查询执行（带节点计时）
# ══════════════════════════════════════════════════════════════════════════════

async def _run_node_quiet(fn, state, config) -> Any:
    """执行节点函数，抑制其内部 print() 输出（减少批量评测时的噪音）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return await fn(state, config)


async def run_single_case(query: str, db) -> Dict[str, Any]:
    """
    手动按拓扑顺序执行各节点，精确记录每个节点的独立耗时。

    手动执行等价于 graph.ainvoke()，但绕过 LangGraph 内部调度，
    从而可以在每个节点前后精确插入计时点。

    Returns
    -------
    {
        "intent":       str,     # router 识别的意图
        "confidence":   float,   # generator 输出的置信度
        "needs_review": bool,    # 是否会触发 human_review_node
        "n_docs":       int,     # retriever 检索到的文档数
        "timings": {
            "router_node":     float,        # 秒
            "retriever_node":  float | None, # None = 未执行 (chat intent)
            "comparator_node": float | None, # None = 未执行 (非 compare)
            "generator_node":  float,        # 秒
            "e2e":             float,        # 秒（全链路）
        },
        "error": str | None,
    }
    """
    # 初始化 state（TypedDict，使用 dict 拷贝）
    state: Dict[str, Any] = {
        "messages":        [HumanMessage(content=query)],
        "intent":          DEFAULT_STATE["intent"],
        "entities":        dict(DEFAULT_STATE["entities"]),
        "retrieved_docs":  list(DEFAULT_STATE["retrieved_docs"]),
        "comparison_data": DEFAULT_STATE["comparison_data"],
        "final_response":  DEFAULT_STATE["final_response"],
        "confidence_score": DEFAULT_STATE["confidence_score"],
    }
    config = {
        "configurable": {
            "db":                 db,
            "knowledge_base_ids": None,
            "thread_id":          "eval_perf",
        }
    }

    timings: Dict[str, Optional[float]] = {n: None for n in ALL_NODES}
    timings["e2e"] = None
    error: Optional[str] = None

    t_start = time.perf_counter()

    try:
        # ── 1. router_node ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        update = await _run_node_quiet(router_node, state, config)
        timings["router_node"] = time.perf_counter() - t0
        state.update(update)

        intent = state.get("intent", "chat")

        # ── 2. retriever_node (search / compare) ─────────────────────────────
        if intent in ("search", "compare"):
            t0 = time.perf_counter()
            update = await _run_node_quiet(retriever_node, state, config)
            timings["retriever_node"] = time.perf_counter() - t0
            state.update(update)

        # ── 3. comparator_node (compare only) ────────────────────────────────
        if intent == "compare":
            t0 = time.perf_counter()
            update = await _run_node_quiet(comparator_node, state, config)
            timings["comparator_node"] = time.perf_counter() - t0
            state.update(update)

        # ── 4. generator_node ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        update = await _run_node_quiet(generator_node, state, config)
        timings["generator_node"] = time.perf_counter() - t0

        # 手动处理 add_messages reducer（追加，不覆盖）
        new_msgs = update.pop("messages", [])
        state.update(update)
        state["messages"] = state["messages"] + new_msgs

    except Exception as exc:
        error = str(exc)

    timings["e2e"] = time.perf_counter() - t_start

    # ── 是否触发 human_review（不实际执行，仅判断路由条件）─────────────────────
    needs_review = False
    if error is None:
        try:
            needs_review = _route_from_generator(state) == "human_review_node"
        except Exception:
            pass

    return {
        "intent":       state.get("intent", "chat"),
        "confidence":   state.get("confidence_score", 0.0),
        "needs_review": needs_review,
        "n_docs":       len(state.get("retrieved_docs", [])),
        "timings":      timings,
        "error":        error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 统计工具
# ══════════════════════════════════════════════════════════════════════════════

def _p95(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(vals, 95))


def _fmt_ms(secs: Optional[float]) -> str:
    if secs is None:
        return "  N/A  "
    if secs >= 1.0:
        return f"{secs:6.2f}s"
    return f"{secs*1000:5.0f}ms"


def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# 终端汇总表
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results, accuracy, by_intent_c, by_intent_t, node_times):
    print()
    print("═" * 72)
    print("  LangGraph ShoppingAgent 性能评测结果")
    print("═" * 72)

    # ── 路由准确率 ──────────────────────────────────────────────────────────
    total  = len(results)
    n_ok   = sum(1 for r in results if r["is_correct"])
    n_err  = sum(1 for r in results if r["error"])

    print(f"\n  ── 路由准确率 ──")
    print(f"  总体: {n_ok}/{total} = {_pct(accuracy)}  (其中 {n_err} 条运行异常)")
    print()
    print(f"  {'意图':<18} {'正确':<6} {'总数':<6} {'准确率':>8}")
    print("  " + "-" * 42)
    for intent in ["CHITCHAT", "PRODUCT_SEARCH", "FUZZY_SEARCH", "COMPARISON"]:
        nc = by_intent_c.get(intent, 0)
        nt = by_intent_t.get(intent, 0)
        acc = nc / nt if nt else 0.0
        print(f"  {intent:<18} {nc:<6} {nt:<6} {_pct(acc):>8}")

    # 路由混淆矩阵（仅展示错误案例的 actual intent）
    wrong = [r for r in results if not r["is_correct"] and not r["error"]]
    if wrong:
        print(f"\n  ⚠️  误分类样本 ({len(wrong)} 条):")
        for r in wrong:
            print(f"    [{r['id']}] 期望={r['expected_lg']:<8} 实际={r['actual_intent']:<8}  {r['query'][:45]}…")

    # ── 节点耗时 ────────────────────────────────────────────────────────────
    print(f"\n  ── 节点耗时 (N=执行次数) ──")
    print(f"  {'节点':<22} {'N':>4} {'均值':>9} {'中位数':>9} {'P95':>9} {'最大':>9}  E2E占比")
    print("  " + "-" * 72)

    e2e_vals = node_times.get("e2e", [])
    e2e_sum  = sum(e2e_vals) or 1.0

    for node in ALL_NODES:
        vals = node_times.get(node, [])
        if not vals:
            print(f"  {node:<22} {'0':>4}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>7}")
            continue
        avg    = statistics.mean(vals)
        med    = statistics.median(vals)
        p95    = _p95(vals)
        mx     = max(vals)
        share  = sum(vals) / e2e_sum * 100
        print(
            f"  {node:<22} {len(vals):>4}"
            f"  {_fmt_ms(avg):>8}"
            f"  {_fmt_ms(med):>8}"
            f"  {_fmt_ms(p95):>8}"
            f"  {_fmt_ms(mx):>8}"
            f"  {share:5.1f}%"
        )

    # ── 端到端延迟 ──────────────────────────────────────────────────────────
    e2e_vals = [r["timings"]["e2e"] for r in results if r["timings"]["e2e"] is not None]
    if e2e_vals:
        print(f"\n  ── 端到端延迟 ──")
        print(f"  均值={_fmt_ms(statistics.mean(e2e_vals))}  "
              f"中位数={_fmt_ms(statistics.median(e2e_vals))}  "
              f"P95={_fmt_ms(_p95(e2e_vals))}  "
              f"最快={_fmt_ms(min(e2e_vals))}  "
              f"最慢={_fmt_ms(max(e2e_vals))}")

        # 按 intent 分组的 E2E
        print()
        print(f"  {'意图':<18} {'N':>4} {'均值':>9} {'中位数':>9}")
        print("  " + "-" * 46)
        for intent in ["CHITCHAT", "PRODUCT_SEARCH", "FUZZY_SEARCH", "COMPARISON"]:
            grp = [r["timings"]["e2e"] for r in results
                   if r["expected_raw"] == intent and r["timings"]["e2e"] is not None]
            if grp:
                print(f"  {intent:<18} {len(grp):>4}"
                      f"  {_fmt_ms(statistics.mean(grp)):>8}"
                      f"  {_fmt_ms(statistics.median(grp)):>8}")

    # ── 瓶颈识别 ────────────────────────────────────────────────────────────
    print(f"\n  ── 性能瓶颈分析 ──")
    node_avgs = {n: statistics.mean(t) for n, t in node_times.items()
                 if n != "e2e" and t}
    if node_avgs:
        bottleneck = max(node_avgs, key=lambda k: node_avgs[k])
        print(f"  🔴 主要瓶颈 : {bottleneck}（平均 {_fmt_ms(node_avgs[bottleneck])}）")
        ranked = sorted(node_avgs.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, t) in enumerate(ranked, 1):
            bar = "█" * int(t / max(node_avgs.values()) * 20)
            print(f"  {rank}. {n:<22} {_fmt_ms(t)}  {bar}")

    n_review = sum(1 for r in results if r.get("needs_review"))
    print(f"\n  human_review_node 触发次数: {n_review}/{total} "
          f"（置信度 < 0.6 或高价值商品）")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Markdown 报告
# ══════════════════════════════════════════════════════════════════════════════

def build_markdown(results, accuracy, by_intent_c, by_intent_t, node_times) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    n_ok  = sum(1 for r in results if r["is_correct"])
    n_err = sum(1 for r in results if r["error"])

    e2e_vals = [r["timings"]["e2e"] for r in results if r["timings"]["e2e"] is not None]

    lines = [
        "# LangGraph ShoppingAgent 性能评测报告",
        "",
        f"> 生成时间：{ts}",
        "",
        "## 实验设置",
        "",
        "| 项目 | 值 |",
        "|------|----|",
        f"| 数据集 | `eval_dataset.json`（{total} 条用例） |",
        "| 节点调用方式 | 直接调用节点函数（bypassing `graph.ainvoke`），精确计时 |",
        "| 意图映射 | CHITCHAT→chat / PRODUCT_SEARCH→search / FUZZY_SEARCH→search / COMPARISON→compare |",
        f"| 运行异常 | {n_err} 条 |",
        "",
        "## 一、路由准确率",
        "",
        f"**总体准确率：{n_ok}/{total} = {accuracy*100:.1f}%**",
        "",
        "| 意图 | 正确 | 总数 | 准确率 |",
        "|------|-----:|-----:|------:|",
    ]

    for intent in ["CHITCHAT", "PRODUCT_SEARCH", "FUZZY_SEARCH", "COMPARISON"]:
        nc  = by_intent_c.get(intent, 0)
        nt  = by_intent_t.get(intent, 0)
        acc = nc / nt if nt else 0.0
        lines.append(f"| {intent} | {nc} | {nt} | {_pct(acc)} |")

    # 误分类明细
    wrong = [r for r in results if not r["is_correct"] and not r["error"]]
    if wrong:
        lines += [
            "",
            "### 误分类明细",
            "",
            "| # | 期望意图 | 实际意图 | 查询 |",
            "|---|---------|---------|------|",
        ]
        for r in wrong:
            lines.append(
                f"| {r['id']} | {r['expected_lg']} | {r['actual_intent']} "
                f"| {r['query'][:50]}… |"
            )

    # ── 节点耗时 ──────────────────────────────────────────────────────────────
    lines += [
        "",
        "## 二、节点耗时",
        "",
        "| 节点 | N | 均值 | 中位数 | P95 | 最大值 | E2E占比 |",
        "|------|--:|-----:|------:|----:|------:|-------:|",
    ]

    e2e_total = sum(node_times.get("e2e", [])) or 1.0
    for node in ALL_NODES:
        vals = node_times.get(node, [])
        if not vals:
            lines.append(f"| {node} | 0 | — | — | — | — | — |")
            continue
        avg   = statistics.mean(vals)
        med   = statistics.median(vals)
        p95   = _p95(vals)
        mx    = max(vals)
        share = sum(vals) / e2e_total * 100
        lines.append(
            f"| `{node}` | {len(vals)} "
            f"| {avg:.2f}s | {med:.2f}s | {p95:.2f}s | {mx:.2f}s "
            f"| {share:.1f}% |"
        )

    # ── 端到端延迟 ────────────────────────────────────────────────────────────
    lines += ["", "## 三、端到端延迟", ""]
    if e2e_vals:
        lines += [
            f"- **均值**：{statistics.mean(e2e_vals):.2f}s",
            f"- **中位数**：{statistics.median(e2e_vals):.2f}s",
            f"- **P95**：{_p95(e2e_vals):.2f}s",
            f"- **最快**：{min(e2e_vals):.2f}s",
            f"- **最慢**：{max(e2e_vals):.2f}s",
            "",
            "### 按意图分组",
            "",
            "| 意图 | N | 均值 | 中位数 |",
            "|------|--:|-----:|------:|",
        ]
        for intent in ["CHITCHAT", "PRODUCT_SEARCH", "FUZZY_SEARCH", "COMPARISON"]:
            grp = [r["timings"]["e2e"] for r in results
                   if r["expected_raw"] == intent and r["timings"]["e2e"] is not None]
            if grp:
                lines.append(
                    f"| {intent} | {len(grp)} "
                    f"| {statistics.mean(grp):.2f}s "
                    f"| {statistics.median(grp):.2f}s |"
                )

    # ── 瓶颈分析 ──────────────────────────────────────────────────────────────
    node_avgs = {n: statistics.mean(t) for n, t in node_times.items()
                 if n != "e2e" and t}
    bottleneck = max(node_avgs, key=lambda k: node_avgs[k]) if node_avgs else "—"
    n_review   = sum(1 for r in results if r.get("needs_review"))

    lines += [
        "",
        "## 四、性能瓶颈分析",
        "",
        f"**主要瓶颈：`{bottleneck}`**（平均 {node_avgs.get(bottleneck, 0):.2f}s）",
        "",
        "| 排名 | 节点 | 平均耗时 |",
        "|------|------|--------:|",
    ]
    for rank, (n, t) in enumerate(
        sorted(node_avgs.items(), key=lambda x: x[1], reverse=True), 1
    ):
        lines.append(f"| {rank} | `{n}` | {t:.2f}s |")

    lines += [
        "",
        f"**human_review_node 触发**：{n_review}/{total} 条",
        "（触发条件：`confidence_score < 0.6` 或高价值商品关键词）",
        "",
        "## 五、优化建议",
        "",
    ]

    # 自动生成建议
    if node_avgs:
        sorted_nodes = sorted(node_avgs.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, t) in enumerate(sorted_nodes, 1):
            if n == "generator_node":
                lines.append(
                    f"{rank}. **generator_node**（平均 {t:.2f}s）是主要瓶颈，"
                    "建议：流式返回（streaming）减少感知延迟；"
                    "对 CHITCHAT 使用轻量模型（如 GLM-4-Flash）；"
                    "缓存高频问题的回复。"
                )
            elif n == "router_node":
                lines.append(
                    f"{rank}. **router_node**（平均 {t:.2f}s）每次都调用 LLM，"
                    "建议：使用规则/正则快速路径预过滤明显的 CHITCHAT；"
                    "或切换为本地轻量分类模型（如 fastText）。"
                )
            elif n == "comparator_node":
                lines.append(
                    f"{rank}. **comparator_node**（平均 {t:.2f}s）仅对 compare 意图触发，"
                    "建议：预生成常见商品对比数据缓存；"
                    "简化 prompt 减少 token 数。"
                )
            elif n == "retriever_node":
                lines.append(
                    f"{rank}. **retriever_node**（平均 {t:.2f}s）当前语料库空（无 DocumentChunk），"
                    "建议：将商品 products 表写入 DocumentChunk，"
                    "启用全量检索能力；Reranker 可批量化以减少逐对打分开销。"
                )

    lines += [
        "",
        "---",
        f"*由 `evaluate_agent_perf.py` 自动生成 @ {ts}*",
        "*调用服务：app/graph/nodes.py 节点函数*",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    eval_path   = Path(__file__).parent / "eval_dataset.json"
    report_path = Path(__file__).parent / "agent_perf_report.md"

    if not eval_path.exists():
        print(f"❌ eval_dataset.json not found at {eval_path}")
        sys.exit(1)

    with open(eval_path, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    db = SessionLocal()

    print("═" * 72)
    print(f"  LangGraph Agent 性能评测  ({len(cases)} 条用例)")
    print("═" * 72)
    print(f"  {'#':<4} {'意图':<18} {'路由结果':<10} {'✓':<3} {'E2E':>8}  查询")
    print("  " + "─" * 68)

    results = []

    for i, case in enumerate(cases, 1):
        expected_raw = case.get("expected_intent", "CHITCHAT")
        expected_lg  = INTENT_MAP.get(expected_raw, "chat")
        query        = case["query"]

        try:
            result       = await run_single_case(query, db)
            actual       = result["intent"]
            is_correct   = actual == expected_lg
            mark         = "✅" if is_correct else "❌"
            e2e_s        = result["timings"]["e2e"] or 0.0
            err_note     = f" ⚠️{result['error'][:40]}" if result["error"] else ""

            results.append({
                "id":           case["id"],
                "expected_raw": expected_raw,
                "expected_lg":  expected_lg,
                "actual_intent": actual,
                "is_correct":   is_correct,
                "timings":      result["timings"],
                "confidence":   result["confidence"],
                "needs_review": result["needs_review"],
                "n_docs":       result["n_docs"],
                "query":        query,
                "error":        result["error"],
            })

            print(
                f"  {i:<4} {expected_raw:<18} {actual:<10} {mark:<3}"
                f" {e2e_s:>6.1f}s  {query[:35]}…{err_note}"
            )

        except Exception as exc:
            results.append({
                "id":           case["id"],
                "expected_raw": expected_raw,
                "expected_lg":  expected_lg,
                "actual_intent": "ERROR",
                "is_correct":   False,
                "timings":      {n: None for n in ALL_NODES + ["e2e"]},
                "confidence":   0.0,
                "needs_review": False,
                "n_docs":       0,
                "query":        query,
                "error":        str(exc),
            })
            print(f"  {i:<4} {expected_raw:<18} ERROR      ❌     —      {query[:35]}… [{exc}]")

    db.close()

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    n_ok        = sum(1 for r in results if r["is_correct"])
    accuracy    = n_ok / len(results) if results else 0.0

    by_intent_c: Dict[str, int] = defaultdict(int)
    by_intent_t: Dict[str, int] = defaultdict(int)
    for r in results:
        by_intent_t[r["expected_raw"]] += 1
        if r["is_correct"]:
            by_intent_c[r["expected_raw"]] += 1

    node_times: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        for node, t in r["timings"].items():
            if t is not None:
                node_times[node].append(t)

    # ── 输出汇总 ─────────────────────────────────────────────────────────────
    print_summary(results, accuracy, by_intent_c, by_intent_t, node_times)

    # ── 保存报告 ─────────────────────────────────────────────────────────────
    md = build_markdown(results, accuracy, by_intent_c, by_intent_t, node_times)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ 报告已保存 → {report_path}")
    print(f"   文件大小: {report_path.stat().st_size / 1024:.1f} KB\n")


if __name__ == "__main__":
    asyncio.run(main())
