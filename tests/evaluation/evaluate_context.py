#!/usr/bin/env python3
"""
evaluate_context.py
===================
大海捞针（Needle-in-a-Haystack）测试 —— 量化 Agent 长上下文处理能力。

测试原理
--------
将 N 个智能手机商品的详细参数文本作为 background context 强行注入给大模型，
在其中埋入唯一的"目标商品"（针），其屏幕刷新率值全局唯一，不可被猜测。
随后提问"第 X 个商品的屏幕刷新率是多少？"考察模型能否在长文本中精准定位。

实验设计
--------
- 上下文规模 (haystack size)：10 / 20 / 30 / 40 / 50 个商品
  预计 Token 量：~2K / ~4K / ~6K / ~8K / ~10K
- 每个规模测试 5 个针位：开头(~20%) / 前段(~40%) / 中间(~50%) / 后段(~75%) / 末尾(100%)
- 共 25 次 API 调用

主要指标
--------
- Accuracy@size     : 给定上下文规模下的回答命中率
- Accuracy@position : 不同针位的命中率（Lost-in-the-Middle 效应）
- prompt_tokens     : API 返回的实际提示词 token 消耗
- latency           : API 响应时间（秒）

运行
----
    conda run -n Shopping_agent python tests/evaluation/evaluate_context.py
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from app.services.llm_service import get_llm_service
from app.core.config import settings

# ═══════════════════════════════════════════════════════════════════════════════
# §0  商品目录（50 个产品，每个有全局唯一的屏幕刷新率作为"针"）
# ═══════════════════════════════════════════════════════════════════════════════

import random

# 生成 50 个唯一刷新率值（Hz），利用固定随机种子保证可复现性
_base_rates = list(range(60, 199, 3))   # 60,63,66,...,198 → 47 个值
_extra_rates = [61, 91, 121]             # 补足至 50 个
_pool = _base_rates + _extra_rates       # 50 个全局唯一值
_rng0 = random.Random(7)
_rng0.shuffle(_pool)
NEEDLE_RATES: List[int] = _pool          # NEEDLE_RATES[i] = 第 i+1 个商品的刷新率

# 50 个品牌 × 型号 × 基础售价
BRAND_MODEL_PRICE: List[Tuple[str, str, int]] = [
    ("苹果",  "iPhone 15",            5999),
    ("苹果",  "iPhone 15 Plus",       6999),
    ("苹果",  "iPhone 15 Pro",        7999),
    ("苹果",  "iPhone 15 Pro Max",    9999),
    ("苹果",  "iPhone SE 第三代",     3299),
    ("三星",  "Galaxy S24",           5999),
    ("三星",  "Galaxy S24 Ultra",     9999),
    ("三星",  "Galaxy A55",           2999),
    ("三星",  "Galaxy A35",           1999),
    ("三星",  "Galaxy Z Fold5",      13999),
    ("小米",  "14",                   3999),
    ("小米",  "14 Pro",               4999),
    ("小米",  "14 Ultra",             5999),
    ("小米",  "CIVI 3",               2999),
    ("小米",  "Redmi Note 13 Pro",    1999),
    ("华为",  "Mate 60",              4999),
    ("华为",  "Mate 60 Pro",          6499),
    ("华为",  "P70",                  4599),
    ("华为",  "nova 12",              2799),
    ("华为",  "nova 12 Pro",          3699),
    ("一加",  "12",                   4299),
    ("一加",  "12R",                  2499),
    ("一加",  "Nord 4",               2199),
    ("一加",  "Open",                 7999),
    ("一加",  "Ace 3V",               1999),
    ("OPPO", "Find X7",              4999),
    ("OPPO", "Find X7 Pro",          6499),
    ("OPPO", "Reno11",               2699),
    ("OPPO", "Reno11 Pro",           3499),
    ("OPPO", "A3 Pro",               1799),
    ("vivo", "X100",                 3999),
    ("vivo", "X100 Pro",             4999),
    ("vivo", "S18",                  2699),
    ("vivo", "S18 Pro",              3499),
    ("vivo", "Y200 GT",              1999),
    ("谷歌",  "Pixel 9",              5999),
    ("谷歌",  "Pixel 9 Pro",          7999),
    ("谷歌",  "Pixel 9 Pro XL",       8999),
    ("谷歌",  "Pixel 8a",             3999),
    ("谷歌",  "Pixel Fold",          11999),
    ("索尼",  "Xperia 1 VI",          8999),
    ("索尼",  "Xperia 5 VI",          5999),
    ("索尼",  "Xperia 10 VI",         3499),
    ("索尼",  "Xperia 1 V",           7499),
    ("索尼",  "Xperia 5 V",           4999),
    ("真我",  "GT5 Pro",              3499),
    ("真我",  "GT Neo 6",             2299),
    ("真我",  "GT6",                  3299),
    ("真我",  "12 Pro+",              2599),
    ("真我",  "C65",                  1299),
]

CPUS = [
    "骁龙8 Gen3", "骁龙8 Gen2", "骁龙8s Gen3", "骁龙7s Gen2", "骁龙7 Gen3",
    "天玑9300+",  "天玑9300",   "天玑9200",    "天玑8300",    "天玑7200",
    "麒麟9010",   "麒麟9000S",  "苹果A17 Pro", "苹果A16",     "苹果A15",
    "Exynos 2400","联发科G99",  "骁龙6s Gen3", "天玑6100+",   "骁龙4 Gen1",
]

FEATURES_POOL = [
    "IP68防水防尘", "卫星通话支持", "三重主摄系统", "光学防抖OIS", "超声波指纹识别",
    "AI实时翻译",   "钛金属边框",   "官方售后保障",  "双卡双待5G",   "NFC便捷支付",
    "Hi-Res音频认证","无线充电50W", "反向无线充电",  "立体声扬声器",  "杜比全景声",
    "IP54防尘防水", "大猩猩Victus2玻璃","人脸快速解锁","超感知摄像","AI智慧场景识别",
]


def generate_products() -> List[Dict]:
    """生成 50 个商品字典，每个商品有唯一的屏幕刷新率（NEEDLE_RATES[i]）。"""
    products = []
    for idx, (brand, model, base_price) in enumerate(BRAND_MODEL_PRICE):
        rng = random.Random(idx * 137 + 99)
        products.append({
            "brand":        brand,
            "model":        model,
            "price":        base_price,
            "screen":       rng.choice([6.1, 6.2, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]),
            "resolution":   rng.choice(["2400×1080","2772×1260","3088×1440","2800×1260"]),
            "refresh_rate": NEEDLE_RATES[idx],   # ← 唯一的"针"
            "brightness":   rng.choice([1000, 1200, 1500, 1600, 2000, 2500]),
            "cpu":          CPUS[idx % len(CPUS)],
            "ram":          rng.choice([8, 12, 16, 24]),
            "storage":      rng.choice([128, 256, 512, 1024]),
            "main_cam":     rng.choice([12, 50, 64, 108, 200]),
            "front_cam":    rng.choice([12, 16, 32]),
            "battery":      rng.randint(28, 60) * 100,
            "charge":       rng.choice([18, 25, 33, 45, 65, 80, 100, 120, 150, 240]),
            "weight":       rng.randint(168, 235),
            "features":     rng.sample(FEATURES_POOL, 3),
        })
    return products


def format_product(p: Dict, position: int) -> str:
    """将商品字典格式化为详细的文本描述（约 130 中文字 + 数字 ≈ 200 token）。"""
    feats = "、".join(p["features"])
    return (
        f"[第{position}个商品]\n"
        f"品牌型号：{p['brand']} {p['model']}\n"
        f"产品类别：智能手机\n"
        f"参考售价：¥{p['price']}\n"
        f"屏幕规格：{p['screen']}英寸 {p['resolution']} AMOLED材质 "
        f"{p['refresh_rate']}Hz屏幕刷新率 峰值亮度{p['brightness']}尼特\n"
        f"处理器配置：{p['cpu']}处理器 {p['ram']}GB运行内存 {p['storage']}GB闪存\n"
        f"影像系统：主摄{p['main_cam']}MP超清镜头 前置{p['front_cam']}MP自拍镜头\n"
        f"续航配置：{p['battery']}mAh大容量电池 支持{p['charge']}W超级快充\n"
        f"机身重量：{p['weight']}克\n"
        f"产品亮点：{feats}"
    )


def build_haystack(products: List[Dict], n: int, needle_pos: int) -> Tuple[str, int]:
    """
    构建包含 n 个商品的背景文本，针（needle）位于 needle_pos 处（1-indexed）。
    返回 (context_string, expected_refresh_rate)。
    """
    selected  = products[:n]
    parts     = [format_product(p, i + 1) for i, p in enumerate(selected)]
    context   = "\n\n".join(parts)
    expected  = selected[needle_pos - 1]["refresh_rate"]
    return context, expected


def get_needle_positions(n: int) -> List[Tuple[int, str]]:
    """返回 5 个针位：(位置, 标签)，覆盖列表的 20%/40%/50%/75%/100%。"""
    return [
        (max(1, n // 5),         "开头 (~20%)"),
        (max(2, 2 * n // 5),     "前段 (~40%)"),
        (max(3, n // 2),         "中间 (~50%)"),
        (max(4, 3 * n // 4),     "后段 (~75%)"),
        (n,                      "末尾 (100%)"),
    ]


def estimate_tokens(text: str) -> int:
    """字符级 token 估算（无 tiktoken）：中文 ×1.5 + ASCII词 ×0.75。"""
    zh    = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_w = len(re.findall(r'[a-zA-Z0-9]+', text))
    other = len(text) - zh - ascii_w
    return int(zh * 1.5 + ascii_w * 0.75 + other * 0.5)


def check_correct(response: str, expected: int) -> bool:
    """验证模型回答是否包含正确的刷新率数值。"""
    nums = re.findall(r'\d+', response)
    return str(expected) in nums


# ═══════════════════════════════════════════════════════════════════════════════
# §1  单次测试执行
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个精确的商品参数检索助手。"
    "你的任务是在给定的商品列表中，找到指定序号的商品，并提取其具体属性值。"
    "只返回所询问的数字，不要任何解释或单位以外的文字。"
    "例如，如果问刷新率，只回答 '120' 或 '120Hz'。"
)


async def run_single_test(
    llm_svc,
    products: List[Dict],
    context_size: int,
    needle_pos: int,
    pos_label: str,
    case_id: int,
    total: int,
) -> Dict[str, Any]:
    """执行一次大海捞针测试并返回结构化结果。"""
    context, expected_rate = build_haystack(products, context_size, needle_pos)

    question = (
        f"请仔细阅读以上 {context_size} 个商品的详细参数，然后回答：\n"
        f"列表中第 {needle_pos} 个商品（标记为[第{needle_pos}个商品]）"
        f"的屏幕刷新率是多少Hz？\n"
        f"只返回数字即可，例如 120 或 120Hz。"
    )

    user_content = (
        f"以下是 {context_size} 个智能手机商品的详细参数列表：\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"{question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    est_prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user_content)

    t0 = time.perf_counter()
    try:
        # GLM-4.6 是推理模型，reasoning_content 会消耗大量 token。
        # 需要 max_tokens≥1500 才能让模型在思维链后输出最终 content。
        resp = await llm_svc.chat_completion(
            messages    = messages,
            temperature = 0.1,
            max_tokens  = 2000,  # 留足推理空间（reasoning_tokens ≈ 400-1000）
        )
        latency = time.perf_counter() - t0

        content = resp.get("content", "").strip()
        tu = resp.get("tokens_used") or {}
        prompt_tokens     = tu.get("prompt_tokens")     or est_prompt_tokens
        completion_tokens = tu.get("completion_tokens") or estimate_tokens(content)
        total_tokens      = tu.get("total_tokens")      or (prompt_tokens + completion_tokens)

        correct = check_correct(content, expected_rate)

        tag = "✅" if correct else "❌"
        print(f"  [{case_id:2d}/{total}] {tag} "
              f"规模={context_size:2d} 针位={needle_pos:2d}({pos_label}) "
              f"期望={expected_rate}Hz 回复='{content[:60]}' "
              f"Tokens={prompt_tokens} 耗时={latency:.1f}s")

        return {
            "context_size":      context_size,
            "needle_pos":        needle_pos,
            "pos_label":         pos_label,
            "pos_pct":           round(needle_pos / context_size, 2),
            "expected":          expected_rate,
            "response":          content,
            "correct":           correct,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
            "est_prompt_tokens": est_prompt_tokens,
            "latency":           round(latency, 2),
            "error":             None,
        }

    except Exception as exc:
        latency = time.perf_counter() - t0
        print(f"  [{case_id:2d}/{total}] ⚠️  "
              f"规模={context_size} 针位={needle_pos}({pos_label}) "
              f"错误: {exc}")
        return {
            "context_size":      context_size,
            "needle_pos":        needle_pos,
            "pos_label":         pos_label,
            "pos_pct":           round(needle_pos / context_size, 2),
            "expected":          expected_rate,
            "response":          "",
            "correct":           False,
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "est_prompt_tokens": est_prompt_tokens,
            "latency":           round(latency, 2),
            "error":             str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §2  报告生成
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXT_SIZES = [10, 20, 30, 40, 50]
POS_LABELS    = ["开头 (~20%)", "前段 (~40%)", "中间 (~50%)", "后段 (~75%)", "末尾 (100%)"]


def acc_bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def print_report(results: List[Dict], products: List[Dict]):
    by_size = defaultdict(list)
    by_pos  = defaultdict(list)
    for r in results:
        by_size[r["context_size"]].append(r)
        by_pos[r["pos_label"]].append(r)

    print("\n" + "═" * 72)
    print("  §A  总体结果：上下文规模 vs 准确率 / Token 消耗 / 延迟")
    print("═" * 72)
    header = f"  {'规模':>8} | {'准确率':>7} | {'实际Token':>10} | {'估算Token':>10} | {'均值延迟':>9}"
    print(header)
    print("  " + "-" * 8 + " | " + "-" * 7 + " | " + "-" * 10 + " | " + "-" * 10 + " | " + "-" * 9)

    size_stats = {}
    for n in CONTEXT_SIZES:
        rs = by_size[n]
        if not rs:
            continue
        valid = [r for r in rs if not r["error"]]
        acc   = sum(r["correct"] for r in valid) / max(len(valid), 1)
        avg_pt  = sum(r["prompt_tokens"] for r in valid) / max(len(valid), 1)
        avg_est = sum(r["est_prompt_tokens"] for r in valid) / max(len(valid), 1)
        avg_lat = sum(r["latency"] for r in rs) / len(rs)
        size_stats[n] = {"acc": acc, "prompt_tokens": avg_pt}
        print(f"  {str(n)+'个商品':>8} | {acc:>6.1%} | {avg_pt:>10,.0f} | {avg_est:>10,.0f} | {avg_lat:>8.1f}s")

    print("\n" + "═" * 72)
    print("  §B  准确率走势图（上下文规模递增）")
    print("═" * 72)
    print()
    for n in CONTEXT_SIZES:
        if n not in size_stats:
            continue
        acc = size_stats[n]["acc"]
        bar = acc_bar(acc)
        print(f"  {str(n)+'个商品':>8}: [{bar}] {acc:.0%}")

    print("\n" + "═" * 72)
    print("  §C  位置效应分析（Lost-in-the-Middle 效应）")
    print("═" * 72)
    print(f"\n  {'针位':>14} | {'命中次数':>8} | {'准确率':>7} | 准确率条形图")
    print("  " + "-" * 14 + " | " + "-" * 8 + " | " + "-" * 7 + " | " + "-" * 25)

    pos_accs = {}
    for label in POS_LABELS:
        rs = by_pos[label]
        if not rs:
            continue
        valid = [r for r in rs if not r["error"]]
        hits  = sum(r["correct"] for r in valid)
        acc   = hits / max(len(valid), 1)
        pos_accs[label] = acc
        bar = acc_bar(acc)
        print(f"  {label:>14} | {hits:>4}/{len(valid):<3} | {acc:>6.1%} | [{bar}]")

    # Find worst position
    if pos_accs:
        worst_pos = min(pos_accs, key=pos_accs.get)
        best_pos  = max(pos_accs, key=pos_accs.get)
        print(f"\n  ▶  最佳位置: {best_pos} ({pos_accs[best_pos]:.0%})")
        print(f"  ▶  最差位置: {worst_pos} ({pos_accs[worst_pos]:.0%})")
        middle_acc = pos_accs.get("中间 (~50%)", None)
        if middle_acc is not None:
            begin_acc = pos_accs.get("开头 (~20%)", middle_acc)
            end_acc   = pos_accs.get("末尾 (100%)", middle_acc)
            gap = max(begin_acc, end_acc) - middle_acc
            if gap > 0.1:
                print(f'  ▶  "Lost in the Middle"效应明显：中间位置比首/末低 {gap:.0%}')
            else:
                print('  ▶  "Lost in the Middle"效应不明显（中间与首末差距 < 10%）')

    print("\n" + "═" * 72)
    print("  §D  Token 消耗详情（API 实际返回）")
    print("═" * 72)
    print(f"\n  {'规模':>8} | {'针位':>14} | {'实际Token':>10} | {'估算Token':>10} | {'误差':>7} | {'回答'}")
    print("  " + "-"*70)
    for r in sorted(results, key=lambda x: (x["context_size"], x["pos_pct"])):
        if r["error"]:
            status = "ERR"
        else:
            status = "✅" if r["correct"] else "❌"
        err_pct = ""
        if r["prompt_tokens"] and r["est_prompt_tokens"]:
            err = (r["prompt_tokens"] - r["est_prompt_tokens"]) / r["est_prompt_tokens"]
            err_pct = f"{err:+.0%}"
        print(f"  {str(r['context_size'])+'个商品':>8} | {r['pos_label']:>14} | "
              f"{r['prompt_tokens']:>10,} | {r['est_prompt_tokens']:>10,} | "
              f"{err_pct:>7} | {status} {r['response'][:30]}")

    # Determine effective context limit
    print("\n" + "═" * 72)
    print("  §E  有效上下文极限判定")
    print("═" * 72)
    threshold = 0.60  # 60% accuracy threshold
    limit_size = None
    for n in CONTEXT_SIZES:
        if n in size_stats and size_stats[n]["acc"] < threshold:
            limit_size = n
            break

    if limit_size:
        prev = CONTEXT_SIZES[CONTEXT_SIZES.index(limit_size) - 1] if CONTEXT_SIZES.index(limit_size) > 0 else limit_size
        print(f"\n  当阈值为 {threshold:.0%} 准确率时：")
        print(f"  → 上下文规模达到 {limit_size} 个商品（~{size_stats[limit_size]['prompt_tokens']:.0f} Token）时准确率首次低于阈值")
        print(f"  → 在 {prev} 个商品（~{size_stats[prev]['prompt_tokens']:.0f} Token）时仍能保持 ≥{threshold:.0%} 准确率")
        print(f"  → 建议有效上下文上限：约 {size_stats[prev]['prompt_tokens']:.0f} Token")
    else:
        max_size = CONTEXT_SIZES[-1]
        if max_size in size_stats:
            print(f"\n  在测试范围内（最大 {max_size} 个商品 ≈ {size_stats[max_size]['prompt_tokens']:.0f} Token），")
            print(f"  准确率始终高于 {threshold:.0%}，未发现有效上下文极限。")
            print(f"  → 建议扩大测试规模（如 100/200 个商品）进一步探测极限。")

    # Middle-position trend
    middle_results = [r for r in results if r["pos_label"] == "中间 (~50%)"]
    if len(middle_results) >= 3:
        print(f"\n  中间位置准确率随规模变化（最能体现 Lost-in-the-Middle 效应）：")
        for r in sorted(middle_results, key=lambda x: x["context_size"]):
            tag = "✅" if r["correct"] else "❌"
            print(f"    规模={r['context_size']:2d}个商品 针位=第{r['needle_pos']:2d}个 {tag} "
                  f"期望={r['expected']}Hz 回复='{r['response']}'")

    print()


def save_markdown_report(results: List[Dict], products: List[Dict]):
    """将测试结果保存为 Markdown 报告。"""
    by_size = defaultdict(list)
    by_pos  = defaultdict(list)
    for r in results:
        by_size[r["context_size"]].append(r)
        by_pos[r["pos_label"]].append(r)

    model_name = settings.text_model
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 长上下文处理能力评测报告（大海捞针）",
        f"",
        f"> 生成时间：{ts}  ",
        f"> 模型：`{model_name}`  ",
        f"> 测试方法：将 N 个商品的详细参数文本注入上下文，询问中间某商品的屏幕刷新率",
        f"",
        f"## 实验设置",
        f"",
        f"| 参数 | 值 |",
        f"|------|-----|",
        f"| 上下文规模 | 10 / 20 / 30 / 40 / 50 个商品 |",
        f"| 针位配置 | 开头(~20%) / 前段(~40%) / 中间(~50%) / 后段(~75%) / 末尾(100%) |",
        f"| 总 API 调用次数 | {len(results)} |",
        f"| 询问属性 | 屏幕刷新率（Hz），每商品值全局唯一 |",
        f"| 判断方式 | 回答中是否包含正确的数字值 |",
        f"| LLM temperature | 0.0（事实性检索） |",
        f"",
        f"## 总体结果",
        f"",
        f"| 上下文规模 | 准确率 | 实际 Prompt Token | 估算 Token | 平均延迟 |",
        f"|-----------|--------|------------------|------------|---------|",
    ]

    size_stats = {}
    for n in CONTEXT_SIZES:
        rs = by_size[n]
        if not rs:
            continue
        valid = [r for r in rs if not r["error"]]
        acc   = sum(r["correct"] for r in valid) / max(len(valid), 1)
        avg_pt  = sum(r["prompt_tokens"] for r in valid) / max(len(valid), 1)
        avg_est = sum(r["est_prompt_tokens"] for r in valid) / max(len(valid), 1)
        avg_lat = sum(r["latency"] for r in rs) / len(rs)
        size_stats[n] = {"acc": acc, "prompt_tokens": avg_pt}
        lines.append(f"| **{n}个商品** | **{acc:.0%}** | {avg_pt:,.0f} | {avg_est:,.0f} | {avg_lat:.1f}s |")

    lines += [
        f"",
        f"## 位置效应分析（Lost-in-the-Middle）",
        f"",
        f"| 针位 | 命中/总数 | 准确率 |",
        f"|------|----------|--------|",
    ]

    for label in POS_LABELS:
        rs = by_pos[label]
        if not rs:
            continue
        valid = [r for r in rs if not r["error"]]
        hits  = sum(r["correct"] for r in valid)
        acc   = hits / max(len(valid), 1)
        lines.append(f"| **{label}** | {hits}/{len(valid)} | {acc:.0%} |")

    lines += [
        f"",
        f"## 逐条明细",
        f"",
        f"| 规模 | 针位 | 位置 | 期望(Hz) | 实际Token | 回答 | 是否正确 | 延迟(s) |",
        f"|------|------|------|---------|----------|------|---------|--------|",
    ]
    for r in sorted(results, key=lambda x: (x["context_size"], x["pos_pct"])):
        ok = "✅" if r["correct"] else "❌"
        err = f"ERROR: {r['error'][:30]}" if r["error"] else r["response"]
        lines.append(
            f"| {r['context_size']}个商品 | 第{r['needle_pos']}个 | {r['pos_label']} "
            f"| {r['expected']} | {r['prompt_tokens']:,} | `{err[:40]}` | {ok} | {r['latency']} |"
        )

    # Conclusion
    threshold = 0.60
    limit_size = None
    for n in CONTEXT_SIZES:
        if n in size_stats and size_stats[n]["acc"] < threshold:
            limit_size = n
            break

    lines += ["", "## 结论", ""]
    if limit_size:
        prev_idx = CONTEXT_SIZES.index(limit_size) - 1
        prev = CONTEXT_SIZES[prev_idx] if prev_idx >= 0 else limit_size
        lines.append(
            f"1. 在 **{limit_size}个商品（≈{size_stats[limit_size]['prompt_tokens']:.0f} Token）** 时，"
            f"准确率首次跌破 {threshold:.0%}（实际为 {size_stats[limit_size]['acc']:.0%}）。"
        )
        lines.append(
            f"2. 有效上下文上限约为 **{size_stats[prev]['prompt_tokens']:.0f} Token**（{prev}个商品时仍≥{threshold:.0%}）。"
        )
    else:
        lines.append(
            f"1. 在测试范围内（≤50个商品，≈{size_stats.get(50,{}).get('prompt_tokens',0):.0f} Token），"
            f"准确率始终高于 {threshold:.0%}，建议扩大规模进一步测试。"
        )

    middle_accs = {}
    for r in results:
        if r["pos_label"] == "中间 (~50%)":
            middle_accs[r["context_size"]] = r["correct"]

    dropping = [n for n in CONTEXT_SIZES if n in middle_accs and not middle_accs[n]]
    if dropping:
        lines.append(
            f'3. 中间位置在上下文规模 **{min(dropping)}个商品** 时开始出错，呈现"Lost in the Middle"现象。'
        )
    else:
        lines.append('3. 未观察到明显的"Lost in the Middle"效应。')

    lines += [
        f"",
        f"---",
        f"*由 `evaluate_context.py` 自动生成 @ {ts}*",
        f"*测试模型：{model_name}*",
    ]

    report_path = EVAL_DIR / "context_utilization_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 报告已保存至: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  主函数
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 72)
    print("  大海捞针 上下文利用率评测（Needle-in-a-Haystack）")
    print(f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型：{settings.text_model}  |  Provider: {settings.llm_provider}")
    print("=" * 72)

    # 初始化产品库 & LLM 服务
    products = generate_products()
    print(f"\n  ✓ 商品目录已生成：{len(products)} 个商品，刷新率唯一值已验证: "
          f"{len(set(p['refresh_rate'] for p in products)) == len(products)}")

    try:
        llm_svc = get_llm_service()
        print(f"  ✓ LLM 服务初始化成功（{settings.text_model}）")
    except Exception as e:
        print(f"  ✗ LLM 服务初始化失败: {e}")
        return

    # 构建测试用例
    test_cases: List[Tuple[int, int, str]] = []
    for n in CONTEXT_SIZES:
        for pos, label in get_needle_positions(n):
            test_cases.append((n, pos, label))

    total = len(test_cases)
    est_time = total * 8
    print(f"\n  测试计划：{total} 个用例 | 预计 {est_time//60}分{est_time%60}秒")
    print(f"  上下文规模：{CONTEXT_SIZES}")
    print(f"  针位分布：每规模 5 个位置（开头/前段/中间/后段/末尾）")
    print(f"\n  开始测试...\n")

    results: List[Dict] = []
    for i, (n, pos, label) in enumerate(test_cases, 1):
        result = await run_single_test(llm_svc, products, n, pos, label, i, total)
        results.append(result)
        # 小延迟避免触发 API 速率限制
        if i < total:
            await asyncio.sleep(0.3)

    # 统计与报告
    total_correct = sum(r["correct"] for r in results)
    total_valid   = sum(1 for r in results if not r["error"])
    overall_acc   = total_correct / max(total_valid, 1)
    total_time    = sum(r["latency"] for r in results)

    print(f"\n  {'─'*68}")
    print(f"  总体准确率: {total_correct}/{total_valid} = {overall_acc:.1%}")
    print(f"  总耗时:    {total_time:.1f}s  |  均值: {total_time/len(results):.1f}s/次")
    print(f"  {'─'*68}")

    print_report(results, products)
    save_markdown_report(results, products)

    # Save raw results JSON
    raw_path = EVAL_DIR / "context_raw_results.json"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  📊 原始数据已保存至: {raw_path}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
