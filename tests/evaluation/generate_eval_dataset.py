#!/usr/bin/env python3
"""
generate_eval_dataset.py
========================
从数据库抽取真实商品，基于规则模板自动生成 50 条评测用例。

意图分布（共 50 条）：
  CHITCHAT        8 条  ← 闲聊，无需检索
  PRODUCT_SEARCH 16 条  ← 精确商品搜索（含品牌/型号）
  FUZZY_SEARCH   16 条  ← 模糊语义搜索（按需求/场景描述）
  COMPARISON     10 条  ← 商品对比

每条用例字段：
  id              序号
  query           用户提问
  expected_intent 预期路由意图
  expected_skus   预期召回的商品 ID 列表
  expected_keywords 预期命中的关键词（辅助评测）
  notes           备注（可选）

运行：
  cd tests/evaluation
  python generate_eval_dataset.py
"""

import json
import random
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ── 路径配置 ──────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]   # Shopping_agent/
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# SQLite URL 是相对路径（./llm_agent.db），必须切到 backend 目录
os.chdir(BACKEND)

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

from app.core.database import SessionLocal
from sqlalchemy import text

random.seed(42)   # 固定随机种子，保证可复现

# ═══════════════════════════════════════════════════════════════════
# 1. 数据库拉取
# ═══════════════════════════════════════════════════════════════════

def fetch_products(db, limit: int = 300) -> List[Dict]:
    """从 products 表拉取高质量 Amazon 美妆商品。"""
    rows = db.execute(text("""
        SELECT product_id, title, brand, price, rating, review_count, description, category
        FROM products
        WHERE platform = 'amazon'
          AND image_url IS NOT NULL
          AND price    > 0
          AND rating   >= 3.0
          AND title    IS NOT NULL
        ORDER BY review_count DESC NULLS LAST
        LIMIT :lim
    """), {"lim": limit}).fetchall()

    products = []
    for r in rows:
        products.append({
            "id":           r[0],
            "title":        (r[1] or "").strip(),
            "brand":        (r[2] or "").strip(),
            "price":        r[3] or 0.0,
            "rating":       r[4] or 0.0,
            "review_count": r[5] or 0,
            "description":  (r[6] or "").strip(),
            "category":     (r[7] or "").strip(),
        })
    return products


# ═══════════════════════════════════════════════════════════════════
# 2. 产品类型标注
# ═══════════════════════════════════════════════════════════════════

# (产品类型, 匹配关键词列表, 中文类型名)
PRODUCT_TYPES = [
    ("lipstick",      ["lipstick", "lip color", "lip stick", "lip matte"],    "口红/唇膏"),
    ("nail polish",   ["nail polish", "nail color", "nail lacquer",
                       "nail prism", "gel nail", "nail light"],                "指甲油/美甲"),
    ("shampoo",       ["shampoo"],                                             "洗发水"),
    ("conditioner",   ["conditioner"],                                         "护发素"),
    ("foundation",    ["foundation", "stay matte", "liquid found"],            "粉底液"),
    ("mascara",       ["mascara"],                                             "睫毛膏"),
    ("eyeshadow",     ["eyeshadow", "eye shadow"],                            "眼影"),
    ("blush",         ["blush"],                                              "腮红"),
    ("perfume",       ["perfume", "fragrance", "cologne", "eau de"],          "香水"),
    ("face cream",    ["cream", "moisturizer", "moisturizing cream"],         "面霜"),
    ("serum",         ["serum"],                                              "精华液"),
    ("sunscreen",     ["sunscreen", "sun lotion", "spf 50"],                  "防晒霜"),
    ("makeup brush",  ["brush", "beauty brush"],                              "化妆刷"),
    ("body lotion",   ["lotion", "body lotion"],                              "身体乳"),
    ("body scrub",    ["scrub"],                                              "磨砂膏"),
    ("body oil",      ["body oil", "essential oil"],                          "精油"),
    ("cleanser",      ["cleanser", "face wash", "cleansing"],                 "洁面产品"),
    ("concealer",     ["concealer"],                                          "遮瑕笔"),
    ("highlighter",   ["highlighter"],                                        "高光"),
    ("wig",           ["wig cap", "wig"],                                     "假发/发套"),
    ("hair dryer",    ["hair dryer", "blow dryer", "nozzle adapter"],        "吹风机"),
    ("foot mask",     ["foot mask", "pedi"],                                  "足膜"),
    ("makeup case",   ["makeup case", "makeup bag", "artist case"],           "化妆包"),
    ("shower cap",    ["shower cap"],                                         "浴帽"),
    ("nail holder",   ["nail polish holder"],                                 "指甲油固定架"),
]


def tag_product_type(title: str):
    """返回 (en_type, zh_type) 或 None。"""
    t = title.lower()
    for en, kws, zh in PRODUCT_TYPES:
        if any(kw in t for kw in kws):
            return en, zh
    return None


def group_by_type(products: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for p in products:
        result = tag_product_type(p["title"])
        if result:
            en, _ = result
            grouped.setdefault(en, []).append(p)
    return grouped


# ═══════════════════════════════════════════════════════════════════
# 3. 意图生成器
# ═══════════════════════════════════════════════════════════════════

# ─── CHITCHAT ────────────────────────────────────────────────────

CHITCHAT_CASES = [
    "你好，你能帮我做什么？",
    "我想买一些美妆产品，有什么推荐吗？",
    "护肤和美妆有什么区别？",
    "你好！请介绍一下你自己。",
    "美妆产品怎么选择适合自己的？",
    "网购美妆产品需要注意什么？",
    "美妆产品的保质期一般是多久？",
    "初学者化妆应该从哪里开始？",
    "什么是清洁型护肤品？",
    "你会推荐美妆产品吗？",
]

def gen_chitchat(n: int) -> List[Dict]:
    samples = random.sample(CHITCHAT_CASES, min(n, len(CHITCHAT_CASES)))
    return [
        {
            "expected_intent":    "CHITCHAT",
            "query":              q,
            "expected_skus":      [],
            "expected_keywords":  [],
            "notes":              "闲聊意图，AI 直接回答，无需检索商品数据库",
        }
        for q in samples[:n]
    ]


# ─── PRODUCT_SEARCH（精确搜索）────────────────────────────────────

# 精确搜索的查询模板（{brand} {en_type} {zh_type} {title_s} {price} 可用）
EXACT_SEARCH_TEMPLATES = [
    "我想找 {brand} 的{zh_type}",
    "帮我搜索 {brand} {en_type}",
    "有没有 {brand} 品牌的{zh_type}？",
    "{brand} {zh_type} 有货吗",
    "我要买 {title_s}",
    "帮我找一下 {title_s}",
    "搜索 {brand} {zh_type}，预算大概 ${price:.0f}",
    "{brand} 的{zh_type}多少钱？",
    "我听说 {brand} 的{zh_type}很好用，帮我找一下",
    "查一查 {brand} {en_type} 的相关产品",
    "{title_s} 这款产品怎么样？",
    "帮我找 {brand} 品牌的{zh_type}，评分比较高的",
]

def gen_product_search(products: List[Dict], n: int) -> List[Dict]:
    """生成精确商品搜索用例（有品牌名 + 品类）。"""
    # 优先选有品牌且能识别类型的商品
    candidates = []
    for p in products:
        if not p["brand"]:
            continue
        result = tag_product_type(p["title"])
        if result:
            en, zh = result
            candidates.append((p, en, zh))

    random.shuffle(candidates)
    cases, used = [], set()

    for p, en, zh in candidates:
        if len(cases) >= n:
            break
        if p["id"] in used:
            continue
        used.add(p["id"])

        title_s = p["title"][:40].rstrip()
        template = random.choice(EXACT_SEARCH_TEMPLATES)
        query = template.format(
            brand=p["brand"],
            en_type=en,
            zh_type=zh,
            title_s=title_s,
            price=p["price"],
        )

        cases.append({
            "expected_intent":    "PRODUCT_SEARCH",
            "query":              query,
            "expected_skus":      [p["id"]],
            "expected_keywords":  [p["brand"].lower(), en],
            "product_ref": {
                "title":   p["title"],
                "brand":   p["brand"],
                "price":   p["price"],
                "rating":  p["rating"],
            },
            "notes": "精确搜索，预期命中该商品",
        })

    return cases[:n]


# ─── FUZZY_SEARCH（模糊语义搜索）─────────────────────────────────

# (type_hint, query) — query 只描述需求/场景，不含品牌
FUZZY_SEARCH_CASES = [
    # 指甲
    ("nail polish",  "我在找一款颜色持久不容易掉的指甲油"),
    ("nail polish",  "有没有适合新手做美甲的套装？需要包含灯"),
    # 洗护
    ("shampoo",      "我头发容易出油，有没有推荐的控油洗发水？"),
    ("shampoo",      "推荐一款适合细软发质、增加蓬松感的洗发水"),
    ("conditioner",  "我的发尾很干燥毛躁，有没有修护效果好的护发素？"),
    # 护肤
    ("face cream",   "皮肤很干，想找一款深层保湿的面霜"),
    ("face cream",   "有没有适合敏感肌、无香精的面霜？"),
    ("serum",        "想找一款抗衰老、改善细纹的精华液"),
    ("serum",        "有没有提亮肤色、淡化色斑的精华产品？"),
    ("sunscreen",    "推荐一款防晒指数高、不油腻、适合日常的防晒霜"),
    ("cleanser",     "有没有温和不紧绷的洁面乳？适合干皮使用"),
    # 彩妆
    ("lipstick",     "我想找一款显色度高、持妆时间长的哑光口红"),
    ("foundation",   "需要一款遮瑕力强但妆感自然的粉底液，油皮适用"),
    ("concealer",    "推荐一款遮盖黑眼圈效果好的遮瑕笔或遮瑕膏"),
    ("mascara",      "有没有防水、防晕染、增长睫毛效果好的睫毛膏？"),
    ("eyeshadow",    "我想要一盘适合日常通勤的大地色眼影盘"),
    # 工具/其他
    ("makeup brush", "推荐一套适合初学者的化妆刷，刷毛柔软不掉毛"),
    ("hair dryer",   "想买一个快干、控温、不伤发质的吹风机"),
    ("body lotion",  "有没有清爽不黏腻、快速吸收的全身身体乳？"),
    ("perfume",      "我喜欢清新花香调的香水，适合春天日常使用"),
]

def gen_fuzzy_search(products: List[Dict], n: int) -> List[Dict]:
    """生成模糊语义搜索用例。"""
    grouped = group_by_type(products)
    samples = random.sample(FUZZY_SEARCH_CASES, min(n, len(FUZZY_SEARCH_CASES)))

    cases = []
    for type_hint, query in samples[:n]:
        # 找到对应类型的商品作为参考 expected_skus
        matched = grouped.get(type_hint, [])
        # 兜底：模糊匹配前缀
        if not matched:
            for key, prods in grouped.items():
                if type_hint in key or key in type_hint:
                    matched.extend(prods)
        expected_skus = [p["id"] for p in matched[:3]]

        cases.append({
            "expected_intent":   "FUZZY_SEARCH",
            "query":             query,
            "expected_skus":     expected_skus,
            "expected_keywords": [type_hint],
            "notes":             f"语义搜索，预期召回 [{type_hint}] 类商品",
        })

    return cases[:n]


# ─── COMPARISON（商品对比）───────────────────────────────────────

COMPARISON_TEMPLATES = [
    "{title1} 和 {title2} 哪个更好用？",
    "帮我比较一下 {brand1} 和 {brand2} 的{zh_type}",
    "{title1} vs {title2}，哪个性价比更高？",
    "{brand1} 的{zh_type}和 {brand2} 的{zh_type}有什么区别？",
    "我在 {title1} 和 {title2} 之间纠结，帮我选一个",
    "比较 {brand1} 和 {brand2} 这两款{zh_type}的优缺点",
    "{brand1} {en_type} 和 {brand2} {en_type} 哪个更适合日常使用？",
    "我看中了 {title1} 和 {title2}，价格差不多，哪个评价更好？",
    "{title1}（${price1:.0f}）和 {title2}（${price2:.0f}），该怎么选？",
    "同样是{zh_type}，{brand1} 和 {brand2} 各有什么特点？",
]

def gen_comparison(products: List[Dict], n: int) -> List[Dict]:
    """生成商品对比用例（同类型、不同品牌）。"""
    grouped = group_by_type(products)

    # 构建可对比的 (ptype, p1, p2) 列表
    pairs = []
    for en, prods in grouped.items():
        # 过滤有品牌的
        branded = [p for p in prods if p["brand"]]
        if len(branded) < 2:
            continue
        # 找同类型中品牌不同的配对
        random.shuffle(branded)
        for i in range(len(branded) - 1):
            for j in range(i + 1, len(branded)):
                if branded[i]["brand"].lower() != branded[j]["brand"].lower():
                    pairs.append((en, branded[i], branded[j]))

    random.shuffle(pairs)
    # 找出 zh_type
    en_to_zh = {en: zh for en, _, zh in PRODUCT_TYPES}

    cases, used_pairs = [], set()
    for en, p1, p2 in pairs:
        if len(cases) >= n:
            break
        pair_key = tuple(sorted([p1["id"], p2["id"]]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        zh = en_to_zh.get(en, en)
        title1 = p1["title"][:35].rstrip()
        title2 = p2["title"][:35].rstrip()
        template = random.choice(COMPARISON_TEMPLATES)
        query = template.format(
            title1=title1, title2=title2,
            brand1=p1["brand"], brand2=p2["brand"],
            en_type=en, zh_type=zh,
            price1=p1["price"], price2=p2["price"],
        )

        cases.append({
            "expected_intent":   "COMPARISON",
            "query":             query,
            "expected_skus":     [p1["id"], p2["id"]],
            "expected_keywords": [en, p1["brand"].lower(), p2["brand"].lower()],
            "product_ref": [
                {"title": p1["title"], "brand": p1["brand"],
                 "price": p1["price"], "rating": p1["rating"]},
                {"title": p2["title"], "brand": p2["brand"],
                 "price": p2["price"], "rating": p2["rating"]},
            ],
            "notes": f"商品对比，预期同时召回两款{zh}产品",
        })

    return cases[:n]


# ═══════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════

INTENT_TARGETS = {
    "CHITCHAT":       8,
    "PRODUCT_SEARCH": 16,
    "FUZZY_SEARCH":   16,
    "COMPARISON":     10,
}

def main():
    out_path = Path(__file__).parent / "eval_dataset.json"

    print("=" * 56)
    print("  Shopping Agent — Eval Dataset Generator")
    print("=" * 56)

    # 1. 拉取数据
    print("\n📦 连接数据库，拉取商品…")
    db = SessionLocal()
    try:
        products = fetch_products(db)
    finally:
        db.close()
    print(f"   ✅ 拉取 {len(products)} 条有效商品")

    # 分组统计
    grouped = group_by_type(products)
    print(f"   识别产品类型 {len(grouped)} 种：{list(grouped.keys())[:8]}…")

    # 2. 生成各意图用例
    print("\n✍️  生成测试用例…")

    chitchat   = gen_chitchat(INTENT_TARGETS["CHITCHAT"])
    ps_cases   = gen_product_search(products, INTENT_TARGETS["PRODUCT_SEARCH"])
    fuzzy      = gen_fuzzy_search(products,   INTENT_TARGETS["FUZZY_SEARCH"])
    comparison = gen_comparison(products,     INTENT_TARGETS["COMPARISON"])

    all_cases  = chitchat + ps_cases + fuzzy + comparison
    random.shuffle(all_cases)          # 打乱顺序

    # 补足：如果某意图生成不足，打印警告
    intent_counts: Dict[str, int] = {}
    for c in all_cases:
        k = c["expected_intent"]
        intent_counts[k] = intent_counts.get(k, 0) + 1

    for intent, target in INTENT_TARGETS.items():
        actual = intent_counts.get(intent, 0)
        status = "✅" if actual >= target else f"⚠️  不足（{actual}/{target}）"
        print(f"   {status}  {intent:20s}: {actual} 条")

    # 如果总数不足 50，用随机模糊查询补足
    if len(all_cases) < 50:
        deficit = 50 - len(all_cases)
        print(f"\n   ⚠️  总量不足，追加 {deficit} 条 FUZZY_SEARCH 补足…")
        extra = gen_fuzzy_search(products, deficit + 5)
        all_cases += extra[: deficit]

    all_cases = all_cases[:50]          # 严格截取 50 条

    # 编号
    for i, c in enumerate(all_cases, 1):
        c["id"] = i

    # 3. 保存
    payload = {
        "meta": {
            "version":              "1.0",
            "total":                len(all_cases),
            "intent_distribution":  intent_counts,
            "generated_at":         datetime.now().isoformat(timespec="seconds"),
            "description": (
                "Shopping Agent 评测数据集。覆盖 4 种意图："
                "CHITCHAT / PRODUCT_SEARCH / FUZZY_SEARCH / COMPARISON。"
                "expected_skus 为 Amazon 数据库中的真实 product_id。"
            ),
            "scoring_guide": {
                "hit@1":  "expected_skus[0] 在检索结果 Top-1 中",
                "hit@3":  "expected_skus 中任意一个在 Top-3 中",
                "hit@10": "expected_skus 中任意一个在 Top-10 中",
                "intent_acc": "预测意图 == expected_intent 的比例",
            },
        },
        "cases": all_cases,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存 → {out_path}")
    print(f"   文件大小：{out_path.stat().st_size / 1024:.1f} KB")

    # 4. 打印每个意图的示例
    print("\n" + "─" * 56)
    print("  示例（每种意图各 1 条）")
    print("─" * 56)
    shown = set()
    for c in all_cases:
        intent = c["expected_intent"]
        if intent in shown:
            continue
        shown.add(intent)
        print(f"\n  ▶ [{intent}]  #{c['id']}")
        print(f"    query         : {c['query']}")
        print(f"    expected_skus : {c['expected_skus'][:2]}")
        print(f"    keywords      : {c.get('expected_keywords', [])}")
        if len(shown) == 4:
            break
    print()


if __name__ == "__main__":
    main()
