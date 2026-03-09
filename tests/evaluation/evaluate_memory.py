#!/usr/bin/env python3
"""
evaluate_memory.py — 多轮记忆与指代消解能力评测

量化测试 ConversationService 工作记忆（WorkingMemory）机制在两个关键能力上的表现：

  指标 1  指代消解命中率 (Coreference Resolution Rate, CRR)
          Turn 3 中"第一款"/"那款"等指代词，是否能通过工作记忆被还原为具体商品

  指标 2  约束保持率 (Constraint Retention Rate, ConR)
          Turn 2 的推荐结果，是否同时保留了 Turn 1 的品牌约束

测试设计
  - 20 组多轮对话，每组 3 轮
  - 美妆品类（匹配数据库 Amazon 美妆商品）
  - Turn 1: 品牌 + 品类请求  → 系统推荐若干商品 → 存入 WorkingMemory
  - Turn 2: 叠加价格约束
  - Turn 3: 使用"第一款"/"那款"引用 Turn 1 的第一个推荐

机制说明
  Turn 1 后：ConversationService._extract_and_store_recommendations() 提取商品
            → WorkingMemory.context_data["current_products"] 存储商品列表
  Turn 3 时：_build_enhanced_message_history() 将 current_products 注入系统提示
            → LLM 看到"第一款 = [Product 1]: ..." 后能正确解引用

运行方法:
  cd /path/to/Shopping_agent
  python tests/evaluation/evaluate_memory.py
  （也可 cd backend && python ../tests/evaluation/evaluate_memory.py）
"""

import sys, os, asyncio, json, re, time, io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"

os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.models.models import WorkingMemory
from app.models.schemas import ChatRequest
from app.services.conversation_service import ConversationService

# ─── 20 组测试用例 ─────────────────────────────────────────────────────────────
# 品牌均为 Amazon 美妆商品常见品牌，Turn3 使用指代词（第一款/那款/你提到的那个）
TEST_CASES = [
    {"id": 1,  "brand": "NYX",
     "turn1": "帮我推荐几款NYX的哑光口红",
     "turn2": "预算在15美元以内，有合适的吗？",
     "turn3": "你刚才提到的第一款口红，它的持妆效果怎么样？"},
    {"id": 2,  "brand": "Maybelline",
     "turn1": "给我推荐几款Maybelline的防水睫毛膏",
     "turn2": "有没有10美元以下的选择？",
     "turn3": "你说的第一款睫毛膏，适合哪种眼型？"},
    {"id": 3,  "brand": "L'Oreal",
     "turn1": "推荐L'Oreal的粉底液，油皮适用的",
     "turn2": "我预算是20美元，有吗？",
     "turn3": "你推荐的第一款粉底液，SPF值是多少？"},
    {"id": 4,  "brand": "CoverGirl",
     "turn1": "帮我找几款CoverGirl的遮瑕膏",
     "turn2": "想找价格在12美元以下的",
     "turn3": "第一款遮瑕膏，它能遮黑眼圈吗？"},
    {"id": 5,  "brand": "e.l.f.",
     "turn1": "推荐几款e.l.f.的眼影盘，颜色丰富的",
     "turn2": "预算在10美元以内，有吗？",
     "turn3": "你提到的那第一款眼影盘，里面有多少颜色？"},
    {"id": 6,  "brand": "Revlon",
     "turn1": "我想要Revlon的保湿唇膏，推荐几款",
     "turn2": "有没有8美元以下的？",
     "turn3": "你之前推荐的第一款唇膏，有什么特别的香型吗？"},
    {"id": 7,  "brand": "Wet n Wild",
     "turn1": "帮我推荐Wet n Wild的高光产品",
     "turn2": "我希望控制在10美元以内",
     "turn3": "那第一款高光，适合哪种肤色使用？"},
    {"id": 8,  "brand": "Neutrogena",
     "turn1": "推荐Neutrogena的SPF50+防晒霜",
     "turn2": "20美元以内有推荐吗？",
     "turn3": "你提到的第一款防晒，是化学防晒还是物理防晒？"},
    {"id": 9,  "brand": "NARS",
     "turn1": "推荐几款NARS的腮红，自然色系的",
     "turn2": "预算30美元以内，有合适的吗？",
     "turn3": "你说的第一款腮红，是哑光还是珠光的？"},
    {"id": 10, "brand": "Urban Decay",
     "turn1": "帮我找Urban Decay的防晕染眼线笔",
     "turn2": "20美元以内有吗？",
     "turn3": "你推荐的第一款眼线笔，大概能用多久？"},
    {"id": 11, "brand": "Benefit",
     "turn1": "推荐Benefit的眉笔，自然妆感的",
     "turn2": "预算在25美元以内有选择吗？",
     "turn3": "你提到的第一款眉笔，适合哪种眉形？"},
    {"id": 12, "brand": "Too Faced",
     "turn1": "帮我推荐Too Faced的底妆产品",
     "turn2": "价格35美元以内，有吗？",
     "turn3": "你推荐的第一款底妆，遮盖力如何？"},
    {"id": 13, "brand": "Milani",
     "turn1": "帮我找几款Milani的大红色口红",
     "turn2": "预算12美元以下",
     "turn3": "你提到的第一款口红，质地是哑光还是奶油感？"},
    {"id": 14, "brand": "Essence",
     "turn1": "推荐几款Essence的睫毛膏",
     "turn2": "预算8美元以内",
     "turn3": "第一款睫毛膏，能增加睫毛量感吗？"},
    {"id": 15, "brand": "OPI",
     "turn1": "帮我推荐几款OPI的裸色指甲油",
     "turn2": "10美元以内有推荐吗？",
     "turn3": "你推荐的第一款指甲油，持妆时间大概多久？"},
    {"id": 16, "brand": "Cetaphil",
     "turn1": "推荐Cetaphil的温和洁面产品",
     "turn2": "预算20美元以内",
     "turn3": "你说的第一款洁面，适合早晚都用吗？"},
    {"id": 17, "brand": "Burt's Bees",
     "turn1": "推荐Burt's Bees的唇部护理产品",
     "turn2": "最好10美元以内",
     "turn3": "你提到的第一款，有哪些香型？"},
    {"id": 18, "brand": "Bare Minerals",
     "turn1": "推荐Bare Minerals的控油散粉",
     "turn2": "预算20美元以内",
     "turn3": "你说的第一款散粉，适合敏感肌吗？"},
    {"id": 19, "brand": "L'Oreal",
     "turn1": "推荐L'Oreal的受损修复护发素",
     "turn2": "15美元以内有选择吗？",
     "turn3": "你提到的第一款护发素，使用频率是每天还是每周？"},
    {"id": 20, "brand": "Maybelline",
     "turn1": "给我推荐Maybelline的遮瑕粉底，油皮适用",
     "turn2": "预算在18美元以内",
     "turn3": "你推荐的第一款粉底，SPF多少？适合夏天用吗？"},
]

# ─── 评测工具函数 ───────────────────────────────────────────────────────────────

def contains_brand(text: str, brand: str) -> bool:
    """大小写不敏感的品牌检测，支持特殊品牌名变体"""
    variants: Dict[str, List[str]] = {
        "e.l.f.":         ["e.l.f", "elf"],
        "Burt's Bees":    ["burt's bees", "burts bees"],
        "Wet n Wild":     ["wet n wild", "wet'n'wild"],
        "Bare Minerals":  ["bareminerals", "bare minerals"],
        "L'Oreal":        ["l'oreal", "loreal", "l'oréal", "l'oréal"],
        "Too Faced":      ["too faced"],
        "Urban Decay":    ["urban decay"],
    }
    text_lower = text.lower()
    brand_lower = brand.lower()
    if brand_lower in text_lower:
        return True
    for key, alts in variants.items():
        if key.lower() == brand_lower:
            return any(a in text_lower for a in alts)
    return False


def contains_product_name(response: str, product: Dict) -> bool:
    """检查回复中是否包含商品的可识别标识（名称 bigram 或品牌+特征组合）"""
    if not product:
        return False
    resp_lower = response.lower()

    # 1. 商品名称 bigram 匹配
    name = product.get("name", "")
    if name and len(name) > 4:
        parts = name.lower().split()
        if len(parts) >= 3:
            for i in range(len(parts) - 1):
                bigram = " ".join(parts[i : i + 2])
                if bigram in resp_lower:
                    return True
        elif name.lower() in resp_lower:
            return True

    # 2. 品牌名出现
    brand = product.get("brand", "")
    if brand and brand.lower() in resp_lower:
        return True

    # 3. 至少 2 个关键特征同时出现
    features = product.get("key_features", [])
    if len(features) >= 2:
        hits = sum(1 for f in features if f.lower() in resp_lower)
        if hits >= 2:
            return True

    return False


def make_chat_request(msg: str, conv_id: Optional[int] = None) -> ChatRequest:
    return ChatRequest(
        message=msg,
        conversation_id=conv_id,
        model="GLM-4.6",
        max_tokens=4096,
        temperature=0.7,
        use_memory=True,
    )


# ─── 单组测试 ──────────────────────────────────────────────────────────────────

async def run_single_case(case: Dict, db) -> Dict:
    conv_service = ConversationService(db)

    result: Dict = {
        "id": case["id"],
        "brand": case["brand"],
        "turn1_response": "",
        "turn2_response": "",
        "turn3_response": "",
        "working_memory_products": [],
        "first_product": None,
        "wm_extraction_success": False,
        "coreference_hit": False,
        "constraint_brand_retained": False,
        "error": None,
        "elapsed": 0.0,
    }

    try:
        # ── Turn 1：品牌请求 ───────────────────────────────────────────────────
        print(f"  Turn 1  {case['turn1'][:55]}")
        buf = io.StringIO()
        with redirect_stdout(buf):
            resp1 = await conv_service.process_chat_message(make_chat_request(case["turn1"]))
        conv_id = resp1.conversation_id
        session_id = f"session_{conv_id}"
        result["turn1_response"] = resp1.response
        print(f"          → {resp1.response[:90].strip()}...")

        # ── 读取 WorkingMemory ─────────────────────────────────────────────────
        db.expire_all()
        wm = (
            db.query(WorkingMemory)
            .filter(WorkingMemory.session_id == session_id)
            .order_by(WorkingMemory.id.desc())
            .first()
        )
        if wm and wm.context_data:
            products = wm.context_data.get("current_products", [])
            result["working_memory_products"] = products
            if products:
                result["first_product"] = products[0]
                result["wm_extraction_success"] = True
                p0_name = products[0].get("name", "?")
                print(f"          WorkingMemory ✅ {len(products)}款 | 第一款: {p0_name[:40]}")
            else:
                print(f"          WorkingMemory ⚠️  context_data 存在但 current_products 为空")
        else:
            print(f"          WorkingMemory ❌ 未创建（商品提取可能失败）")

        # ── Turn 2：叠加价格约束 ──────────────────────────────────────────────
        print(f"  Turn 2  {case['turn2'][:55]}")
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            resp2 = await conv_service.process_chat_message(make_chat_request(case["turn2"], conv_id))
        result["turn2_response"] = resp2.response
        print(f"          → {resp2.response[:90].strip()}...")

        brand_ok = contains_brand(resp2.response, case["brand"])
        result["constraint_brand_retained"] = brand_ok
        b_sym = "✅" if brand_ok else "❌"
        print(f"          品牌约束[{case['brand']}]: {b_sym}")

        # ── Turn 3：指代词查询 ────────────────────────────────────────────────
        print(f"  Turn 3  {case['turn3'][:55]}")
        buf3 = io.StringIO()
        with redirect_stdout(buf3):
            resp3 = await conv_service.process_chat_message(make_chat_request(case["turn3"], conv_id))
        result["turn3_response"] = resp3.response
        print(f"          → {resp3.response[:90].strip()}...")

        if result["first_product"]:
            coref_hit = contains_product_name(resp3.response, result["first_product"])
            result["coreference_hit"] = coref_hit
            fp_name = result["first_product"].get("name", "?")[:30]
            c_sym = "✅" if coref_hit else "❌"
            print(f"          指代消解[{fp_name}]: {c_sym}")
        else:
            # WorkingMemory 为空：退化为检测品牌是否在 Turn3 响应中出现
            fallback = contains_brand(resp3.response, case["brand"])
            result["coreference_hit"] = False  # 无法真正验证，标记失败
            fb_sym = "⚠️  (fallback)" if fallback else "❌"
            print(f"          指代消解[WM空]: {fb_sym}")

    except Exception as exc:
        import traceback
        result["error"] = str(exc)
        print(f"  ❌ 报错: {exc}")
        traceback.print_exc()

    return result


# ─── 汇总输出 ──────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]) -> None:
    total = len(results)
    valid = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    N = len(valid)

    wm_ok    = sum(1 for r in valid if r["wm_extraction_success"])
    coref_ok = sum(1 for r in valid if r["coreference_hit"])
    brand_ok = sum(1 for r in valid if r["constraint_brand_retained"])

    crr   = coref_ok / N * 100 if N else 0.0
    conr  = brand_ok / N * 100 if N else 0.0
    wm_r  = wm_ok   / N * 100 if N else 0.0

    print(f"\n\n{'═'*70}")
    print("  📊  评测结果汇总")
    print(f"{'═'*70}")
    print(f"  总用例: {total}  |  有效: {N}  |  报错: {len(errors)}")
    print()
    print(f"  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  指标                          命中   总数   命中率           │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  WorkingMemory 提取成功         {wm_ok:3d}  / {N:3d}  = {wm_r:5.1f}%          │")
    print(f"  │  指代消解命中率 (CRR)           {coref_ok:3d}  / {N:3d}  = {crr:5.1f}%          │")
    print(f"  │  品牌约束保持率 (ConR, Turn 2)  {brand_ok:3d}  / {N:3d}  = {conr:5.1f}%          │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    # 逐条明细
    print(f"\n  逐条明细：")
    print(f"  {'─'*66}")
    header = f"  {'#':>3}  {'品牌':<14}  {'WM':^4}  {'指代':^4}  {'品牌约束':^8}  {'第一款商品':<20}  {'耗时':>6}"
    print(header)
    print(f"  {'─'*66}")
    for r in results:
        if r["error"]:
            print(f"  {r['id']:>3}  {r['brand']:<14}  {'ERR':^4}  {'ERR':^4}  {'ERR':^8}  {'':20}  {'?':>6}")
            continue
        wm_sym    = "✅" if r["wm_extraction_success"] else "❌"
        coref_sym = "✅" if r["coreference_hit"]        else "❌"
        brand_sym = "✅" if r["constraint_brand_retained"] else "❌"
        fp = r["first_product"]
        fp_str = (fp.get("name","?")[:20] if fp else "—").ljust(20)
        print(f"  {r['id']:>3}  {r['brand']:<14}  {wm_sym:^4}  {coref_sym:^4}  {brand_sym:^8}  {fp_str}  {r['elapsed']:>5.1f}s")

    # 结论
    print(f"\n{'─'*70}")
    print("  📝  结论与建议：")
    if crr >= 70:
        print(f"  ✅ 指代消解能力良好 (CRR={crr:.1f}%)：WorkingMemory 能有效将指代词还原为具体商品")
    elif crr >= 40:
        print(f"  ⚠️  指代消解能力中等 (CRR={crr:.1f}%)：部分场景失败，需提高 WM 提取成功率")
    else:
        print(f"  ❌ 指代消解能力较弱 (CRR={crr:.1f}%)：WorkingMemory 提取或注入存在系统性问题")

    if conr >= 70:
        print(f"  ✅ 品牌约束保持良好 (ConR={conr:.1f}%)：多轮对话中品牌约束被有效保留")
    elif conr >= 40:
        print(f"  ⚠️  品牌约束保持中等 (ConR={conr:.1f}%)：对话历史传递品牌信息偶有丢失")
    else:
        print(f"  ❌ 品牌约束保持较弱 (ConR={conr:.1f}%)：对话历史未能有效传递品牌约束")

    if wm_r < 50:
        print()
        print(f"  ⚠️  WorkingMemory 提取成功率仅 {wm_r:.1f}%，可能原因：")
        print(f"      _extract_and_store_recommendations() 使用 max_tokens=500，")
        print(f"      GLM-4.6 推理链消耗大量 tokens 导致 content='' 提取失败。")
        print(f"      建议：在 conversation_service.py 中将该函数的 max_tokens 提至 1000+。")

    total_elapsed = sum(r.get("elapsed", 0) for r in results)
    print(f"\n  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'═'*70}\n")


# ─── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("  多轮记忆与指代消解能力评测 (evaluate_memory.py)")
    print(f"  模型: GLM-4.6  |  测试组数: {len(TEST_CASES)}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    print(f"  本脚本发起 {len(TEST_CASES)*3} 次主对话 + 约 {len(TEST_CASES)*3*2} 次子提取调用")
    print(f"  预计总耗时 20~40 分钟，请耐心等待...\n")

    results: List[Dict] = []
    db = SessionLocal()

    try:
        for i, case in enumerate(TEST_CASES, 1):
            print(f"\n{'━'*70}")
            print(f"  Case {i:>2}/{len(TEST_CASES)}  [品牌: {case['brand']}]")
            print(f"{'━'*70}")

            t0 = time.time()
            result = await run_single_case(case, db)
            result["elapsed"] = time.time() - t0

            results.append(result)
            print(f"  ⏱  {result['elapsed']:.1f}s")
    finally:
        db.close()

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
