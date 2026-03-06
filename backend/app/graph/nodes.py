"""
LangGraph 节点函数定义

节点规范
--------
- 入参: (state: ShoppingAgentState, config: RunnableConfig)
  * state  : 当前完整状态快照（只读，通过返回 dict 来更新）
  * config : 运行时配置，db / knowledge_base_ids 从 config["configurable"] 获取
- 出参: Dict[str, Any]  ← 仅包含本节点需要更新的字段，LangGraph 负责 merge

DB 注入方式（调用方）:
    graph.invoke(
        initial_state,
        config={"configurable": {"db": db_session, "knowledge_base_ids": [1, 2]}},
    )
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import RunnableConfig

from .state import ShoppingAgentState
from ..services.hybrid_search_service import get_hybrid_search_service
from ..services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 内部工具                                                                     #
# --------------------------------------------------------------------------- #

def _last_human_text(state: ShoppingAgentState) -> str:
    """从对话历史中取出最后一条用户消息的文本内容。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else ""
    return ""


def _get_db(config: RunnableConfig):
    """从 config.configurable 中安全取出 SQLAlchemy Session。"""
    return (config or {}).get("configurable", {}).get("db", None)


def _get_kb_ids(config: RunnableConfig) -> Optional[List[int]]:
    """从 config.configurable 中安全取出 knowledge_base_ids。"""
    return (config or {}).get("configurable", {}).get("knowledge_base_ids", None)


def _sigmoid(x: float) -> float:
    """将实数映射到 (0, 1)，用于将 reranker logit 转换为置信度。"""
    return 1.0 / (1.0 + math.exp(-x))


def _estimate_confidence(retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    根据重排分数估算置信度。

    bge-reranker-v2-m3 输出的是原始 logit：
      - logit > 0  表示相关
      - logit < 0  表示不相关
    经 sigmoid 后映射至 (0, 1)，取 top-3 均值作为置信度。
    """
    if not retrieved_docs:
        return 0.3  # 无任何检索结果，置信度低

    scores = [_sigmoid(d.get("rerank_score", 0.0)) for d in retrieved_docs[:3]]
    return round(sum(scores) / len(scores), 4)


# --------------------------------------------------------------------------- #
# Node 1 : router_node                                                         #
# --------------------------------------------------------------------------- #

async def router_node(
    state: ShoppingAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    路由节点：分析用户意图，提取关键实体。

    职责
    ----
    - 读取最后一条用户消息
    - 调用 LLM，输出结构化 JSON（intent + entities）
    - 将结果写入 state["intent"] 和 state["entities"]

    LLM 输出格式（强制 JSON）:
    {
        "intent": "chat" | "search" | "compare",
        "entities": {
            "products":   ["商品名A", "商品名B"],
            "brands":     ["品牌名"],
            "category":   "品类",
            "budget":     数字或null,
            "attributes": ["关注的属性，如电池、拍照"]
        }
    }
    """
    user_text = _last_human_text(state)
    logger.info(f"[router_node] 用户输入: '{user_text}'")

    system_prompt = (
        "你是一个电商导购意图识别模型。\n"
        "请分析用户的输入，判断其意图，并提取相关实体。\n\n"
        "意图类别:\n"
        "  - chat    : 普通闲聊、打招呼、通用问题\n"
        "  - search  : 希望查找、推荐某种商品\n"
        "  - compare : 希望对比两个或多个具体商品\n\n"
        "请严格按照如下 JSON 格式输出，不要添加任何额外文字:\n"
        "{\n"
        '  "intent": "<chat|search|compare>",\n'
        '  "entities": {\n'
        '    "products":   [],\n'
        '    "brands":     [],\n'
        '    "category":   "",\n'
        '    "budget":     null,\n'
        '    "attributes": []\n'
        "  }\n"
        "}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]

    # 默认值，LLM 失败时保底
    intent   = "chat"
    entities: Dict[str, Any] = {
        "products": [], "brands": [], "category": "",
        "budget": None, "attributes": [],
    }

    try:
        llm = get_llm_service()
        resp = await llm.chat_completion(messages, temperature=0.0, max_tokens=512)
        raw  = resp.get("content", "").strip()

        # 兼容 LLM 在 JSON 前后附加 markdown 代码块的情况
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed   = json.loads(raw)
        intent   = parsed.get("intent", "chat")
        entities = parsed.get("entities", entities)
        logger.info(f"[router_node] 识别意图={intent}, 实体={entities}")

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(f"[router_node] 意图识别失败，回退到 chat: {exc}")

    return {
        "intent":   intent,
        "entities": entities,
    }


# --------------------------------------------------------------------------- #
# Node 2 : retriever_node                                                      #
# --------------------------------------------------------------------------- #

async def retriever_node(
    state: ShoppingAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    检索节点：调用 advanced_hybrid_search 执行混合检索。

    职责
    ----
    - 仅在 intent == "search" 或 "compare" 时激活（调用方通过条件边控制）
    - 从 entities 中拼接检索 query
    - 调用 HybridSearchService.advanced_hybrid_search，结果写入 retrieved_docs

    触发条件（在 graph.py 的边定义中设置）:
        intent in {"search", "compare"}
    """
    db     = _get_db(config)
    kb_ids = _get_kb_ids(config)

    if db is None:
        logger.error("[retriever_node] config 中未提供 db，跳过检索。")
        return {"retrieved_docs": []}

    # 从实体拼接检索 query
    entities = state.get("entities", {})
    parts: List[str] = []
    if entities.get("products"):
        parts.extend(entities["products"])
    if entities.get("brands"):
        parts.extend(entities["brands"])
    if entities.get("category"):
        parts.append(entities["category"])
    if entities.get("attributes"):
        parts.extend(entities["attributes"])
    if entities.get("budget"):
        parts.append(f"预算{entities['budget']}元")

    # 若实体为空，则直接使用用户原始输入
    query = " ".join(parts) if parts else _last_human_text(state)
    logger.info(f"[retriever_node] 检索 query='{query}' kb_ids={kb_ids}")

    try:
        svc  = get_hybrid_search_service()
        docs = svc.advanced_hybrid_search(
            query=query,
            db=db,
            knowledge_base_ids=kb_ids,
            bm25_top_k=20,
            dense_top_k=20,
            final_top_k=5,
        )
        logger.info(f"[retriever_node] 检索到 {len(docs)} 条文档")
    except Exception as exc:
        logger.error(f"[retriever_node] 检索失败: {exc}")
        docs = []

    return {"retrieved_docs": docs}


# --------------------------------------------------------------------------- #
# Node 3 : comparator_node                                                     #
# --------------------------------------------------------------------------- #

async def comparator_node(
    state: ShoppingAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    对比节点：专门处理多商品对比逻辑。

    职责
    ----
    - 仅在 intent == "compare" 时激活
    - 将 retrieved_docs 中的内容送入 LLM，生成结构化对比矩阵
    - 结果写入 state["comparison_data"]

    comparison_data 格式:
    {
        "products":   ["商品A", "商品B"],
        "dimensions": {
            "维度名": {"商品A": "值", "商品B": "值"},
            ...
        },
        "recommendation": "综合建议文字..."
    }
    """
    entities      = state.get("entities", {})
    retrieved_docs = state.get("retrieved_docs", [])
    product_names  = entities.get("products", [])

    if not retrieved_docs:
        logger.warning("[comparator_node] 无检索文档，无法生成对比数据。")
        return {"comparison_data": None}

    # 将检索文档内容拼接为上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[文档{i}] {doc.get('content', '')}")
    context = "\n\n".join(context_parts)

    products_hint = "、".join(product_names) if product_names else "用户提及的商品"

    prompt = (
        f"用户想对比以下商品: {products_hint}\n\n"
        "以下是检索到的相关产品信息：\n"
        f"{context}\n\n"
        "请从以上信息中提取关键对比维度（如价格、处理器、内存、电池、拍照、重量等），"
        "输出如下 JSON 格式的对比数据（确保 JSON 合法）:\n"
        "{\n"
        '  "products": ["商品A", "商品B"],\n'
        '  "dimensions": {\n'
        '    "价格":   {"商品A": "...", "商品B": "..."},\n'
        '    "处理器": {"商品A": "...", "商品B": "..."}\n'
        "  },\n"
        '  "recommendation": "综合建议..."\n'
        "}"
    )

    comparison_data: Optional[Dict[str, Any]] = None

    try:
        llm  = get_llm_service()
        resp = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        raw = resp.get("content", "").strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        comparison_data = json.loads(raw)
        logger.info(f"[comparator_node] 对比数据生成完成，维度数={len(comparison_data.get('dimensions', {}))}")

    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"[comparator_node] 对比数据生成失败: {exc}")
        # 降级：将原始文档内容保存为非结构化对比
        comparison_data = {
            "products":       product_names,
            "dimensions":     {},
            "recommendation": context[:500],  # 截取部分文档内容作为兜底
        }

    return {"comparison_data": comparison_data}


# --------------------------------------------------------------------------- #
# Node 4 : generator_node                                                      #
# --------------------------------------------------------------------------- #

async def generator_node(
    state: ShoppingAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    生成节点：整合所有上下文，生成最终导购回复，并评估置信度。

    职责
    ----
    - 读取 retrieved_docs、comparison_data、entities、intent
    - 拼接结构化 prompt，调用 LLM 生成导购回复
    - 计算 confidence_score（基于 rerank_score 的 sigmoid 均值）
    - 将回复以 AIMessage 追加到 messages，并写入 final_response
    """
    intent          = state.get("intent", "chat")
    entities        = state.get("entities", {})
    retrieved_docs  = state.get("retrieved_docs", [])
    comparison_data = state.get("comparison_data")
    user_text       = _last_human_text(state)

    # ── 构建检索上下文 ──────────────────────────────────────────────────────
    context_lines: List[str] = []

    if comparison_data:
        # 对比场景：优先展示结构化对比数据
        context_lines.append("=== 商品对比数据 ===")
        for dim, vals in comparison_data.get("dimensions", {}).items():
            row = "  |  ".join(f"{prod}: {val}" for prod, val in vals.items())
            context_lines.append(f"【{dim}】{row}")
        if comparison_data.get("recommendation"):
            context_lines.append(f"\n初步建议: {comparison_data['recommendation']}")

    elif retrieved_docs:
        # 搜索场景：展示检索到的文档片段
        context_lines.append("=== 检索到的相关产品信息 ===")
        for i, doc in enumerate(retrieved_docs, 1):
            score_hint = f"(相关度: {round(doc.get('rerank_score', 0.0), 2)})"
            context_lines.append(f"[{i}] {doc.get('content', '')} {score_hint}")

    context_text = "\n".join(context_lines) if context_lines else "（暂无检索到相关商品信息）"

    # ── 构建 prompt ──────────────────────────────────────────────────────────
    system_msg = (
        "你是一位专业、亲切的电商导购助手。\n"
        "请基于提供的商品信息为用户给出精准、实用的购买建议。\n"
        "要求:\n"
        "  1. 语言简洁，重点突出，避免废话\n"
        "  2. 若有多个商品，明确指出推荐顺序及理由\n"
        "  3. 如信息不足，主动说明并引导用户补充需求\n"
        "  4. 不要编造不在参考信息中的规格数据"
    )

    # 拼接实体摘要
    budget_hint = f"（预算约 {entities['budget']} 元）" if entities.get("budget") else ""
    attrs_hint  = (
        f"，用户关注: {', '.join(entities['attributes'])}"
        if entities.get("attributes") else ""
    )

    user_msg = (
        f"用户问题: {user_text}\n"
        f"意图类型: {intent}{budget_hint}{attrs_hint}\n\n"
        f"{context_text}\n\n"
        "请根据以上信息，给出专业的导购回复："
    )

    messages_for_llm = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    final_response = "抱歉，我暂时无法生成回复，请稍后重试。"

    try:
        llm  = get_llm_service()
        resp = await llm.chat_completion(
            messages=messages_for_llm,
            temperature=0.5,
            max_tokens=1024,
        )
        final_response = resp.get("content", final_response).strip()
        logger.info(f"[generator_node] 回复生成完成，字数={len(final_response)}")

    except Exception as exc:
        logger.error(f"[generator_node] LLM 调用失败: {exc}")

    # ── 置信度评估 ───────────────────────────────────────────────────────────
    confidence = _estimate_confidence(retrieved_docs)
    # 无任何检索支撑的纯闲聊，置信度固定 0.9（不依赖文档）
    if intent == "chat" and not retrieved_docs:
        confidence = 0.9

    logger.info(f"[generator_node] confidence_score={confidence}")

    return {
        # add_messages reducer 会将新消息追加到历史
        "messages":        [AIMessage(content=final_response)],
        "final_response":  final_response,
        "confidence_score": confidence,
    }


# --------------------------------------------------------------------------- #
# Node 5 : human_review_node                                                   #
# --------------------------------------------------------------------------- #

def human_review_node(
    state: ShoppingAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    人工审核节点（占位 + 拦截逻辑）。

    职责
    ----
    - 检测是否需要人工介入（置信度 < 0.6，或涉及高价值商品）
    - 需要介入时，通过 NodeInterrupt 暂停图执行，等待外部恢复
    - 恢复后（人工审核完成），透传 state，不修改内容

    启用拦截的方式（在 graph.py 中编译图时设置）:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review_node"],
        )

    高价值商品判断规则（可根据业务调整）:
        - entities["budget"] > 5000 元
        - 或 entities["products"] 中含高价类目关键词
    """
    confidence = state.get("confidence_score", 1.0)
    entities   = state.get("entities", {})
    budget     = entities.get("budget") or 0

    HIGH_VALUE_KEYWORDS = {"苹果", "Apple", "iPhone", "MacBook", "华为", "Mate", "奢侈"}
    products_str = " ".join(entities.get("products", []))
    is_high_value = budget > 5000 or any(kw in products_str for kw in HIGH_VALUE_KEYWORDS)

    needs_review = confidence < 0.6 or is_high_value

    if needs_review:
        reason = []
        if confidence < 0.6:
            reason.append(f"置信度偏低({confidence:.2f})")
        if is_high_value:
            reason.append(f"高价值商品(预算={budget}元)")

        logger.warning(f"[human_review_node] 触发人工审核: {', '.join(reason)}")

        try:
            # LangGraph >= 0.2 推荐方式：暂停图执行，等待外部恢复
            from langgraph.errors import NodeInterrupt
            raise NodeInterrupt(
                f"需要人工审核 — {', '.join(reason)}。"
                f"当前回复草稿: {state.get('final_response', '')[:100]}..."
            )
        except ImportError:
            # 降级：仅记录日志，继续执行
            logger.error("[human_review_node] NodeInterrupt 不可用，跳过拦截。请升级 langgraph>=0.2。")

    logger.info("[human_review_node] 无需人工审核，透传 state。")
    # 不需要审核时返回空 dict，state 保持不变
    return {}
