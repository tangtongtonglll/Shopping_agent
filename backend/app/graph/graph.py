"""
ShoppingAgent StateGraph 组装与对外调用接口

Graph 拓扑
----------
                START
                  │
            router_node           ← 意图识别 + 实体提取
           /      │       \
    search/compare  compare  chat
          │          │        │
   retriever_node    │   generator_node
          │          │        │
          └──►comparator_node─┘
                     │
               generator_node     ← 整合上下文，生成回复，评估置信度
                     │
          ┌──────────┴──────────┐
     needs_review            normal
          │                    │
   human_review_node          END   ← interrupt_before 在此暂停
          │
         END

SqliteSaver Checkpointing：
  - 每个节点执行后自动持久化 state，支持断点续跑
  - interrupt_before=["human_review_node"] 在低置信度 / 高价值商品时暂停

公开接口：
  graph    = build_graph(checkpointer)   # 返回编译好的 CompiledStateGraph
  response = await run(query, db, ...)   # 一次完整的对话调用
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .nodes import (
    comparator_node,
    generator_node,
    human_review_node,
    retriever_node,
    router_node,
)
from .state import DEFAULT_STATE, ShoppingAgentState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 条件路由函数                                                                  #
# --------------------------------------------------------------------------- #

def _route_from_router(state: ShoppingAgentState) -> str:
    """
    router_node 完成后，根据意图选择下一跳。

    - search / compare → retriever_node  （两者都需要检索，再由 retriever 后续分流）
    - chat             → generator_node  （无需检索，直接生成）
    """
    intent = state.get("intent", "chat")
    if intent in ("search", "compare"):
        return "retriever_node"
    return "generator_node"


def _route_from_retriever(state: ShoppingAgentState) -> str:
    """
    retriever_node 完成后，根据意图决定是否需要对比节点。

    - compare → comparator_node  （需要格式化对比矩阵）
    - search  → generator_node   （直接生成推荐回复）
    """
    intent = state.get("intent", "chat")
    if intent == "compare":
        return "comparator_node"
    return "generator_node"


def _route_from_generator(state: ShoppingAgentState) -> str:
    """
    generator_node 完成后，判断是否需要人工审核。

    触发条件（与 human_review_node 内部逻辑保持一致）：
    - confidence_score < 0.6
    - 或实体中包含高价值商品（预算 > 5000 元 / 特定品牌关键词）
    """
    confidence  = state.get("confidence_score", 1.0)
    entities    = state.get("entities", {})
    budget      = entities.get("budget") or 0

    HIGH_VALUE_KEYWORDS = {"苹果", "Apple", "iPhone", "MacBook", "华为", "Mate", "奢侈"}
    products_str = " ".join(entities.get("products", []))
    is_high_value = (
        budget > 5000
        or any(kw in products_str for kw in HIGH_VALUE_KEYWORDS)
    )

    if confidence < 0.6 or is_high_value:
        logger.info(
            f"[Graph] 路由至 human_review_node "
            f"(confidence={confidence:.2f}, high_value={is_high_value})"
        )
        return "human_review_node"

    return END


# --------------------------------------------------------------------------- #
# 图的构建与编译                                                                #
# --------------------------------------------------------------------------- #

def build_graph(checkpointer=None):
    """
    构建并编译 ShoppingAgent StateGraph。

    Args:
        checkpointer: LangGraph Checkpointer 实例（SqliteSaver / AsyncSqliteSaver）。
                      传入时自动启用 interrupt_before=["human_review_node"]。
                      传入 None 时图无持久化，适用于无状态单次调用或单元测试。

    Returns:
        CompiledStateGraph
    """
    builder = StateGraph(ShoppingAgentState)

    # ── 注册节点 ─────────────────────────────────────────────────────────────
    builder.add_node("router_node",       router_node)
    builder.add_node("retriever_node",    retriever_node)
    builder.add_node("comparator_node",   comparator_node)
    builder.add_node("generator_node",    generator_node)
    builder.add_node("human_review_node", human_review_node)

    # ── 固定边 ───────────────────────────────────────────────────────────────
    builder.add_edge(START,              "router_node")
    builder.add_edge("comparator_node",  "generator_node")
    builder.add_edge("human_review_node", END)

    # ── 条件边 ───────────────────────────────────────────────────────────────
    builder.add_conditional_edges("router_node",    _route_from_router)
    builder.add_conditional_edges("retriever_node", _route_from_retriever)
    builder.add_conditional_edges("generator_node", _route_from_generator)

    # ── 编译 ─────────────────────────────────────────────────────────────────
    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"]     = checkpointer
        compile_kwargs["interrupt_before"] = ["human_review_node"]
        logger.info("[Graph] 启用 SqliteSaver checkpointer + interrupt_before")

    compiled = builder.compile(**compile_kwargs)
    logger.info("[Graph] StateGraph 编译完成")
    return compiled


# --------------------------------------------------------------------------- #
# SqliteSaver 工厂                                                              #
# --------------------------------------------------------------------------- #

_CHECKPOINT_DB = os.path.join(
    os.path.dirname(__file__), "../../../../", "shopping_agent_checkpoints.db"
)
_CHECKPOINT_DB = os.path.normpath(_CHECKPOINT_DB)


def get_sqlite_checkpointer(db_path: str = _CHECKPOINT_DB):
    """
    返回 SqliteSaver 上下文管理器。

    推荐在 with 语句中使用：
        with get_sqlite_checkpointer() as cp:
            graph = build_graph(cp)
            await graph.ainvoke(...)

    若 langgraph-checkpoint-sqlite 未安装，返回 None（图无持久化）。
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(db_path)
    except ImportError:
        logger.warning(
            "[Graph] langgraph-checkpoint-sqlite 未安装，Checkpointer 不可用。"
            "请运行: pip install langgraph-checkpoint-sqlite"
        )
        return None


# --------------------------------------------------------------------------- #
# 对外调用接口                                                                  #
# --------------------------------------------------------------------------- #

async def run(
    query: str,
    db,
    *,
    thread_id: str                        = "default",
    knowledge_base_ids: Optional[List[int]] = None,
    checkpointer                           = None,
) -> Dict[str, Any]:
    """
    购物导购 Agent 的统一异步调用接口。

    Args:
        query:               用户输入的自然语言查询。
        db:                  SQLAlchemy Session（由调用方管理生命周期）。
        thread_id:           对话线程 ID，相同 ID 的多次调用共享历史（需 checkpointer）。
        knowledge_base_ids:  限定检索的知识库 ID 列表；None 表示全库检索。
        checkpointer:        Checkpointer 实例；None 时无持久化（单次无状态调用）。

    Returns:
        Dict 包含完整 ShoppingAgentState 字段，重点字段：
            final_response   (str)   ← 直接呈现给用户的回复
            intent           (str)   ← 识别的意图
            confidence_score (float) ← 置信度
            retrieved_docs   (list)  ← 检索到的文档
    """
    graph = build_graph(checkpointer)

    initial_state: ShoppingAgentState = {
        **DEFAULT_STATE,
        "messages": [HumanMessage(content=query)],
    }

    config = {
        "configurable": {
            "thread_id":          thread_id,
            "db":                 db,
            "knowledge_base_ids": knowledge_base_ids,
        }
    }

    logger.info(f"[Graph] 开始执行 | thread_id={thread_id} | query='{query[:60]}'")

    result: ShoppingAgentState = await graph.ainvoke(initial_state, config=config)

    logger.info(
        f"[Graph] 执行完成 | intent={result.get('intent')} "
        f"confidence={result.get('confidence_score', 0):.2f}"
    )
    return result
