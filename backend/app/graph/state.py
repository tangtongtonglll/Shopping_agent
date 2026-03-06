"""
LangGraph 共享状态定义

所有节点通过读写 ShoppingAgentState 进行协调。
每个节点返回 dict，LangGraph 自动将其 merge 进当前 state。

DB Session 注入方式（不放入 state，避免序列化问题）：
    graph.invoke(
        initial_state,
        config={"configurable": {"db": db_session, "knowledge_base_ids": [1, 2]}},
    )
"""

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ShoppingAgentState(TypedDict):
    """
    购物导购 Agent 的全局共享状态。

    字段说明
    --------
    messages
        完整对话历史（HumanMessage / AIMessage / SystemMessage）。
        使用 add_messages reducer：节点返回的新消息会被追加，而非覆盖。

    intent
        路由节点识别出的用户意图：
        - "chat"    : 普通闲聊 / 通用问答
        - "search"  : 商品搜索 / 推荐
        - "compare" : 多商品对比

    entities
        从用户输入中提取的结构化实体，格式示例：
        {
            "products":   ["iPhone 15 Pro", "Pixel 8 Pro"],
            "brands":     ["Apple", "Google"],
            "category":   "手机",
            "budget":     5000,          # 元，None 表示未提及
            "attributes": ["电池", "拍照"]
        }

    retrieved_docs
        hybrid_search_service.advanced_hybrid_search 的返回值列表，每项包含：
        chunk_id, content, document_name, retrieval_source, rerank_score 等。

    comparison_data
        comparator_node 格式化后的多商品对比结构，格式示例：
        {
            "products": ["A", "B"],
            "dimensions": {
                "价格":  {"A": "6999", "B": "5999"},
                "处理器": {"A": "A17", "B": "Tensor G3"},
                ...
            },
            "recommendation": "综合来看 A 更适合重度用户..."
        }

    final_response
        generator_node 生成的最终导购回复文本，直接返回给用户。

    confidence_score
        generator_node 对本次回复的置信度评估，范围 [0.0, 1.0]。
        低于 0.6 时，human_review_node 会触发人工审核拦截。
    """

    # ── 对话层 ──────────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── 路由层（router_node 写入）───────────────────────────────────────────
    intent:   str
    entities: Dict[str, Any]

    # ── 检索层（retriever_node 写入）────────────────────────────────────────
    retrieved_docs: List[Dict[str, Any]]

    # ── 对比层（comparator_node 写入）───────────────────────────────────────
    comparison_data: Optional[Dict[str, Any]]

    # ── 生成层（generator_node 写入）────────────────────────────────────────
    final_response:  str
    confidence_score: float


# 默认初始状态，调用方可以直接 unpack 后再覆盖自定义字段
DEFAULT_STATE: ShoppingAgentState = {
    "messages":        [],
    "intent":          "chat",
    "entities":        {},
    "retrieved_docs":  [],
    "comparison_data": None,
    "final_response":  "",
    "confidence_score": 0.0,
}
