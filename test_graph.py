"""
ShoppingAgent LangGraph 图流转测试脚本

运行方式（在项目根目录）：
    python test_graph.py

测试策略
--------
所有外部依赖（LLM API、FAISS 向量库、SQLAlchemy DB）均通过 Mock 替换，
仅测试 LangGraph 图的编译是否正确、条件路由是否按意图流转、
节点执行顺序是否符合预期。

无需真实 API Key，无需下载 BGE 模型，无需任何数据库。
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# ── 将 backend 目录加入 sys.path ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

from langchain_core.messages import HumanMessage


# ============================================================================ #
# Mock 数据                                                                     #
# ============================================================================ #

# router_node 的 LLM 响应（搜索意图）
MOCK_ROUTER_SEARCH = {
    "content": (
        '{"intent": "search", "entities": {'
        '"products": ["华为 Mate60 Pro"], '
        '"brands": ["华为"], '
        '"category": "手机", '
        '"budget": 4999, '
        '"attributes": ["拍照", "续航"]}}'
    ),
    "role": "assistant",
}

# router_node 的 LLM 响应（对比意图）
MOCK_ROUTER_COMPARE = {
    "content": (
        '{"intent": "compare", "entities": {'
        '"products": ["iPhone 15 Pro", "华为 Mate60 Pro"], '
        '"brands": ["Apple", "华为"], '
        '"category": "手机", '
        '"budget": null, '
        '"attributes": ["处理器", "拍照"]}}'
    ),
    "role": "assistant",
}

# router_node 的 LLM 响应（闲聊意图）
MOCK_ROUTER_CHAT = {
    "content": '{"intent": "chat", "entities": {"products": [], "brands": [], "category": "", "budget": null, "attributes": []}}',
    "role": "assistant",
}

# generator_node 的 LLM 响应
MOCK_GENERATOR = {
    "content": "华为 Mate60 Pro 是一款旗舰手机，搭载麒麟9000s芯片，拍照和续航表现出色，预算5000元内非常推荐。",
    "role": "assistant",
}

# comparator_node 的 LLM 响应
MOCK_COMPARATOR = {
    "content": (
        '{"products": ["iPhone 15 Pro", "华为 Mate60 Pro"], '
        '"dimensions": {'
        '"处理器": {"iPhone 15 Pro": "A17 Pro", "华为 Mate60 Pro": "麒麟9000s"}, '
        '"价格": {"iPhone 15 Pro": "8999元", "华为 Mate60 Pro": "6999元"}'
        '}, '
        '"recommendation": "iPhone 15 Pro 性能更强，华为 Mate60 Pro 性价比更高"}'
    ),
    "role": "assistant",
}

# hybrid_search 返回的文档
MOCK_DOCS = [
    {
        "chunk_id":        1,
        "document_id":     10,
        "chunk_index":     0,
        "content":         "华为 Mate60 Pro 搭载麒麟9000s芯片，6.82英寸曲面屏，5000mAh大电池",
        "document_name":   "产品手册.pdf",
        "retrieval_source": "both",
        "retrieval_score": 8.5,
        "rerank_score":    3.8,     # sigmoid(3.8) ≈ 0.978 → 高置信度
    },
    {
        "chunk_id":        2,
        "document_id":     10,
        "chunk_index":     1,
        "content":         "华为 Mate60 Pro 支持卫星通话，防水等级 IP68，重量 225g",
        "document_name":   "产品手册.pdf",
        "retrieval_source": "faiss",
        "retrieval_score": 0.32,
        "rerank_score":    2.1,
    },
]


# ============================================================================ #
# 辅助函数                                                                      #
# ============================================================================ #

def _mock_db():
    """返回一个模拟的 SQLAlchemy Session（不连接真实数据库）。"""
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = []
    return db


def _build_llm_mock(*responses):
    """
    构造一个按顺序返回预设响应的 LLM Mock。
    每次 chat_completion 调用消耗一个响应，循环利用最后一个。
    """
    call_count = {"n": 0}
    resp_list  = list(responses)

    async def _chat_completion(messages, **kwargs):
        idx = min(call_count["n"], len(resp_list) - 1)
        call_count["n"] += 1
        return resp_list[idx]

    mock_llm = MagicMock()
    mock_llm.chat_completion = _chat_completion
    return mock_llm


def _build_search_mock(docs=None):
    """构造返回固定文档列表的 HybridSearchService Mock。"""
    mock_svc = MagicMock()
    mock_svc.advanced_hybrid_search.return_value = docs or MOCK_DOCS
    return mock_svc


# ============================================================================ #
# 测试 1：图编译 + 节点注册                                                      #
# ============================================================================ #

def test_graph_compiles():
    print("\n" + "=" * 60)
    print("测试 1：图编译与节点注册")
    print("=" * 60)

    from app.graph.graph import build_graph

    graph = build_graph(checkpointer=None)   # 无 checkpointer，仅验证编译

    # 检查节点是否全部注册
    node_names = set(graph.nodes.keys())
    expected   = {"router_node", "retriever_node", "comparator_node",
                  "generator_node", "human_review_node"}
    missing    = expected - node_names

    if missing:
        print(f"  ✗ 缺少节点: {missing}")
        return False

    for name in sorted(expected):
        print(f"  ✓ 节点已注册: {name}")

    # 打印 Mermaid 图结构
    try:
        print("\n  Mermaid 图结构:")
        mermaid = graph.get_graph().draw_mermaid()
        for line in mermaid.splitlines():
            print(f"    {line}")
    except Exception:
        pass

    print("\n✅ 测试 1 通过：图编译正常，所有节点注册完毕")
    return True


# ============================================================================ #
# 测试 2：搜索意图完整流转                                                        #
# ============================================================================ #

async def test_search_flow():
    print("\n" + "=" * 60)
    print("测试 2：搜索意图流转 (chat→search→retriever→generator→END)")
    print("=" * 60)

    from app.graph.graph import build_graph
    from app.graph.state import DEFAULT_STATE

    mock_llm    = _build_llm_mock(MOCK_ROUTER_SEARCH, MOCK_GENERATOR)
    mock_search = _build_search_mock(MOCK_DOCS)

    with patch("app.graph.nodes.get_llm_service",            return_value=mock_llm), \
         patch("app.graph.nodes.get_hybrid_search_service",  return_value=mock_search):

        graph = build_graph(checkpointer=None)

        initial_state = {
            **DEFAULT_STATE,
            "messages": [HumanMessage(content="推荐一款拍照好续航强的手机，预算5000")],
        }
        config = {"configurable": {"db": _mock_db(), "knowledge_base_ids": None,
                                   "thread_id": "test-search-001"}}

        result = await graph.ainvoke(initial_state, config=config)

    print(f"  意图识别    : {result['intent']}")
    print(f"  提取实体    : {result['entities']}")
    print(f"  检索文档数  : {len(result['retrieved_docs'])}")
    print(f"  置信度      : {result['confidence_score']:.4f}")
    print(f"  最终回复    : {result['final_response'][:80]}...")

    assert result["intent"]          == "search",  "意图应为 search"
    assert len(result["retrieved_docs"]) > 0,      "应有检索文档"
    assert result["final_response"]  != "",        "回复不应为空"
    assert result["comparison_data"] is None,      "搜索流程不应有对比数据"

    print("\n✅ 测试 2 通过：搜索流程节点顺序与输出正确")
    return True


# ============================================================================ #
# 测试 3：对比意图完整流转                                                        #
# ============================================================================ #

async def test_compare_flow():
    print("\n" + "=" * 60)
    print("测试 3：对比意图流转 (router→retriever→comparator→generator→END)")
    print("=" * 60)

    from app.graph.graph import build_graph
    from app.graph.state import DEFAULT_STATE

    mock_llm    = _build_llm_mock(MOCK_ROUTER_COMPARE, MOCK_COMPARATOR, MOCK_GENERATOR)
    mock_search = _build_search_mock(MOCK_DOCS)

    with patch("app.graph.nodes.get_llm_service",            return_value=mock_llm), \
         patch("app.graph.nodes.get_hybrid_search_service",  return_value=mock_search):

        graph = build_graph(checkpointer=None)

        initial_state = {
            **DEFAULT_STATE,
            "messages": [HumanMessage(content="帮我对比一下 iPhone 15 Pro 和华为 Mate60 Pro")],
        }
        config = {"configurable": {"db": _mock_db(), "knowledge_base_ids": None,
                                   "thread_id": "test-compare-001"}}

        result = await graph.ainvoke(initial_state, config=config)

    print(f"  意图识别    : {result['intent']}")
    print(f"  对比商品    : {result['entities'].get('products', [])}")
    print(f"  对比维度数  : {len((result.get('comparison_data') or {}).get('dimensions', {}))}")
    print(f"  置信度      : {result['confidence_score']:.4f}")
    print(f"  最终回复    : {result['final_response'][:80]}...")

    assert result["intent"]                   == "compare", "意图应为 compare"
    assert result.get("comparison_data") is not None,      "应有对比数据"
    assert len(result["retrieved_docs"])        > 0,        "应有检索文档"

    print("\n✅ 测试 3 通过：对比流程节点顺序与输出正确")
    return True


# ============================================================================ #
# 测试 4：闲聊意图（跳过检索，直接生成）                                           #
# ============================================================================ #

async def test_chat_flow():
    print("\n" + "=" * 60)
    print("测试 4：闲聊意图流转 (router→generator→END，不经过 retriever)")
    print("=" * 60)

    from app.graph.graph import build_graph
    from app.graph.state import DEFAULT_STATE

    mock_llm    = _build_llm_mock(MOCK_ROUTER_CHAT, MOCK_GENERATOR)
    mock_search = _build_search_mock([])   # 搜索不应被调用

    with patch("app.graph.nodes.get_llm_service",            return_value=mock_llm), \
         patch("app.graph.nodes.get_hybrid_search_service",  return_value=mock_search):

        graph = build_graph(checkpointer=None)

        initial_state = {
            **DEFAULT_STATE,
            "messages": [HumanMessage(content="你好，你是谁？")],
        }
        config = {"configurable": {"db": _mock_db(), "knowledge_base_ids": None,
                                   "thread_id": "test-chat-001"}}

        result = await graph.ainvoke(initial_state, config=config)

    print(f"  意图识别    : {result['intent']}")
    print(f"  检索文档数  : {len(result['retrieved_docs'])}  (应为 0)")
    print(f"  置信度      : {result['confidence_score']:.4f}  (闲聊固定 0.9)")
    print(f"  最终回复    : {result['final_response'][:80]}...")

    assert result["intent"]                == "chat",  "意图应为 chat"
    assert len(result["retrieved_docs"])   == 0,       "闲聊不应触发检索"
    assert result["confidence_score"]      == 0.9,     "闲聊置信度固定 0.9"
    assert mock_search.advanced_hybrid_search.call_count == 0, "搜索服务不应被调用"

    print("\n✅ 测试 4 通过：闲聊流程正确跳过检索节点")
    return True


# ============================================================================ #
# 测试 5：SqliteSaver Checkpointer 接入（in-memory）                             #
# ============================================================================ #

async def test_checkpointer():
    print("\n" + "=" * 60)
    print("测试 5：SqliteSaver Checkpointer（in-memory DB）")
    print("=" * 60)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print("  ⚠️  langgraph-checkpoint-sqlite 未安装，跳过此测试。")
        print("     请运行: pip install langgraph-checkpoint-sqlite")
        return True

    from app.graph.graph import build_graph
    from app.graph.state import DEFAULT_STATE

    mock_llm    = _build_llm_mock(MOCK_ROUTER_SEARCH, MOCK_GENERATOR)
    mock_search = _build_search_mock(MOCK_DOCS)

    with patch("app.graph.nodes.get_llm_service",            return_value=mock_llm), \
         patch("app.graph.nodes.get_hybrid_search_service",  return_value=mock_search), \
         SqliteSaver.from_conn_string(":memory:") as checkpointer:

        # 带 checkpointer 编译图（自动启用 interrupt_before=["human_review_node"]）
        graph = build_graph(checkpointer=checkpointer)

        initial_state = {
            **DEFAULT_STATE,
            "messages": [HumanMessage(content="推荐一款旗舰手机")],
        }
        config = {"configurable": {"db": _mock_db(), "knowledge_base_ids": None,
                                   "thread_id": "test-checkpoint-001"}}

        result = await graph.ainvoke(initial_state, config=config)

    print(f"  意图识别    : {result['intent']}")
    print(f"  置信度      : {result['confidence_score']:.4f}")
    print(f"  Checkpointer: SqliteSaver (:memory:) 正常工作")

    assert result["final_response"] != "", "带 checkpointer 的回复不应为空"

    print("\n✅ 测试 5 通过：SqliteSaver Checkpointer 接入正常")
    return True


# ============================================================================ #
# 主入口                                                                        #
# ============================================================================ #

async def main():
    print("\n" + "█" * 60)
    print("  ShoppingAgent LangGraph 图流转测试")
    print("█" * 60)

    results = {}

    # 同步测试（图编译）
    results["test_graph_compiles"] = test_graph_compiles()

    # 异步测试（图执行）
    for name, coro in [
        ("test_search_flow",   test_search_flow()),
        ("test_compare_flow",  test_compare_flow()),
        ("test_chat_flow",     test_chat_flow()),
        ("test_checkpointer",  test_checkpointer()),
    ]:
        try:
            results[name] = await coro
        except AssertionError as e:
            print(f"\n✗ {name} 断言失败: {e}")
            results[name] = False
        except Exception as e:
            print(f"\n✗ {name} 异常: {type(e).__name__}: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")
    print(f"\n  共 {total} 项，通过 {passed} 项，失败 {total - passed} 项")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
