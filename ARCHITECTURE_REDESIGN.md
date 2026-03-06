# 后端重构架构设计方案

> LangGraph 多智能体 + 高级 RAG（混合检索 + 重排序）
>
> 版本：v2.0（实际落地版）| 日期：2026-03-06

---

## 一、重构背景与目标

本项目原为基于 React + FAISS 的导购浏览器插件，后端使用简单的 `agent_service.py` for-loop 编排 + 单路 FAISS 检索。
重构目标是将其后端升级为符合算法工程师岗位要求的生产级架构，核心方向：

- **LangGraph 多智能体架构**：引入 StateGraph 替代手写编排，支持有状态路由、检查点持久化与 Human-in-the-loop
- **高级 RAG**：实现 BM25 稀疏检索 + FAISS 稠密检索 + BGE Cross-Encoder 重排序的混合检索管道

### 原始架构缺陷

| 维度 | 原始实现 | 核心缺陷 |
|------|----------|---------|
| Agent 编排 | `agent_service.py` 顺序调用 | 无状态图、无条件路由、无持久化、无 Human-in-the-loop |
| RAG 检索 | 单路 FAISS + all-MiniLM-L6-v2（384 维） | 仅语义检索，无稀疏检索、无融合、无重排序 |
| 向量索引 | `IndexFlatL2` 暴力扫描 | 当前规模可用，文档量增大后性能下降 |
| Embedding 模型 | all-MiniLM-L6-v2（多语言平均水平） | 中文理解弱，BEIR 基准约 42 |
| 查询理解 | 原始 query 直接送入 FAISS | 无意图识别、无实体提取 |

---

## 二、重构总览（四个阶段）

```
Phase 1：基础依赖 & 嵌入模型升级
  └── 替换 all-MiniLM → BAAI/bge-large-zh（1024 维）
  └── 升级 requirements.txt（langgraph / rank-bm25 / FlagEmbedding）
  └── FAISS 保持 IndexFlatL2（明确不引入 IVF/PQ 复杂度）

Phase 2：混合检索管道（Advanced Hybrid RAG）
  └── BM25 稀疏检索（rank_bm25）
  └── FAISS 稠密检索（BGE-large-zh 向量）
  └── 两路合并去重 → BGE-Reranker-v2-m3 重排序
  └── 封装为 HybridSearchService.advanced_hybrid_search()

Phase 3：LangGraph 状态与节点定义
  └── ShoppingAgentState（TypedDict 共享状态）
  └── 5 个核心节点函数（符合 LangGraph Node 规范）

Phase 4：StateGraph 装配 & 测试验证
  └── 条件路由（3 个路由函数）
  └── SqliteSaver Checkpointer + interrupt_before
  └── 对外统一接口 run()
  └── 5 项 Mock 测试全通过
```

---

## 三、Phase 1 — 基础依赖与嵌入模型升级

### 3.1 核心依赖变更

**文件**：`backend/requirements.txt`

新增依赖：

```
# LangGraph 多智能体
langgraph>=0.2.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langgraph-checkpoint-sqlite>=1.0.0

# 混合检索
rank-bm25>=0.2.2

# BGE 嵌入与重排序
FlagEmbedding>=1.2.0
```

### 3.2 Embedding 模型升级

**文件**：`backend/app/core/config.py`、`backend/app/services/vector_service.py`

| 对比维度 | 升级前（all-MiniLM-L6-v2） | 升级后（BAAI/bge-large-zh） |
|---------|--------------------------|--------------------------|
| 向量维度 | 384 | 1024 |
| 参数量 | 22M | 335M |
| 中文理解 | 多语言平均水平 | 专门中文优化 |
| BEIR 基准 | ~42 | ~56 |
| Python 库 | `sentence-transformers` | `FlagEmbedding`（FlagModel） |

**关键代码变更（`vector_service.py`）**：

```python
# 升级前
from sentence_transformers import SentenceTransformer
self.model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维
embedding = self.model.encode([text])[0]

# 升级后
from FlagEmbedding import FlagModel
self.model = FlagModel("BAAI/bge-large-zh", use_fp16=False)  # 1024 维
# 查询使用 encode_queries（自动添加 retrieval instruction 前缀）
embedding = self.model.encode_queries([text])[0]
# 文档使用 encode（无前缀）
embedding = self.model.encode([text])[0]
```

### 3.3 FAISS 索引说明

保持使用 `IndexFlatL2`（暴力 L2 距离），不引入 IVF/PQ 量化：

- L2 距离：值越小表示越相似（与 IndexFlatIP 内积相反）
- 相应修正：检索阈值判断由 `distance < threshold` 改为 `distance > threshold`
- 结果排序由降序改为升序（最小距离优先）

---

## 四、Phase 2 — 混合检索管道（Advanced Hybrid RAG）

**文件**：`backend/app/services/hybrid_search_service.py`

### 4.1 架构图

```
用户 Query
    │
    ├──► BM25 稀疏检索 ──────► Top-20 候选（关键词匹配）
    │       (rank_bm25)
    │
    └──► FAISS 稠密检索 ─────► Top-20 候选（语义匹配）
            (BGE-large-zh)
                │
                ▼
        合并去重（按 chunk_id）
        标注来源：bm25 / faiss / both
                │
                ▼
        BGE-Reranker-v2-m3 重排序
        (Cross-Encoder 精排)
                │
                ▼
           Top-5 最终结果
```

### 4.2 核心类接口

```python
class HybridSearchService:
    def advanced_hybrid_search(
        self,
        query: str,
        db,                          # SQLAlchemy Session
        knowledge_base_ids=None,     # 限定知识库范围，None 表示全库
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 5,
    ) -> List[Dict]:
        """
        返回字段：
          chunk_id, document_id, chunk_index, content,
          document_name, metadata, retrieval_source,
          retrieval_score, rerank_score
        """
```

### 4.3 中文分词策略

不依赖 jieba，使用正则分词：

```python
def _tokenize(text: str) -> List[str]:
    # 英文/数字保持完整词，CJK 字符逐字拆分
    return re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text.lower())
```

适合商品名称、品牌、型号的关键词匹配场景。

### 4.4 Reranker 评分

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
pairs = [[query, chunk["content"]] for chunk in candidates]
scores = reranker.compute_score(pairs, normalize=False)  # 返回原始 logits
```

置信度通过 sigmoid 转换：`sigmoid(score) ≈ 0.978` 表示高置信度。

### 4.5 单例工厂

```python
def get_hybrid_search_service() -> HybridSearchService:
    """模块级单例，节点函数通过此入口调用，便于 Mock 测试替换。"""
```

---

## 五、Phase 3 — LangGraph 状态与节点定义

**文件**：`backend/app/graph/state.py`、`backend/app/graph/nodes.py`

### 5.1 共享状态（ShoppingAgentState）

```python
class ShoppingAgentState(TypedDict):
    # 对话层（add_messages reducer：追加而非覆盖）
    messages: Annotated[List[BaseMessage], add_messages]

    # 路由层（router_node 写入）
    intent:   str   # "chat" | "search" | "compare"
    entities: Dict[str, Any]
    # entities 结构：
    # {
    #   "products":   ["华为 Mate60 Pro"],
    #   "brands":     ["华为"],
    #   "category":   "手机",
    #   "budget":     4999,    # 元，None 表示未提及
    #   "attributes": ["拍照", "续航"]
    # }

    # 检索层（retriever_node 写入）
    retrieved_docs: List[Dict[str, Any]]

    # 对比层（comparator_node 写入）
    comparison_data: Optional[Dict[str, Any]]

    # 生成层（generator_node 写入）
    final_response:   str
    confidence_score: float  # 范围 [0.0, 1.0]
```

**设计决策**：`db`（SQLAlchemy Session）和 `knowledge_base_ids` 不放入 state，
通过 `RunnableConfig.configurable` 注入，避免序列化问题影响 SqliteSaver 持久化。

### 5.2 五个核心节点

#### `router_node`（意图路由）
- 调用 LLM（`temperature=0.0`），输出结构化 JSON
- 强制 JSON 格式：`{"intent": "search|compare|chat", "entities": {...}}`
- 解析失败时回退为 `chat` 意图（防御性设计）
- 写入状态：`intent`、`entities`

#### `retriever_node`（混合检索）
- 从 `entities` 构建检索查询（拼接 products + brands + category + attributes + budget）
- 调用 `get_hybrid_search_service().advanced_hybrid_search()`
- 写入状态：`retrieved_docs`

#### `comparator_node`（商品对比）
- 将 `retrieved_docs` 发送给 LLM，生成结构化对比矩阵 JSON
- 格式：`{"products": [...], "dimensions": {"处理器": {...}, ...}, "recommendation": "..."}`
- JSON 解析失败时保留原始文本（防御性降级）
- 写入状态：`comparison_data`

#### `generator_node`（回复生成）
- 组装上下文：优先使用 `comparison_data`，否则使用 `retrieved_docs`
- 调用 LLM（`temperature=0.5`），生成最终回复
- 置信度计算：`sigmoid(mean(top-3 rerank_scores))`；闲聊意图固定 `0.9`
- 写入状态：`messages`（追加 AIMessage）、`final_response`、`confidence_score`

#### `human_review_node`（人工审核）
- 触发条件：`confidence_score < 0.6` 或实体包含高价值商品
- 高价值关键词：`{"苹果", "Apple", "iPhone", "MacBook", "华为", "Mate", "奢侈"}` 或 `budget > 5000`
- 触发时抛出 `NodeInterrupt`（LangGraph 内置机制），暂停图执行
- 写入状态：`{}`（透传，无修改）

---

## 六、Phase 4 — StateGraph 装配与测试

**文件**：`backend/app/graph/graph.py`、`test_graph.py`

### 6.1 图拓扑

```
START
  │
  ▼
router_node          ← 意图识别 + 实体提取（LLM, temp=0.0）
  │
  ├─(search/compare)─► retriever_node    ← BM25 + FAISS + Reranker
  │                         │
  │                   ├─(compare)─► comparator_node ─► generator_node
  │                   └─(search)──────────────────────► generator_node
  │
  └─(chat)──────────────────────────────────────────► generator_node
                                                            │
                                          ┌─────────────────┤
                             (低置信/高价值)                  │（正常）
                                          ▼                  ▼
                                  human_review_node         END
                                          │
                                         END
```

### 6.2 三个条件路由函数

```python
def _route_from_router(state) -> str:
    # "search" / "compare" → "retriever_node"
    # "chat"               → "generator_node"

def _route_from_retriever(state) -> str:
    # "compare" → "comparator_node"
    # "search"  → "generator_node"

def _route_from_generator(state) -> str:
    # confidence < 0.6 或高价值商品 → "human_review_node"
    # 其他                          → END
```

### 6.3 图编译与 Checkpointer

```python
def build_graph(checkpointer=None) -> CompiledStateGraph:
    builder = StateGraph(ShoppingAgentState)
    # 注册 5 个节点 + 固定边 + 条件边

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"]     = checkpointer
        compile_kwargs["interrupt_before"] = ["human_review_node"]

    return builder.compile(**compile_kwargs)
```

- `checkpointer=None`：无状态单次调用，适用于单元测试
- 传入 `SqliteSaver`：启用检查点持久化 + `interrupt_before` 拦截

### 6.4 对外统一接口

```python
async def run(
    query: str,
    db,
    *,
    thread_id: str = "default",
    knowledge_base_ids: Optional[List[int]] = None,
    checkpointer = None,
) -> Dict[str, Any]:
    """
    返回完整 ShoppingAgentState，关键字段：
      final_response   (str)   — 直接呈现给用户的回复
      intent           (str)   — 识别的意图
      confidence_score (float) — 置信度
      retrieved_docs   (list)  — 检索到的文档
    """
```

### 6.5 测试验证结果

测试文件：`test_graph.py`（5 项 Mock 测试，无需真实 API Key 或下载模型）

```
✅ test_graph_compiles   — 图编译正常，5 个节点全部注册
✅ test_search_flow      — search 意图：router → retriever → generator → END
✅ test_compare_flow     — compare 意图：router → retriever → comparator → generator → END
✅ test_chat_flow        — chat 意图：router → generator → END（不触发检索，置信度固定 0.9）
✅ test_checkpointer     — SqliteSaver 未安装时优雅跳过（非失败）

共 5 项，通过 5 项，失败 0 项
```

---

## 七、最终文件结构（已落地部分）

```
backend/
├── requirements.txt                        ★ Phase 1 更新
├── app/
│   ├── core/
│   │   └── config.py                       ★ Phase 1 更新（embedding_model → bge-large-zh）
│   │
│   ├── services/
│   │   ├── vector_service.py               ★ Phase 1 重构（FlagModel, IndexFlatL2 修正）
│   │   └── hybrid_search_service.py        ★ Phase 2 新增（HybridSearchService）
│   │
│   └── graph/                              ★ Phase 3 & 4 新增
│       ├── __init__.py
│       ├── state.py                        — ShoppingAgentState 定义
│       ├── nodes.py                        — 5 个节点函数
│       └── graph.py                        — StateGraph 装配 + run() 接口
│
test_graph.py                               ★ Phase 4 新增（根目录测试脚本）
```

---

## 八、技术选型与设计决策记录

| 决策点 | 选择 | 原因 |
|--------|------|------|
| FAISS 索引类型 | 保持 `IndexFlatL2` | 当前文档量级不需要近似索引，避免引入额外复杂度 |
| 中文分词 | 正则（无 jieba） | 商品名/品牌名/型号不需要语义分词，正则足够且零依赖 |
| DB 注入方式 | `RunnableConfig.configurable` | SQLAlchemy Session 不可序列化，不能放入 state |
| `interrupt_before` 时机 | 仅在有 checkpointer 时启用 | 无 checkpointer 时 `interrupt_before` 会抛出编译错误 |
| Reranker 评分 | `normalize=False`（原始 logits） | 保留完整精度用于置信度 sigmoid 转换 |
| 置信度计算 | `sigmoid(mean(top-3 rerank_scores))` | 避免单个异常分数影响判断 |
| 闲聊置信度 | 固定 `0.9` | 无检索文档时无法通过 rerank_score 计算，直接赋值 |
| 测试策略 | `unittest.mock` patch | 无需下载 BGE 模型 / 连接数据库，CI 友好 |

---

## 九、后续可扩展方向

以下为架构设计阶段规划但本次未实现的方向，可按需扩展：

| 方向 | 实现要点 |
|------|---------|
| 查询改写（HyDE/Multi-Query） | 在 `retriever_node` 前新增节点，生成多角度查询或假设文档 |
| RRF 融合替代简单合并 | `rrf_score = Σ 1/(k + rank_i(d))`，k=60 |
| `v2` API 接入 LangGraph | 新增 `/api/v2/chat` 路由，调用 `graph.run()` |
| SqliteSaver 生产化 | 替换为 `AsyncSqliteSaver` 或 `RedisSaver`（分布式） |
| 流式输出 | `graph.astream()` + SSE 推送各节点中间状态 |
| RAGAS 评估 | 对 hybrid RAG 管道量化评估 `faithfulness` / `context_relevance` |
| FAISS IVF-PQ | 当文档量超过 50 万时引入近似索引，牺牲约 1-3% 精度换毫秒级延迟 |
