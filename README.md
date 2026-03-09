# 基于 LangGraph 与高级混合 RAG 的多智能体导购系统

> 一个面向电商场景、由 LangGraph StateGraph 编排、集成 BM25 + FAISS + BGE Cross-Encoder 三路混合检索的多智能体导购助手，配备浏览器插件前端与完整自动化评测 Pipeline。

---

## 目录

- [项目概览](#项目概览)
- [核心架构](#核心架构)
- [高阶检索与生成](#高阶检索与生成)
- [量化评测指标](#量化评测指标)
- [技术栈](#技术栈)
- [快速启动](#快速启动)
- [项目结构](#项目结构)

---

## 项目概览

本项目是一个**基于 LangGraph 与高级混合 RAG 的多智能体导购系统**，核心能力：

- **意图感知路由**：LLM 自动识别用户意图（闲聊 / 商品搜索 / 多商品对比），驱动 StateGraph 差异化流转
- **三路混合检索**：BM25 稀疏检索 + BGE-large-zh 稠密检索 + bge-reranker-v2-m3 Cross-Encoder 精排，覆盖关键词精确匹配与中文语义检索两类场景
- **Human-in-the-Loop**：当置信度低于阈值（0.6）或命中高价值商品关键词时，通过 `NodeInterrupt` 中断图执行、等待人工审核后恢复
- **图像导购**：浏览器插件端 GLM-4.6V 视觉模型识别商品品类，结合 SQL 关键词匹配返回相似商品
- **全链路可观测**：SqliteSaver Checkpointing 持久化每步状态，支持断点续跑；终端彩色日志逐节点追踪流转

---

## 核心架构

### LangGraph StateGraph 拓扑

```
          START
            │
      router_node              ← 意图识别 + 结构化实体提取
     /      │       \
 search/compare    chat
      │               │
retriever_node    generator_node
      │
  compare? ── comparator_node ──┐
      │                         │
  search? ──────────────────────┴─→ generator_node
                                              │
                         ┌────────────────────┘
                    needs_review?
                    /           \
          human_review_node     END
                │
               END   ← interrupt_before 暂停，等待外部恢复
```

**固定边**：
- `START → router_node`
- `comparator_node → generator_node`
- `human_review_node → END`

**条件边**：

| 源节点 | 条件函数 | search/compare → | chat → |
|--------|---------|-----------------|--------|
| `router_node` | `_route_from_router` | `retriever_node` | `generator_node` |
| `retriever_node` | `_route_from_retriever` | compare → `comparator_node`；search → `generator_node` | — |
| `generator_node` | `_route_from_generator` | confidence < 0.6 或高价值商品 → `human_review_node` | → `END` |

### 节点职责

| 节点 | 类型 | 职责 |
|------|------|------|
| **router_node** | 异步 | 调用 GLM-4.6，输出结构化 JSON（`intent` + `entities`），支持 3 类意图 × 5 类实体（products / brands / category / budget / attributes）|
| **retriever_node** | 异步 | 从 `entities` 拼接检索 query，调用 `HybridSearchService.advanced_hybrid_search(bm25_top_k=20, dense_top_k=20, final_top_k=5)` |
| **comparator_node** | 异步 | 仅在 `intent == "compare"` 时激活，将检索文档送入 LLM，输出结构化对比矩阵（products / dimensions / recommendation）|
| **generator_node** | 异步 | 整合 retrieved_docs / comparison_data，调用 LLM 生成导购回复；基于 bge-reranker logit 经 sigmoid 映射计算 `confidence_score` |
| **human_review_node** | 同步 | 检测 confidence < 0.6 或高价值商品（预算 > 5000 元 / 特定品牌），触发 `NodeInterrupt` 暂停图执行 |

### 共享状态（ShoppingAgentState）

```python
class ShoppingAgentState(TypedDict):
    messages:         Annotated[List[BaseMessage], add_messages]  # 对话历史，add_messages reducer
    intent:           str          # "chat" | "search" | "compare"
    entities:         Dict[str, Any]  # products / brands / category / budget / attributes
    retrieved_docs:   List[Dict[str, Any]]  # 混合检索结果（含 rerank_score）
    comparison_data:  Optional[Dict[str, Any]]  # 结构化对比矩阵
    final_response:   str          # 最终导购回复
    confidence_score: float        # [0.0, 1.0]，低于 0.6 触发人工审核
```

> **DB 注入方式**：SQLAlchemy Session 通过 `config["configurable"]["db"]` 传入，不序列化进 State，避免 Checkpointer 序列化报错。

### Checkpointing 与中断恢复

```python
# 编译图时注册 SqliteSaver + interrupt_before
graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string("shopping_agent_checkpoints.db"),
    interrupt_before=["human_review_node"],
)

# 外部恢复（人工确认后继续）
graph.invoke(None, config={"configurable": {"thread_id": "xxx"}})
```

---

## 高阶检索与生成

### 混合检索 Pipeline

```
用户 Query
    │
    ├── BM25 稀疏检索（rank-bm25）─────────── top-20
    │      _tokenize: 英文整词 + 中文单字
    │      BM25Okapi.get_scores → 丢弃 score ≤ 0
    │
    └── BGE 稠密检索（BAAI/bge-large-zh）── top-20
           1024-dim embedding → FAISS IndexFlatL2
           threshold=2.0（不预过滤，交给重排）
                │
                ▼
        合并去重（chunk_id 去重，标记 "bm25"/"faiss"/"both"）
                │
                ▼
    BGE Cross-Encoder 重排（BAAI/bge-reranker-v2-m3）
        compute_score(pairs, normalize=False) → 原始 logit
        按 logit 降序截取 top-5
                │
                ▼
           final_top_k = 5 条结果
```

**各路优势互补**：

| 检索路 | 优势 | 劣势 |
|--------|------|------|
| BM25 | 品牌名/型号精确匹配（"Magick Botanicals"）速度极快 | 中文语义查询 vs 英文语料命中率接近 0 |
| BGE Dense | 跨语言语义对齐（"防水睫毛膏" → 英文商品） | 精确关键词编码存在压缩损失 |
| Hybrid + Rerank | 综合两路信号，Cross-Encoder 对每对 (query, passage) 做完整 Attention 精排 | 延迟较高（需逐对打分） |

### 置信度计算

```python
def _estimate_confidence(retrieved_docs):
    # bge-reranker-v2-m3 输出原始 logit（正值相关，负值不相关）
    # 经 sigmoid 映射至 (0, 1)，取 top-3 均值
    scores = [sigmoid(d["rerank_score"]) for d in retrieved_docs[:3]]
    return round(sum(scores) / len(scores), 4)
```

### Human-in-the-Loop 实现机制

`human_review_node` 使用 `langgraph.errors.NodeInterrupt` 实现图中断：

```python
# 触发条件：置信度 < 0.6 OR 高价值商品关键词（苹果/华为/iPhone 等）OR 预算 > 5000 元
if needs_review:
    raise NodeInterrupt(f"需要人工审核 — {reason}。回复草稿: {draft[:100]}...")
```

图在 `interrupt_before=["human_review_node"]` 处暂停，调用方可读取当前 State 草稿，人工修改或确认后通过 `graph.invoke(None, config=...)` 恢复执行。

### 视觉搜索（浏览器插件端）

```
POST /api/visual-search/search/base64
    │
    ├── asyncio.gather(
    │       ① numpy LBP 图像特征提取
    │       ② GLM-4.6V 视觉模型：输出结构化 JSON
    │          {"type": "lipstick", "keywords": [...], "zh": "..."}
    │   )
    │
    └── SQL LIKE 查询（title/description × 关键词）
           评分：type_match×0.50 + kw_title×0.20 + kw_desc×0.15
                + rating×0.10 + popularity×0.05
           → 返回 similar_products（top-10）
```

---

## 量化评测指标

所有指标均由 `tests/evaluation/` 下的自动化评测脚本在本地运行生成，数据集为 50 条标注用例（覆盖 4 类意图）。

### 一、RAG 检索策略对比（`evaluate_rag.py`）

> 数据集：42 条含 expected_skus 的检索用例；语料库：Amazon 美妆 1000 条商品

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@10 | 平均查询耗时 |
|------|------:|------:|------:|-------:|----------:|
| A: BM25 Only | 50.0% | 54.8% | 57.1% | 0.5377 | 11.3 ms |
| B: BGE Dense (bge-large-zh) | 33.3% | 42.9% | 47.6% | 0.3937 | 80.1 ms |
| **C: Hybrid + Rerank** | **61.9%** | **61.9%** | **64.3%** | **0.6302** | 5456 ms |

**按意图分组（策略 C）**：

| 意图 | Hit@5 | MRR@10 | N |
|------|------:|-------:|---|
| PRODUCT_SEARCH | 93.8% | 0.9464 | 16 |
| COMPARISON | **100.0%** | **1.0000** | 10 |
| FUZZY_SEARCH（中文语义 vs 英文语料） | 12.5% | 0.0828 | 16 |

> FUZZY_SEARCH 命中率偏低，根因为语料库均为英文商品文本，中文模糊查询存在语言鸿沟；已在评测报告中提出加入中文翻译字段的优化建议。

### 二、LangGraph Agent 性能（`evaluate_agent_perf.py`）

> 50 条用例，直接调用节点函数精确计时，0 条运行异常

**路由准确率：49/50 = 98.0%**

| 意图 | 正确 | 总数 | 准确率 |
|------|-----:|-----:|------:|
| CHITCHAT | 7 | 8 | 87.5% |
| PRODUCT_SEARCH | 16 | 16 | **100.0%** |
| FUZZY_SEARCH | 16 | 16 | **100.0%** |
| COMPARISON | 10 | 10 | **100.0%** |

**节点耗时分布**：

| 节点 | 均值 | 中位数 | P95 | E2E 占比 |
|------|-----:|------:|----:|--------:|
| `generator_node` | 23.98 s | 22.49 s | 40.81 s | 56.3% |
| `router_node` | 18.54 s | 15.71 s | 38.39 s | 43.6% |
| `retriever_node` | 0.04 s | 0.00 s | 0.01 s | 0.1% |
| `comparator_node` | 0.00 s | 0.00 s | 0.00 s | 0.0% |

**端到端延迟**：均值 42.55 s，中位数 37.95 s，P95 73.56 s

> 延迟主要来自 GLM-4.6 reasoning 模型的推理时间（含 CoT 链式思考），本地检索节点（BM25 + FAISS）耗时可忽略不计。

### 三、长上下文处理能力——大海捞针（`evaluate_context.py`）

> 测试方法：将 N 个商品详细参数注入上下文，询问特定位置商品的屏幕刷新率；模型：GLM-4.6

| 上下文规模 | Prompt Token | 准确率 | 平均延迟 |
|-----------|------------:|------:|-------:|
| 10 个商品 | 1,537 | **100%** | 6.9 s |
| 20 个商品 | 2,947 | **100%** | 6.6 s |
| 30 个商品 | 4,349 | **100%** | 5.4 s |
| 40 个商品 | 5,751 | **100%** | 9.1 s |
| 50 个商品 | 7,182 | **100%** | 5.2 s |

**位置效应（Lost-in-the-Middle）**：开头 / 前段 / 中间 / 后段 / 末尾 各位置均 5/5 命中，未观测到位置偏差。

### 四、评测 Pipeline 组成

```
tests/evaluation/
├── generate_eval_dataset.py    # 构建 50 条标注评测集（意图 + expected_skus）
├── evaluate_rag.py             # 三路检索策略 Hit@K / MRR 对比评测
├── evaluate_agent_perf.py      # LangGraph 节点路由准确率 + 耗时分析
├── evaluate_context.py         # 大海捞针长上下文理解测试
├── evaluate_memory.py          # 记忆系统提取与召回评测
├── eval_dataset.json           # 标注数据集（50 条）
├── rag_metrics.md              # RAG 评测报告（自动生成）
├── agent_perf_report.md        # Agent 性能报告（自动生成）
└── context_utilization_report.md  # 长上下文报告（自动生成）
```

---

## 技术栈

### 后端核心

| 类别 | 依赖 | 版本 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 0.104.1 / 0.24.0 |
| ORM + 数据库 | SQLAlchemy + SQLite | 2.0.23 |
| Agent 编排 | langgraph | ≥0.2.0 |
| 图状态持久化 | langgraph-checkpoint-sqlite | ≥1.0.0 |
| LangChain 基础 | langchain-core / langchain-community | ≥0.3.0 |
| 稀疏检索 | rank-bm25 | ≥0.2.2 |
| 稠密检索 + 重排序 | FlagEmbedding (BAAI/bge-large-zh + bge-reranker-v2-m3) | ≥1.2.0 |
| 向量索引 | faiss-cpu | 1.7.4 |
| 向量计算 | numpy | 1.24.3 |
| LLM SDK | zhipuai | 2.1.5 |
| 图像处理 | Pillow + opencv-python | — |
| 数据模型 | pydantic | 2.5.0 |

### LLM 模型

| 用途 | 模型 ID | 特点 |
|------|---------|------|
| 文本推理（路由 / 生成 / 对比） | `GLM-4.6` | Reasoning 模型，含 `reasoning_content` CoT 字段 |
| 视觉理解（图像导购） | `glm-4.6v` | 多模态，输出结构化 JSON 品类关键词 |

### 浏览器插件

- Vanilla JS + Chrome Extension Manifest V3
- 侧边栏（Side Panel）UI：聊天 / 视觉搜索 / 商品对比

---

## 快速启动

### 1. 环境配置

```bash
# 推荐 Python 3.9+
conda create -n Shopping_agent python=3.9
conda activate Shopping_agent
```

### 2. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

> 关键依赖说明：`FlagEmbedding` 首次运行会自动下载 `BAAI/bge-large-zh`（约 1.3 GB）和 `BAAI/bge-reranker-v2-m3`（约 1.1 GB），请确保网络畅通或已提前缓存模型。

### 3. 配置环境变量

```bash
# 在 backend/ 目录下创建 .env
ZHIPUAI_API_KEY=your_zhipuai_api_key
TEXT_MODEL=GLM-4.6
VISION_MODEL=glm-4.6v
DATABASE_URL=sqlite:///./llm_agent.db
SECRET_KEY=your_secret_key
```

### 4. 启动后端服务

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，API 文档可访问：`http://localhost:8000/docs`

### 5. 加载 Amazon 商品数据

```bash
# 在 backend/ 目录下运行，导入 Amazon 美妆数据集（~1000 条）
python import_amazon_data.py
```

### 6. 安装浏览器插件

1. 打开 Chrome，进入 `chrome://extensions/`
2. 开启「开发者模式」
3. 点击「加载已解压的扩展程序」，选择 `browser-extension/` 目录

### 7. 运行评测（可选）

```bash
# RAG 检索策略评测
conda run -n Shopping_agent python tests/evaluation/evaluate_rag.py

# Agent 路由准确率 + 延迟评测
conda run -n Shopping_agent python tests/evaluation/evaluate_agent_perf.py

# 长上下文大海捞针评测
conda run -n Shopping_agent python tests/evaluation/evaluate_context.py
```

---

## 项目结构

```
Shopping_agent/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py                  # /api/chat/chat + /chat/enhanced 路由
│   │   │   ├── visual_search.py         # /api/visual-search/search/base64
│   │   │   ├── price_tracker.py
│   │   │   └── product_comparison.py
│   │   ├── graph/
│   │   │   ├── state.py                 # ShoppingAgentState 定义
│   │   │   ├── nodes.py                 # 5 个 LangGraph 节点函数
│   │   │   └── graph.py                 # StateGraph 组装、条件边、build_graph()
│   │   ├── services/
│   │   │   ├── hybrid_search_service.py # BM25 + FAISS + BGE Reranker
│   │   │   ├── vector_service.py        # BGE-large-zh embedding + FAISS IndexFlatL2
│   │   │   ├── visual_search_service.py # 图像搜索（LBP + GLM-4.6V）
│   │   │   ├── llm_service.py           # ZhipuAI LLM 封装
│   │   │   └── rag_service.py           # 文档切分 + 知识库管理
│   │   ├── models/
│   │   │   ├── ecommerce_models.py      # Product ORM（title 字段）
│   │   │   └── models.py                # KnowledgeBase / Document / DocumentChunk
│   │   └── core/
│   │       ├── config.py                # 环境变量配置
│   │       └── database.py              # SQLAlchemy SessionLocal
│   ├── requirements.txt
│   └── import_amazon_data.py            # 数据导入脚本
├── browser-extension/
│   ├── sidepanel.js                     # 插件主逻辑
│   ├── sidepanel.css
│   └── api.js                           # 后端 API 调用封装
├── tests/
│   └── evaluation/
│       ├── generate_eval_dataset.py
│       ├── evaluate_rag.py
│       ├── evaluate_agent_perf.py
│       ├── evaluate_context.py
│       ├── evaluate_memory.py
│       ├── rag_metrics.md               # 自动生成评测报告
│       ├── agent_perf_report.md
│       └── context_utilization_report.md
└── README.md
```

---

## 数据集

- **来源**：Amazon McAuley Lab 美妆品类（`meta_All_Beauty.jsonl.gz.parquet`）
- **规模**：~1000 条真实商品，platform = `amazon`
- **字段**：`product_id / title / description / brand / category / price / rating / review_count` 等
- **评测集**：`tests/evaluation/eval_dataset.json`，50 条人工标注用例，覆盖 CHITCHAT / PRODUCT_SEARCH / FUZZY_SEARCH / COMPARISON 四类意图
