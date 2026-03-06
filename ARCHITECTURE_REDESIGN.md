# 后端重构架构设计方案

> LangGraph 多智能体 + 高级 RAG（混合检索 + 重排序）
>
> 版本：v1.0 | 日期：2026-03-06

---

## 一、现状诊断（问题清单）

在看方案之前，先明确现有架构的核心缺陷：

| 维度 | 当前实现 | 缺陷 |
|------|----------|------|
| **Agent 编排** | `agent_service.py` 简单 for-loop 调用，顺序/并行通过 if-else 分支决定 | 无状态图，无条件路由，无持久化检查点，无 human-in-the-loop |
| **RAG 检索** | 单路 FAISS `IndexFlatL2` + all-MiniLM-L6-v2 (384维) | 仅语义检索，无稀疏检索，无混合融合，无重排序 |
| **向量索引** | 暴力扫描 O(N)，无量化压缩 | 文档量增大后性能崩溃 |
| **查询理解** | 直接将用户原始 query 送入 FAISS | 无意图识别、无查询改写、无假设文档生成 |
| **上下文处理** | 检索结果直接拼接进 prompt | 无压缩、无归因、token 浪费严重 |
| **记忆系统** | 三层记忆靠简单相似度检索，无图结构 | 记忆之间无关联，重要性打分规则死板 |
| **价格预测** | 多模型简单平均，无特征工程 | 无交叉验证，无置信度校准，无外部特征 |

---

## 二、目标架构全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TypeScript)                 │
│   ShoppingAssistant / Chat / Dashboard / Analytics                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ HTTP / WebSocket (streaming)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway Layer                           │
│  /api/v2/chat  /api/v2/search  /api/v2/compare  /api/v2/predict    │
└─────────┬──────────────────────────────────────────────┬────────────┘
          │                                              │
          ▼                                              ▼
┌─────────────────────────┐              ┌───────────────────────────┐
│   LangGraph Agent Graph  │              │  Advanced RAG Pipeline    │
│                         │              │                           │
│  ┌──────────────────┐   │              │  1. Query Analyzer        │
│  │  ShoppingState   │   │◄────────────►│  2. Query Rewriter (HyDE) │
│  │  (共享状态)       │   │              │  3. Hybrid Retriever      │
│  └──────────────────┘   │              │     ├─ BM25 (稀疏)        │
│                         │              │     └─ BGE Dense (稠密)   │
│  ┌─ Nodes ─────────┐   │              │  4. RRF Fusion            │
│  │ IntentClassifier │   │              │  5. Cross-Encoder Rerank  │
│  │ QueryRewriter    │   │              │  6. Context Compressor    │
│  │ HybridRetriever  │   │              │  7. Grounded Generator    │
│  │ ProductSearcher  │   │              └───────────────────────────┘
│  │ PriceAnalyst     │   │
│  │ RiskAnalyst      │   │              ┌───────────────────────────┐
│  │ Comparator       │   │              │    Data / Storage Layer   │
│  │ Recommender      │   │◄────────────►│                           │
│  │ MemoryManager    │   │              │  PostgreSQL (主数据库)     │
│  │ Synthesizer      │   │              │  FAISS IVF-PQ (向量库)    │
│  │ HumanReview      │   │              │  Elasticsearch (稀疏索引) │
│  └─────────────────┘   │              │  Redis (缓存 + 会话)      │
│                         │              │  SQLite (轻量化回退)      │
│  ┌─ Edges ─────────┐   │              └───────────────────────────┘
│  │ conditional      │   │
│  │ routing based on │   │              ┌───────────────────────────┐
│  │ intent + state   │   │              │    ML Model Registry      │
│  └─────────────────┘   │              │                           │
│                         │              │  BGE-large-zh (嵌入)      │
│  Checkpointer:          │              │  BGE-Reranker-v2-m3 (重排)│
│  SqliteSaver /          │              │  Price LightGBM (预测)    │
│  RedisSaver             │              │  Risk Classifier (风险)   │
└─────────────────────────┘              └───────────────────────────┘
```

---

## 三、LangGraph 多智能体架构设计

### 3.1 共享状态定义（StateGraph 核心）

这是整个 Agent Graph 的"神经中枢"，所有节点共享并修改它：

```
ShoppingAgentState:
├── 输入层
│   ├── messages: list[BaseMessage]      # 完整对话历史 (add_messages reducer)
│   ├── user_id: str
│   ├── session_id: str
│   └── raw_query: str
│
├── 理解层（IntentClassifier 写入）
│   ├── intent: Enum                     # PRODUCT_SEARCH / PRICE_QUERY /
│   │                                    # COMPARISON / RECOMMENDATION /
│   │                                    # RISK_QUERY / GENERAL_CHAT
│   ├── entities: dict                   # {product, brand, category, budget, ...}
│   ├── rewritten_query: str             # QueryRewriter 改写后的查询
│   └── hyde_documents: list[str]        # 假设文档（HyDE 策略）
│
├── 检索层（HybridRetriever 写入）
│   ├── bm25_results: list[ScoredChunk]  # 稀疏检索结果
│   ├── dense_results: list[ScoredChunk] # 稠密检索结果
│   ├── fused_results: list[ScoredChunk] # RRF 融合后
│   └── reranked_results: list[ScoredChunk]  # 重排序后最终文档
│
├── 业务层（各专业 Agent 写入）
│   ├── products: list[Product]
│   ├── price_analysis: PriceAnalysis | None
│   ├── risk_report: RiskReport | None
│   ├── comparison_matrix: ComparisonMatrix | None
│   └── recommendations: list[Product]
│
├── 记忆层
│   ├── memory_context: str              # 从记忆系统取出的相关历史
│   └── memory_updates: list[MemoryItem] # 本轮需要写入记忆的内容
│
└── 输出层
    ├── final_response: str
    ├── citations: list[Citation]        # 归因来源
    ├── confidence: float
    └── next_action: str | None          # 是否需要 human review
```

### 3.2 节点（Nodes）设计

#### Node 1: `intent_classifier`
- **职责**：分析用户意图，提取实体（商品、品牌、价格、场景）
- **实现**：调用 LLM + Few-shot prompt，输出结构化 JSON
- **写入状态**：`intent`, `entities`
- **技术亮点**：支持多意图（"比较A和B，然后告诉我哪个值得买" → COMPARISON + RECOMMENDATION）

#### Node 2: `query_rewriter`
- **职责**：查询改写与扩展，提升召回率
- **实现**：两种策略可选：
  - **Multi-Query**：生成 3 个不同角度的查询，并行检索后合并
  - **HyDE (Hypothetical Document Embeddings)**：让 LLM 先生成一段"理想答案"，用这段文本的向量去检索（适合开放域问答）
- **写入状态**：`rewritten_query`, `hyde_documents`

#### Node 3: `hybrid_retriever`（Advanced RAG 核心节点）
- **职责**：执行混合检索 + RRF 融合
- 详见第四章

#### Node 4: `cross_encoder_reranker`
- **职责**：对 RRF 融合结果进行精排
- **实现**：BGE-Reranker-v2-m3 (cross-encoder)，每个 (query, chunk) 对计算相关分
- **写入状态**：`reranked_results` (top-K, K=5)

#### Node 5: `product_searcher`
- **职责**：在商品数据库执行结构化搜索（价格区间、类别、品牌筛选）
- **触发条件**：intent 包含 PRODUCT_SEARCH
- **实现**：向量检索 + 关键词过滤组合，调用爬虫兜底

#### Node 6: `price_analyst`
- **职责**：价格历史分析 + 未来预测
- **触发条件**：intent 包含 PRICE_QUERY
- **实现**：重构为 Time-Series 特征工程 + LightGBM/Prophet 组合，详见第五章

#### Node 7: `risk_analyst`
- **职责**：商品风险评估
- **触发条件**：intent 包含 RISK_QUERY 或产品搜索时附带
- **实现**：从重排后的评论文档中提取风险信号，LLM 进行结构化分析

#### Node 8: `comparator`
- **职责**：多商品对比分析
- **触发条件**：intent == COMPARISON
- **实现**：构建属性对齐矩阵，LLM 生成自然语言对比报告

#### Node 9: `recommender`
- **职责**：个性化推荐
- **触发条件**：intent == RECOMMENDATION
- **实现**：从记忆系统取出用户偏好，结合检索结果打分排序

#### Node 10: `memory_manager`
- **职责**：读取历史记忆（前置）& 写入新记忆（后置）
- **实现**：
  - 前置：语义检索最相关的 episodic + semantic 记忆，注入 state
  - 后置：将本次对话关键信息写入长期记忆，自动摘要压缩

#### Node 11: `synthesizer`（Grounded Generation）
- **职责**：最终答案生成，强制要求归因
- **实现**：
  - 输入：所有业务层结果 + reranked_results + memory_context
  - Prompt 结构：SYSTEM 包含角色定义 + 归因要求，USER 包含结构化上下文 + 原始问题
  - 输出：答案 + 引用标注 `[1][2]` + 置信度

#### Node 12: `human_review`（可选，Human-in-the-Loop）
- **职责**：当置信度 < 0.6 或涉及高价商品时，暂停图执行，等待人工确认
- **实现**：LangGraph 的 `interrupt_before` 机制

### 3.3 边（Edges）与条件路由

```
START
  │
  ▼
[memory_manager] ──读取历史记忆──►
  │
  ▼
[intent_classifier]
  │
  ├─ intent == GENERAL_CHAT ──────────────────────────────► [synthesizer]
  │
  ├─ intent ∈ {PRODUCT_SEARCH, RECOMMENDATION} ──────────► [query_rewriter]
  │                                                              │
  │                                                              ▼
  │                                                     [hybrid_retriever]
  │                                                              │
  │                                                              ▼
  │                                                    [cross_encoder_reranker]
  │                                                              │
  │                                                    ┌─────────┴─────────┐
  │                                                    ▼                   ▼
  │                                           [product_searcher]   [recommender]
  │                                                    │
  ├─ intent == PRICE_QUERY ───────────────────────────►│
  │                                                    ▼
  │                                           [price_analyst]
  │
  ├─ intent == RISK_QUERY ────────────────────────────────────► [risk_analyst]
  │
  └─ intent == COMPARISON ────────────────────────────────────► [comparator]
                                                                     │
所有业务节点完成后 ◄───────────────────────────────────────────────────┘
  │
  ▼
[synthesizer]
  │
  ├─ confidence < 0.6 ──────────────────────────────────► [human_review] ──► 继续或终止
  │
  └─ confidence >= 0.6 ─────────────────────────────────► [memory_manager] ──写入记忆──► END
```

### 3.4 检查点持久化（Checkpointing）

| 环境 | 方案 | 说明 |
|------|------|------|
| 开发环境 | `SqliteSaver` | LangGraph 内置，零依赖 |
| 生产环境 | `RedisSaver` | 支持分布式会话恢复 |

**核心价值**：每次节点执行后保存完整 state，支持：
- 对话中断后恢复
- human-in-the-loop 后继续执行
- 调试时回放任意节点

---

## 四、高级 RAG 管道设计（Hybrid Retrieval + Reranking）

这是最核心的技术升级，完整 7 阶段流水线：

```
用户 Query
    │
    ▼
━━━ Stage 1: Query Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│  NER 抽取：品牌 / 类别 / 价格区间 / 使用场景             │
│  语言检测 + 规范化（繁→简，全→半角）                    │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 2: Query Rewriting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│  策略 A - Multi-Query：生成 3 个语义等价变体             │
│  策略 B - HyDE：生成假设答案文档取其向量                │
│  策略 C - Step-Back：提炼更抽象的上位问题                │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 3: Hybrid Retrieval ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│                                                          │
│  ┌── 稀疏检索 (BM25) ──────────────────────────────┐   │
│  │ 引擎: Elasticsearch / rank_bm25                  │   │
│  │ 优势: 精确关键词匹配（SKU 号 / 品牌名 / 型号）   │   │
│  │ 返回: top-50 with BM25 scores                    │   │
│  └─────────────────────────────────────────────────┘   │
│                          +                               │
│  ┌── 稠密检索 (Dense) ─────────────────────────────┐   │
│  │ 模型: BAAI/bge-large-zh (1024 维)                │   │
│  │ 索引: FAISS IVF4096,PQ64（近似检索）             │   │
│  │ 优势: 语义理解（"便宜的游戏本"→预算有限）        │   │
│  │ 返回: top-50 with cosine scores                  │   │
│  └─────────────────────────────────────────────────┘   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 4: RRF Fusion ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│  算法: Reciprocal Rank Fusion                            │
│  RRF_score(d) = Σ 1 / (k + rank_i(d))                  │
│  k = 60（平衡两路信号）                                 │
│  Multi-query 结果同样 RRF 合并                          │
│  输出: top-100 候选文档                                 │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 5: Cross-Encoder Reranking ━━━━━━━━━━━━━━━━━━━━
│  模型: BAAI/bge-reranker-v2-m3（支持中英双语）          │
│  原理: 对每个 (query, chunk) 对做完整注意力计算         │
│  输入: top-100 候选                                     │
│  输出: top-10，每个附带 relevance_score                 │
│  与双塔模型的区别: 精度更高，无近似误差                 │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 6: Contextual Compression ━━━━━━━━━━━━━━━━━━━━━
│  对 top-10 文档进行压缩，仅保留与 query 相关的句子      │
│  方法: LLM-based 或 Extractive（句子相似度过滤）        │
│  目的: 减少 context window 占用，降噪                   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ▼
━━━ Stage 7: Grounded Generation ━━━━━━━━━━━━━━━━━━━━━━━━
│  Prompt 结构:                                           │
│    [System] 你是购物助手，必须基于以下文档回答          │
│    [Context] <压缩后的文档1>[来源1] ...                │
│    [Query] 用户问题                                     │
│  要求: 引用标注 [1][2]，无法确定时说明                  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.1 嵌入模型升级：从 all-MiniLM 到 BGE

| 对比维度 | 当前（all-MiniLM-L6-v2） | 升级后（BGE-large-zh） |
|---------|------------------------|----------------------|
| 向量维度 | 384 | 1024 |
| 模型大小 | 22M 参数 | 335M 参数 |
| 中文理解 | 弱（多语言模型平均水平） | 强（专门中文优化） |
| BEIR 基准 | ~42 | ~56 |
| 推理速度 | 快 | 中等（可量化加速） |

### 4.2 FAISS 索引升级

| 对比维度 | 当前（IndexFlatL2） | 升级后（IVF4096,PQ64） |
|---------|-------------------|----------------------|
| 搜索复杂度 | O(N) 暴力 | O(N/nlist) 近似 |
| 内存占用 | 高（全精度向量） | 低（PQ 量化压缩 4x-8x） |
| 100万文档延迟 | 秒级 | 毫秒级 |
| 精度损失 | 0% | ~1-3%（可接受） |

### 4.3 BM25 集成方案

| 方案 | 工具 | 适用场景 |
|------|------|--------|
| 轻量方案 | `rank_bm25`，in-memory | < 100k 文档 |
| 生产方案 | Elasticsearch + IK 中文分词 | > 100k 文档，需持久化 |

数据同步策略：文档上传时同时写入 FAISS + ES，保持双路索引一致。

### 4.4 RAG 质量评估体系（RAGAS）

引入 [RAGAS](https://github.com/explodinggradients/ragas) 框架对 RAG 管道进行量化评估：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| `faithfulness` | 答案是否忠于检索文档 | > 0.85 |
| `context_relevance` | 检索文档与问题的相关性 | > 0.75 |
| `answer_relevance` | 答案与问题的相关性 | > 0.80 |
| `context_recall` | 参考答案所需信息是否被检索到 | > 0.70 |

---

## 五、其他模块重构设计

### 5.1 价格预测升级

**当前问题**：仅用历史价格做时序，无外部特征，无评估指标，多模型简单平均。

**重构方向**：

```
特征工程：
├── 时序特征：lag-1/7/14/30, rolling_mean, rolling_std, 季节性分解
├── 外部特征：节假日标志（双11/618/春节）, 星期几, 月份
├── 竞品特征：竞品价格走势
└── 平台特征：平台促销日历

模型选型：
├── LightGBM（梯度提升，特征重要性可解释）
├── Prophet（处理节假日效应，Facebook 开源）
└── Ensemble：基于验证集表现的加权平均（非简单平均）

评估体系：
├── 时序交叉验证（TimeSeriesSplit）
├── 指标：MAE, MAPE, RMSE
└── 在线监控：预测误差告警
```

### 5.2 风险检测升级

**当前问题**：简单字符串匹配，无语境理解，假阳性高。

**重构方向**：

```
评论向量化：
├── 将商品评论分块后存入 FAISS
└── 检索时拉取负面评论（通过 "质量差/物流慢/售后差" 等 anchor query）

风险分析 Pipeline：
├── Stage 1：语义检索负面评论（复用 Advanced RAG 流水线）
├── Stage 2：Sentiment 分类（positive / negative / neutral）
├── Stage 3：LLM 结构化抽取（Chain-of-Thought）
│   Prompt："从以下评论中提取：①质量问题 ②物流问题 ③服务问题"
└── Stage 4：置信度校准，输出各维度风险分 + 原始证据
```

### 5.3 记忆系统升级

**重构方向**：

```
分层记忆架构：
├── Working Memory（Redis, TTL=1h）
│   └── 当前会话上下文，快速读写
├── Episodic Memory（PostgreSQL + FAISS）
│   └── 具体对话记录，按时间索引
├── Semantic Memory（PostgreSQL + FAISS）
│   └── 提炼出的用户偏好 / 购物习惯 / 品牌倾向
└── Procedural Memory（规则库）
    └── 用户设置的固定偏好规则

新增能力：
├── 记忆摘要：当 episodic 记忆超过阈值，LLM 自动压缩为 semantic 条目
├── 记忆遗忘：基于访问频率 + 时间衰减的重要性公式，定期清理低权重记忆
└── 记忆关联：相关记忆之间建立引用关系（邻接表），支持关联查询
```

---

## 六、新增依赖栈

### 核心新增

```
# LangGraph（替换手写 agent 编排）
langgraph>=0.2.0
langchain-core>=0.3.0

# 混合检索
rank-bm25>=0.2.2                    # 轻量级 BM25
elasticsearch>=8.0.0                # 生产级 BM25（可选）

# 重排序模型（BGE）
FlagEmbedding>=1.2.0                # BAAI/bge 系列封装
                                    # 内含 FlagReranker, FlagModel

# Checkpointing
langgraph-checkpoint-sqlite>=1.0.0
langgraph-checkpoint-redis>=1.0.0   # 生产环境用

# 特征工程 & 时序预测
lightgbm>=4.0.0
prophet>=1.1.4                      # 替代 custom ARIMA
optuna>=3.0.0                       # 超参数搜索

# RAG 评估
ragas>=0.1.0                        # 评估 faithfulness / context_relevance / answer_relevance
```

### 保留不变

```
fastapi, uvicorn, sqlalchemy, pydantic  # Web 框架层不动
langchain-community                     # 文档处理工具链
sentence-transformers                   # 作为备选嵌入方案
scrapy, playwright                      # 爬虫层不动
openai, zhipuai                         # LLM 接入层不动
redis                                   # 缓存层扩展使用
```

---

## 七、目标文件结构

```
backend/
├── app/
│   ├── main.py                           # 不变，添加 v2 路由
│   ├── core/
│   │   ├── config.py                     # 新增 BGE_MODEL_PATH, ES_URL 等配置
│   │   └── database.py                  # 不变
│   │
│   ├── graph/                            # ★ 新增：LangGraph 核心
│   │   ├── __init__.py
│   │   ├── state.py                      # ShoppingAgentState 定义
│   │   ├── graph.py                      # StateGraph 组装，边与路由定义
│   │   ├── checkpointer.py              # SqliteSaver / RedisSaver 配置
│   │   └── nodes/
│   │       ├── intent_classifier.py
│   │       ├── query_rewriter.py
│   │       ├── hybrid_retriever.py       # ★ RAG 核心节点
│   │       ├── reranker.py               # ★ 重排序节点
│   │       ├── product_searcher.py
│   │       ├── price_analyst.py
│   │       ├── risk_analyst.py
│   │       ├── comparator.py
│   │       ├── recommender.py
│   │       ├── memory_manager.py
│   │       └── synthesizer.py
│   │
│   ├── rag/                              # ★ 新增：高级 RAG 组件
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py             # BM25 稀疏检索封装
│   │   ├── dense_retriever.py            # BGE 稠密检索封装
│   │   ├── rrf_fusion.py                # Reciprocal Rank Fusion
│   │   ├── cross_encoder.py              # BGE Reranker 封装
│   │   ├── query_rewriter.py             # HyDE / Multi-query 实现
│   │   ├── context_compressor.py         # 上下文压缩
│   │   └── indexer.py                    # 文档写入双路索引（FAISS + BM25）
│   │
│   ├── api/
│   │   ├── v2/                           # ★ 新增：v2 版本 API
│   │   │   ├── chat.py                   # 接入 LangGraph graph
│   │   │   ├── search.py
│   │   │   └── stream.py                # SSE 流式输出
│   │   └── [旧 API 保留，逐步迁移]
│   │
│   ├── models/                           # 不变，按需新增字段
│   └── services/
│       ├── [旧服务保留兼容]
│       ├── embedding_service.py          # ★ 重构：统一 BGE 嵌入入口
│       ├── price_prediction_v2.py        # ★ 重构：LightGBM + Prophet
│       └── risk_detection_v2.py          # ★ 重构：语义分析版本
```

---

## 八、迁移策略（平滑过渡，三阶段）

### Phase 1：RAG 升级（不动 Agent 层）

1. 新增 `rag/` 目录，实现 BM25 + Dense + RRF + Reranker
2. 在 `enhanced_rag_service.py` 中接入新管道
3. 用 RAGAS 框架量化评估新旧 RAG 效果
4. A/B 测试：50% 流量走旧管道，50% 走新管道，对比指标

### Phase 2：Agent 迁移到 LangGraph

1. 新增 `graph/` 目录，实现 StateGraph
2. 新增 `/api/v2/chat` 接入 LangGraph
3. 旧 `/api/chat` 保留不删，前端可切换
4. 逐步将 `agent_service.py` 中的业务逻辑迁入 `graph/nodes/`

### Phase 3：模型升级 + 生产化

1. 替换嵌入模型为 BGE-large-zh，重建 FAISS 索引（需离线 rebuild）
2. 价格预测和风险检测升级为 v2 版本
3. 接入 RAGAS 持续评估，接入 Prometheus 监控
4. 视数据规模决定是否引入 Elasticsearch 替换 in-memory BM25

---

## 九、面向算法工程师岗位的技术亮点对照

| 岗位要求 | 本方案对应实现 | 核心关键词 |
|---------|--------------|-----------|
| LangGraph 多智能体架构 | 12 节点 StateGraph，条件路由，检查点持久化，human-in-the-loop | `StateGraph`, `add_messages`, `SqliteSaver`, `interrupt_before` |
| 高级 RAG | 7 阶段流水线：BM25 + BGE → RRF → BGE-Reranker → 压缩 → 归因生成 | `Hybrid Retrieval`, `RRF Fusion`, `Cross-Encoder Reranking`, `HyDE` |
| 向量检索优化 | FAISS IVF+PQ 替换暴力搜索，BGE-large-zh 替换 MiniLM | `IVF4096,PQ64`, `ANN`, `量化压缩` |
| RAG 评估体系 | RAGAS 框架持续评估 faithfulness / context_relevance / answer_relevance | `RAGAS`, `Groundedness`, `归因` |
| 时序预测 | LightGBM + 特征工程 + TimeSeriesSplit 交叉验证 | `LightGBM`, `Prophet`, `特征工程` |
| 可观测性 | 每个节点 trace，RAG 链路可解释，引用归因 | `LangSmith`, `Trace`, `Citations` |

---

> **核心设计原则**：不是推倒重来，而是在保留 80% 业务逻辑的前提下，精准替换掉那些"说不出口"的实现——简单 for-loop 换成 LangGraph StateGraph，单路 FAISS 换成 7 阶段 Hybrid RAG Pipeline。每一处改动都有明确的算法理由和可量化的收益。
