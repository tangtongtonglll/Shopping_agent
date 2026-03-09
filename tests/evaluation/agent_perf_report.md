# LangGraph ShoppingAgent 性能评测报告

> 生成时间：2026-03-08 15:46:04

## 实验设置

| 项目 | 值 |
|------|----|
| 数据集 | `eval_dataset.json`（50 条用例） |
| 节点调用方式 | 直接调用节点函数（bypassing `graph.ainvoke`），精确计时 |
| 意图映射 | CHITCHAT→chat / PRODUCT_SEARCH→search / FUZZY_SEARCH→search / COMPARISON→compare |
| 运行异常 | 0 条 |

## 一、路由准确率

**总体准确率：49/50 = 98.0%**

| 意图 | 正确 | 总数 | 准确率 |
|------|-----:|-----:|------:|
| CHITCHAT | 7 | 8 |  87.5% |
| PRODUCT_SEARCH | 16 | 16 | 100.0% |
| FUZZY_SEARCH | 16 | 16 | 100.0% |
| COMPARISON | 10 | 10 | 100.0% |

### 误分类明细

| # | 期望意图 | 实际意图 | 查询 |
|---|---------|---------|------|
| 44 | chat | search | 我想买一些美妆产品，有什么推荐吗？… |

## 二、节点耗时

| 节点 | N | 均值 | 中位数 | P95 | 最大值 | E2E占比 |
|------|--:|-----:|------:|----:|------:|-------:|
| `router_node` | 50 | 18.54s | 15.71s | 38.39s | 49.67s | 43.6% |
| `retriever_node` | 43 | 0.04s | 0.00s | 0.01s | 1.61s | 0.1% |
| `comparator_node` | 10 | 0.00s | 0.00s | 0.00s | 0.00s | 0.0% |
| `generator_node` | 50 | 23.98s | 22.49s | 40.81s | 51.10s | 56.3% |

## 三、端到端延迟

- **均值**：42.55s
- **中位数**：37.95s
- **P95**：73.56s
- **最快**：23.96s
- **最慢**：90.19s

### 按意图分组

| 意图 | N | 均值 | 中位数 |
|------|--:|-----:|------:|
| CHITCHAT | 8 | 41.71s | 42.65s |
| PRODUCT_SEARCH | 16 | 41.24s | 38.20s |
| FUZZY_SEARCH | 16 | 36.01s | 34.53s |
| COMPARISON | 10 | 55.79s | 56.46s |

## 四、性能瓶颈分析

**主要瓶颈：`generator_node`**（平均 23.98s）

| 排名 | 节点 | 平均耗时 |
|------|------|--------:|
| 1 | `generator_node` | 23.98s |
| 2 | `router_node` | 18.54s |
| 3 | `retriever_node` | 0.04s |
| 4 | `comparator_node` | 0.00s |

**human_review_node 触发**：41/50 条
（触发条件：`confidence_score < 0.6` 或高价值商品关键词）

## 五、优化建议

1. **generator_node**（平均 23.98s）是主要瓶颈，建议：流式返回（streaming）减少感知延迟；对 CHITCHAT 使用轻量模型（如 GLM-4-Flash）；缓存高频问题的回复。
2. **router_node**（平均 18.54s）每次都调用 LLM，建议：使用规则/正则快速路径预过滤明显的 CHITCHAT；或切换为本地轻量分类模型（如 fastText）。
3. **retriever_node**（平均 0.04s）当前语料库空（无 DocumentChunk），建议：将商品 products 表写入 DocumentChunk，启用全量检索能力；Reranker 可批量化以减少逐对打分开销。
4. **comparator_node**（平均 0.00s）仅对 compare 意图触发，建议：预生成常见商品对比数据缓存；简化 prompt 减少 token 数。

---
*由 `evaluate_agent_perf.py` 自动生成 @ 2026-03-08 15:46:04*
*调用服务：app/graph/nodes.py 节点函数*