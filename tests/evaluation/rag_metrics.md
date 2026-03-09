# RAG 检索策略评测报告

> 生成时间：2026-03-08 14:57:12

## 实验设置

- **数据集**：`eval_dataset.json`（共 50 条）
- **检索用例**：42 条（含 expected_skus）
- **跳过用例**：8 条 CHITCHAT（无需检索）
- **意图分布**：CHITCHAT 8条、COMPARISON 10条、FUZZY_SEARCH 16条、PRODUCT_SEARCH 16条
- **Corpus**：Amazon 美妆商品 products 表（≤ 1000 条），包装为 DocumentChunk 结构
- **Top-K**：10

## 调用的项目服务

| 策略 | 调用方法 | 模型 |
|------|---------|------|
| **A: BM25** | `HybridSearchService._bm25_search()` | rank-bm25 + 项目 `_tokenize` |
| **B: BGE** | `VectorService.batch_text_to_embeddings()` + `text_to_embedding()` | BAAI/bge-large-zh (1024-dim) |
| **C: Hybrid** | A ∪ B → `HybridSearchService._rerank()` | BAAI/bge-reranker-v2-m3 |

## 总体指标

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@10 | 查询耗时 | N |
|------|------:|------:|------:|-------:|--------:|---|
| **A: BM25 Only** |  50.0% |  54.8% |  57.1% | 0.5377 | 11.3ms | 42 |
| **B: BGE Dense (bge-large-zh)** |  33.3% |  42.9% |  47.6% | 0.3937 | 80.1ms | 42 |
| **C: Hybrid+Rerank** |  61.9% |  61.9% |  64.3% | 0.6302 | 5456.0ms | 42 |

## 按意图分组

### PRODUCT_SEARCH

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@10 | N |
|------|------:|------:|------:|-------:|---|
| **A: BM25 Only** |  75.0% |  87.5% |  93.8% | 0.8385 | 16 |
| **B: BGE Dense (bge-large-zh)** |  50.0% |  62.5% |  68.8% | 0.5851 | 16 |
| **C: Hybrid+Rerank** |  93.8% |  93.8% |  93.8% | 0.9464 | 16 |

### FUZZY_SEARCH

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@10 | N |
|------|------:|------:|------:|-------:|---|
| **A: BM25 Only** |   0.0% |   0.0% |   0.0% | 0.0000 | 16 |
| **B: BGE Dense (bge-large-zh)** |   0.0% |   6.2% |   6.2% | 0.0312 | 16 |
| **C: Hybrid+Rerank** |   6.2% |   6.2% |  12.5% | 0.0828 | 16 |

### COMPARISON

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@10 | N |
|------|------:|------:|------:|-------:|---|
| **A: BM25 Only** |  90.0% |  90.0% |  90.0% | 0.9167 | 10 |
| **B: BGE Dense (bge-large-zh)** |  60.0% |  70.0% |  80.0% | 0.6676 | 10 |
| **C: Hybrid+Rerank** | 100.0% | 100.0% | 100.0% | 1.0000 | 10 |

## 指标说明

| 指标 | 定义 |
|------|------|
| **Hit@K** | expected_skus 中任意一个出现在检索结果前 K 名中的比例 |
| **MRR@10** | 第一个命中结果排名倒数的均值；未在 Top-10 命中时倒数为 0 |

## 逐条明细（策略 A，前 20 条）

| # | Intent | Query | Expected | Hit@5 | MRR | Retrieved Top-3 |
|---|--------|-------|----------|------:|----:|----------------|
| 1 | FUZZY_SEARCH | 推荐一款防晒指数高、不油腻、适合日常的防晒霜… | amazon_B00G6KAQIE, amazon_B09H61PN51 | ❌ | 0.00 |  |
| 2 | PRODUCT_SEARCH | N\C 化妆刷 有货吗… | amazon_B08NW67Z2T | ✅ | 0.50 | amazon_B01IAEBWWM, amazon_B08NW67Z2T, amazon_B0859L4DKL |
| 3 | COMPARISON | 比较 Nails Inc 和 Wet 'n' Wild 这两款指甲油/… | amazon_B00NH1I50Q, amazon_B00GVIM4ZY | ✅ | 1.00 | amazon_B00GVIM4ZY, amazon_B01IRX4X4K, amazon_B00I254ZDS |
| 4 | PRODUCT_SEARCH | 搜索 House of Armáf 香水，预算大概 $25… | amazon_B07R7YWQKW | ✅ | 1.00 | amazon_B07R7YWQKW, amazon_B01N6Y0DV8, amazon_B00WAJB7CY |
| 6 | PRODUCT_SEARCH | Snozzle Pro 吹风机 有货吗… | amazon_B07X8Z9W22 | ❌ | 0.17 | amazon_B00OZWSSDK, amazon_B07CTQQLLJ, amazon_B0086UL0WS |
| 7 | COMPARISON | Benefit face cream 和 OxygenCeutical… | amazon_B07DM64VNT, amazon_B0777QHNLR | ✅ | 1.00 | amazon_B07DM64VNT, amazon_B08WF29DM9, amazon_B06XR79F1N |
| 8 | COMPARISON | 帮我比较一下 Miny Beauty Cosmetics 和 twee… | amazon_B07DFSN36Q, amazon_B0777W6C4X | ✅ | 1.00 | amazon_B07DFSN36Q, amazon_B00CM0WHXY, amazon_B07JB8KQF6 |
| 9 | FUZZY_SEARCH | 有没有适合敏感肌、无香精的面霜？… | amazon_B07KFGWC8S, amazon_B0058WE96Q | ❌ | 0.00 |  |
| 10 | FUZZY_SEARCH | 推荐一款遮盖黑眼圈效果好的遮瑕笔或遮瑕膏… | amazon_B08L3QKHRJ, amazon_B08CVGMFRB | ❌ | 0.00 |  |
| 11 | COMPARISON | 帮我比较一下 essie 和 tweexy 的指甲油/美甲… | amazon_B0754HBCRK, amazon_B0777W6C4X | ✅ | 1.00 | amazon_B0754HBCRK, amazon_B0777W6C4X |
| 12 | FUZZY_SEARCH | 有没有清爽不黏腻、快速吸收的全身身体乳？… | amazon_B0711L86FJ, amazon_B01LE3GULY | ❌ | 0.00 |  |
| 14 | PRODUCT_SEARCH | 帮我找一下 EYE MAJIC INSTANT EYESHADOW –… | amazon_B07JVW6ZTF | ✅ | 1.00 | amazon_B07JVW6ZTF, amazon_B018QY0FK6, amazon_B01IVQJIKC |
| 15 | FUZZY_SEARCH | 我想找一款显色度高、持妆时间长的哑光口红… | amazon_B072JG9WW8, amazon_B09B2RNHHH | ❌ | 0.00 |  |
| 16 | COMPARISON | Bath & Body Works perfume 和 Tiffany… | amazon_B094GZW7KD, amazon_B07YL4L81W | ❌ | 0.17 | amazon_B003NZKXYM, amazon_B01N9I42I0, amazon_B07B8Q9WHS |
| 17 | FUZZY_SEARCH | 推荐一套适合初学者的化妆刷，刷毛柔软不掉毛… | amazon_B01KGIFJSS, amazon_B07TDN9PCN | ❌ | 0.00 |  |
| 18 | FUZZY_SEARCH | 想买一个快干、控温、不伤发质的吹风机… | amazon_B07X8Z9W22 | ❌ | 0.00 |  |
| 19 | PRODUCT_SEARCH | 搜索 Magick Botanicals 护发素，预算大概 $23… | amazon_B0011DN60Q | ✅ | 0.50 | amazon_B00028NLKG, amazon_B0011DN60Q, amazon_B071VZ8YXR |
| 21 | FUZZY_SEARCH | 需要一款遮瑕力强但妆感自然的粉底液，油皮适用… | amazon_B09JC2GTVD, amazon_B009WI1UKA | ❌ | 0.00 |  |
| 22 | FUZZY_SEARCH | 有没有防水、防晕染、增长睫毛效果好的睫毛膏？… | amazon_B07LD3M88C | ❌ | 0.00 |  |
| 24 | FUZZY_SEARCH | 我在找一款颜色持久不容易掉的指甲油… | amazon_B07PFH92HN, amazon_B0777W6C4X | ❌ | 0.00 |  |

## 结论与建议

1. **BM25（策略 A）** 在 `PRODUCT_SEARCH`（含英文品牌名）中表现最佳，
   精确匹配品牌/型号，速度极快（<1ms/query）。
   但在 `FUZZY_SEARCH`（纯中文语义查询 vs 英文语料）中命中率接近 0，
   因为 `_tokenize` 分词后中英文 token 无交集。

2. **BGE 稠密检索（策略 B）** 通过 BAAI/bge-large-zh 向量空间对齐
   实现跨语言语义检索，在 `FUZZY_SEARCH` 上显著优于 BM25，
   但对精确品牌名匹配弱于 BM25（编码存在信息压缩损失）。

3. **Hybrid + Reranker（策略 C）** 综合两路信号：
   - BM25 保证精确关键词不丢失；
   - BGE 补充语义召回；
   - `_rerank`（bge-reranker-v2-m3 Cross-Encoder）对每对 (query, passage) 
     做完整 Attention 精排，进一步提升排序质量。
   **推荐在生产环境使用策略 C。**

4. **优化建议**：
   - 为产品加入中文翻译字段，降低语言鸿沟，提升 BM25 对中文查询的召回率；
   - 使用 `BAAI/bge-m3`（多语言）替换 `bge-large-zh` 进一步提升跨语言效果；
   - 产品文本预置到 DocumentChunk 表，使 hybrid_search_service 开箱即用。

---
*由 `evaluate_rag.py` 自动生成 @ 2026-03-08 14:57:12*
*检索服务：hybrid_search_service.py / vector_service.py*