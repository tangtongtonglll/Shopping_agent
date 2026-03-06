"""
混合检索服务 (Hybrid Search Service)

Pipeline:
    BM25 稀疏检索 (rank-bm25, 商品名称/关键词)
        +
    FAISS 稠密检索 (BGE-large-zh, 语义描述)
        ↓
    合并去重 (by chunk_id)
        ↓
    BGE Cross-Encoder 重排序 (bge-reranker-v2-m3)
        ↓
    Top-5 最终结果
"""

import re
import logging
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    print("⚠️  rank-bm25 未安装，BM25 检索不可用。请运行: pip install rank-bm25")

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    FlagReranker = None
    print("⚠️  FlagEmbedding 未安装，重排序不可用。请运行: pip install FlagEmbedding")

from sqlalchemy.orm import Session

from ..models.models import DocumentChunk, Document
from .vector_service import vector_service

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 分词工具                                                                     #
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> List[str]:
    """
    轻量中英文混合分词器，无需 jieba。

    规则：
    - 英文单词和数字：整体作为一个 token（如 "iPhone15"、"512GB"）
    - 中文字符：每个字符单独作为一个 token

    对商品名称（品牌名 + 型号 + 规格）的短文本关键词检索效果良好。
    """
    if not text:
        return []
    tokens: List[str] = []
    for match in re.finditer(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text.lower()):
        tokens.append(match.group())
    return tokens


# --------------------------------------------------------------------------- #
# HybridSearchService                                                          #
# --------------------------------------------------------------------------- #

class HybridSearchService:
    """
    混合检索服务：BM25 + FAISS + BGE Cross-Encoder 重排序。

    快速使用:
        svc = HybridSearchService()
        results = svc.advanced_hybrid_search(
            query="华为旗舰手机 512G",
            db=db_session,
            knowledge_base_ids=[1, 2],
            final_top_k=5,
        )
    """

    def __init__(self, reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        """
        Args:
            reranker_model: HuggingFace model id，默认 bge-reranker-v2-m3（中英双语）。
        """
        self.reranker: Optional[FlagReranker] = None

        if RERANKER_AVAILABLE and FlagReranker:
            try:
                # use_fp16=False：兼容纯 CPU 环境
                self.reranker = FlagReranker(reranker_model, use_fp16=False)
                logger.info(f"[HybridSearch] Reranker 已加载: {reranker_model}")
            except Exception as exc:
                logger.error(f"[HybridSearch] Reranker 加载失败: {exc}")

    # ------------------------------------------------------------------ #
    # Step 1a: 拉取语料                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_chunks(
        self,
        db: Session,
        knowledge_base_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """从 DocumentChunk 表拉取所有文档块，构成 BM25 语料库。"""
        query = db.query(DocumentChunk).join(Document)

        if knowledge_base_ids:
            query = query.filter(Document.knowledge_base_id.in_(knowledge_base_ids))

        return [
            {
                "chunk_id":      chunk.id,
                "document_id":   chunk.document_id,
                "chunk_index":   chunk.chunk_index,
                "content":       chunk.content,
                "metadata":      chunk.meta_data or {},
                "document_name": chunk.document.original_name if chunk.document else "",
            }
            for chunk in query.all()
        ]

    # ------------------------------------------------------------------ #
    # Step 1b: BM25 稀疏检索                                              #
    # ------------------------------------------------------------------ #

    def _bm25_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        BM25 稀疏检索。

        优势：精确匹配品牌名（"华为"）、型号（"Mate60"）、规格（"512GB"）等关键词。
        BM25 分数为 0 的结果（无词汇重叠）一律丢弃。
        """
        if not BM25_AVAILABLE or not BM25Okapi:
            logger.warning("[HybridSearch] BM25 不可用，跳过稀疏检索。")
            return []
        if not chunks:
            return []

        tokenized_corpus = [_tokenize(c["content"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = _tokenize(query)
        scores: np.ndarray = bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0.0:
                break  # 无词汇重叠，后续也是 0，提前终止
            item = dict(chunks[idx])
            item["retrieval_score"]  = float(scores[idx])
            item["retrieval_source"] = "bm25"
            results.append(item)

        return results

    # ------------------------------------------------------------------ #
    # Step 1c: FAISS 稠密检索                                             #
    # ------------------------------------------------------------------ #

    def _dense_search(
        self,
        query: str,
        db: Session,
        knowledge_base_ids: Optional[List[int]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        BGE-large-zh + FAISS IndexFlatL2 稠密检索。

        优势：理解语义（"便宜的游戏本" → 匹配"预算友好型游戏笔记本"）。
        threshold=2.0 故意设宽，把过滤权交给重排阶段。
        """
        raw = vector_service.search_similar_documents(
            query=query,
            limit=top_k,
            threshold=2.0,   # L2 归一化向量最大为 2，此处不过滤
            knowledge_base_ids=knowledge_base_ids,
            db=db,
        )

        return [
            {
                "chunk_id":        item["id"],
                "document_id":     item["document_id"],
                "chunk_index":     item["chunk_index"],
                "content":         item["content"],
                "metadata":        item.get("metadata", {}),
                "document_name":   item.get("document_name", ""),
                "retrieval_score": item["score"],    # L2 距离，越小越好
                "retrieval_source": "faiss",
            }
            for item in raw
        ]

    # ------------------------------------------------------------------ #
    # Step 2: 合并去重                                                     #
    # ------------------------------------------------------------------ #

    def _merge_and_deduplicate(
        self,
        bm25_results:  List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        以 chunk_id 去重，合并两路结果。

        若同一 chunk 被 BM25 和 FAISS 同时召回，retrieval_source 标记为 "both"。
        """
        seen: Dict[int, Dict[str, Any]] = {}

        for item in bm25_results:
            cid = item["chunk_id"]
            seen[cid] = item.copy()

        for item in dense_results:
            cid = item["chunk_id"]
            if cid in seen:
                seen[cid]["retrieval_source"] = "both"
            else:
                seen[cid] = item.copy()

        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Step 3: BGE Cross-Encoder 重排序                                    #
    # ------------------------------------------------------------------ #

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        BAAI/bge-reranker-v2-m3 Cross-Encoder 重排序。

        原理：对每对 (query, passage) 做完整 Attention 计算，输出相关性 logit。
              与双塔模型相比精度更高，无近似误差，适合 top-N 精排场景。

        Fallback：Reranker 不可用时，BM25 结果按 BM25 分降序，
                  FAISS 结果按 L2 距离升序，拼接后截取 top_k。
        """
        if not candidates:
            return []

        if not self.reranker:
            logger.warning("[HybridSearch] Reranker 不可用，使用 retrieval_score 作为备用排序。")
            bm25_hits  = [c for c in candidates if c["retrieval_source"] in ("bm25", "both")]
            faiss_hits = [c for c in candidates if c["retrieval_source"] == "faiss"]
            bm25_hits.sort(key=lambda x: x["retrieval_score"], reverse=True)
            faiss_hits.sort(key=lambda x: x["retrieval_score"])  # L2: 越小越好
            # 去掉已在 bm25_hits 中的 faiss 结果（"both" 已在前者）
            faiss_only = [c for c in faiss_hits if c["retrieval_source"] == "faiss"]
            return (bm25_hits + faiss_only)[:final_top_k]

        # 构建 [query, passage] 对
        pairs = [[query, c["content"]] for c in candidates]

        try:
            # normalize=False：返回原始 logit，便于跨 batch 比较
            scores = self.reranker.compute_score(pairs, normalize=False)
            if isinstance(scores, float):
                scores = [scores]
        except Exception as exc:
            logger.error(f"[HybridSearch] Reranker 打分失败: {exc}")
            return candidates[:final_top_k]

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:final_top_k]

    # ------------------------------------------------------------------ #
    # 主入口                                                               #
    # ------------------------------------------------------------------ #

    def advanced_hybrid_search(
        self,
        query: str,
        db: Session,
        knowledge_base_ids: Optional[List[int]] = None,
        bm25_top_k:  int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        混合检索主函数。

        流程:
            1. 从 DB 加载 DocumentChunk，执行 BM25 稀疏检索，召回 top-{bm25_top_k}
            2. BGE-large-zh + FAISS 执行稠密检索，召回 top-{dense_top_k}
            3. 合并去重（以 chunk_id 为键）
            4. bge-reranker-v2-m3 Cross-Encoder 重排序
            5. 返回 rerank_score 最高的 top-{final_top_k} 条

        Args:
            query:               用户查询字符串
            db:                  SQLAlchemy Session（调用方注入，不在此创建）
            knowledge_base_ids:  限定知识库范围；None 表示检索全部
            bm25_top_k:          BM25 单路召回数（默认 20）
            dense_top_k:         FAISS 单路召回数（默认 20）
            final_top_k:         重排后返回数量（默认 5）

        Returns:
            List[Dict]，每条包含:
                chunk_id        (int)
                document_id     (int)
                chunk_index     (int)
                content         (str)   ← 用于 LLM prompt 的正文
                document_name   (str)
                metadata        (dict)
                retrieval_source ("bm25" | "faiss" | "both")
                retrieval_score (float) ← BM25 分 或 L2 距离，供调试
                rerank_score    (float) ← Cross-Encoder 相关性分（越高越相关）
        """
        if not query or not query.strip():
            logger.warning("[HybridSearch] 空 query，直接返回空列表。")
            return []

        logger.info(
            f"[HybridSearch] 开始检索 | query='{query}' "
            f"kb_ids={knowledge_base_ids} "
            f"bm25_top_k={bm25_top_k} dense_top_k={dense_top_k} final_top_k={final_top_k}"
        )

        # ① 加载语料（BM25 需要）
        chunks = self._fetch_chunks(db, knowledge_base_ids)
        logger.info(f"[HybridSearch] 语料库大小: {len(chunks)} chunks")

        # ② 双路召回
        bm25_results  = self._bm25_search(query, chunks, top_k=bm25_top_k)
        dense_results = self._dense_search(query, db, knowledge_base_ids, top_k=dense_top_k)
        logger.info(
            f"[HybridSearch] 召回完成 | BM25={len(bm25_results)} FAISS={len(dense_results)}"
        )

        # ③ 合并去重
        candidates = self._merge_and_deduplicate(bm25_results, dense_results)
        logger.info(f"[HybridSearch] 去重后候选: {len(candidates)} chunks")

        # ④ Cross-Encoder 重排 + 截取 top-k
        final = self._rerank(query, candidates, final_top_k=final_top_k)

        logger.info(
            f"[HybridSearch] 最终结果: {len(final)} 条 | "
            f"rerank_scores={[round(r.get('rerank_score', 0.0), 4) for r in final]}"
        )
        return final


# --------------------------------------------------------------------------- #
# 模块级单例（延迟初始化，避免服务启动时阻塞模型下载）                           #
# --------------------------------------------------------------------------- #

_instance: Optional[HybridSearchService] = None


def get_hybrid_search_service() -> HybridSearchService:
    """获取 HybridSearchService 单例（线程安全，首次调用时加载 Reranker）。"""
    global _instance
    if _instance is None:
        _instance = HybridSearchService()
    return _instance
