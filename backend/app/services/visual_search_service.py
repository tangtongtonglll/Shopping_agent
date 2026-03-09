"""
视觉搜索和商品识别服务
支持图像搜索、商品识别和视觉推荐

速度优化：
  - 仅 1 次视觉模型 API 调用（原来 3 次）
  - 特征提取与视觉模型调用并行执行
  - K-means 采样像素（避免全量运算）

相似度优化：
  - 视觉模型输出结构化 JSON（含英文关键词），直接用于 SQL LIKE 查询
  - 优先过滤同品类商品（title 关键词匹配）
  - 量化评分：title匹配(55%) + desc匹配(20%) + 评分(15%) + 热度(10%)
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import os
import re
import json
import logging
from datetime import datetime
import asyncio
from io import BytesIO
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageFont = None
    print("⚠️  PIL未安装，视觉搜索功能将不可用。请运行: pip install Pillow")

if TYPE_CHECKING:
    from PIL import Image as PILImage
else:
    PILImage = Image

import base64
import numpy as np
from ..models.models import KnowledgeBase, Document, DocumentChunk
from ..models.ecommerce_models import Product, ProductImage
from ..services.llm_service import LLMService
from ..services.vector_service import vector_service
from ..core.config import settings

logger = logging.getLogger(__name__)

# ANSI 颜色常量
_VP = "\033[35m\033[1m"   # 紫色加粗
_G  = "\033[32m\033[1m"   # 绿色加粗
_Y  = "\033[33m\033[1m"   # 黄色加粗
_R  = "\033[0m"            # 重置


class VisualSearchEngine:
    """视觉搜索引擎"""

    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
        self.supported_formats = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
        self.max_image_size = 10 * 1024 * 1024  # 10MB

    # ──────────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────────

    async def search_by_image(self, image_data: bytes, search_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """通过图像搜索商品（优化版：1 次 API 调用，并行执行）"""
        import time
        t0 = time.time()
        print(f"\n  {_VP}{'─'*52}{_R}")
        print(f"  {_VP}🖼️  Visual Search  size={len(image_data)//1024}KB{_R}")
        print(f"  {_VP}{'─'*52}{_R}")

        try:
            # 1. 验证 + 预处理（快，< 0.1s）
            validation = self._validate_image(image_data)
            if not validation["valid"]:
                print(f"  \033[31m[vs]{_R} ❌ 验证失败: {validation['error']}")
                return {"success": False, "error": validation["error"]}
            print(f"  {_VP}[vs]{_R} ✅ 图像有效 format={validation.get('format')} size={validation.get('size')}")

            processed = await self._preprocess_image(image_data)

            # 2. 特征提取 & 视觉模型 并行（节省串行等待时间）
            print(f"  {_VP}[vs]{_R} ⚡ 并行：特征提取 + 视觉模型识别…")
            t1 = time.time()
            image_features, image_info = await asyncio.gather(
                self._extract_image_features(processed),
                self._generate_image_info(processed),
            )
            t_parallel = time.time() - t1
            kws   = image_info.get("keywords", [])
            ptype = image_info.get("type", "")
            zh    = image_info.get("zh", "")
            print(f"  {_VP}[vs]{_R} ✅ 并行完成 ({t_parallel:.1f}s)")
            print(f"  {_VP}[vs]{_R} 🏷️  识别结果: type={ptype!r}  keywords={kws}  zh={zh[:40]!r}")

            # 3. 关键词 SQL 检索候选商品
            t2 = time.time()
            candidates = self._search_by_keywords(kws, ptype, search_options)
            print(f"  {_VP}[vs]{_R} 🗄️  候选商品 {len(candidates)} 条 ({time.time()-t2:.2f}s)")

            # 4. 量化评分 + 排序
            t3 = time.time()
            ranked = self._rank_and_score(candidates, kws, ptype)
            print(f"  {_VP}[vs]{_R} 📊 评分排序完成 ({time.time()-t3:.2f}s)")

            # 5. 打印 Top-5 评分明细
            self._print_score_breakdown(ranked[:5], kws)

            total = time.time() - t0
            print(f"  {_G}{'─'*52}{_R}")
            print(f"  {_G}✅ 图搜完成  耗时={total:.1f}s  结果={len(ranked)} 条{_R}")
            print(f"  {_G}{'─'*52}{_R}\n")

            return {
                "success": True,
                "data": {
                    "image_features":   image_features,
                    "image_description": zh or f"{ptype} {' '.join(kws)}",
                    "similar_products":  ranked[:10],
                    "visual_analysis":   {"product_type": ptype, "keywords": kws},
                    "search_options":    search_options or {},
                },
            }

        except Exception as e:
            logger.error(f"Error in visual search: {e}", exc_info=True)
            print(f"  \033[31m[vs]{_R} ❌ 图搜异常: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # 图像预处理
    # ──────────────────────────────────────────────────────────────

    def _validate_image(self, image_data: bytes) -> Dict[str, Any]:
        if len(image_data) > self.max_image_size:
            return {"valid": False, "error": f"图像超过 {self.max_image_size//(1024*1024)}MB 限制"}
        try:
            img = Image.open(BytesIO(image_data))
            fmt = (img.format or "").lower()
            if fmt not in self.supported_formats:
                return {"valid": False, "error": f"不支持的格式: {fmt}"}
            size, mode = img.size, img.mode
            img.verify()  # verify 会关闭 image，故先保存 size/mode
            return {"valid": True, "format": fmt, "size": size, "mode": mode}
        except Exception as e:
            return {"valid": False, "error": f"图像损坏: {e}"}

    async def _preprocess_image(self, image_data: bytes) -> "PILImage.Image":
        img = Image.open(BytesIO(image_data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        max_side = 600
        if max(img.size) > max_side:
            r = max_side / max(img.size)
            img = img.resize((int(img.size[0] * r), int(img.size[1] * r)), Image.Resampling.LANCZOS)
        return img

    # ──────────────────────────────────────────────────────────────
    # 特征提取（numpy，无 IO）
    # ──────────────────────────────────────────────────────────────

    async def _extract_image_features(self, image: "PILImage.Image") -> Dict[str, Any]:
        arr = np.array(image)
        return {
            "color_histogram":  self._calculate_color_histogram(arr),
            "edge_density":     self._calculate_edge_density(arr),
            "texture_features": self._calculate_texture_features(arr),
            "shape_features":   self._calculate_shape_features(arr),
            "dominant_colors":  self._get_dominant_colors(arr),
        }

    def _calculate_color_histogram(self, arr: np.ndarray) -> Dict[str, list]:
        return {
            "red":   np.histogram(arr[:, :, 0], bins=32, range=(0, 256))[0].tolist(),
            "green": np.histogram(arr[:, :, 1], bins=32, range=(0, 256))[0].tolist(),
            "blue":  np.histogram(arr[:, :, 2], bins=32, range=(0, 256))[0].tolist(),
        }

    def _calculate_edge_density(self, arr: np.ndarray) -> float:
        try:
            gray = arr.mean(axis=2)
            # Sobel via slicing (fast approximation)
            gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
            gy = np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1])
            edge = gx + gy
            thresh = edge.mean() + edge.std()
            return float((edge > thresh).mean())
        except Exception:
            return 0.0

    def _calculate_texture_features(self, arr: np.ndarray) -> Dict[str, float]:
        """向量化 LBP 纹理特征"""
        try:
            gray = arr.mean(axis=2)
            c = gray[1:-1, 1:-1]
            neighbors = [
                gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
                gray[1:-1, 2:],
                gray[2:,   2:],   gray[2:,   1:-1], gray[2:,   0:-2],
                gray[1:-1, 0:-2],
            ]
            lbp = sum((nb >= c).astype(np.uint8) * (1 << i) for i, nb in enumerate(neighbors))
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            p = hist / (lbp.size or 1)
            return {
                "lbp_uniformity": float(np.sum(p ** 2)),
                "lbp_entropy":    float(-np.sum(p * np.log2(p + 1e-10))),
            }
        except Exception:
            return {"lbp_uniformity": 0.0, "lbp_entropy": 0.0}

    def _calculate_shape_features(self, arr: np.ndarray) -> Dict[str, float]:
        h, w = arr.shape[:2]
        return {"aspect_ratio": w / h, "compactness": (h * w) / (2 * (h + w)) ** 2}

    def _get_dominant_colors(self, arr: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """采样像素后做 K-means，避免全量运算（原来对 640k 像素全量，现采样 2000）"""
        try:
            pixels = arr.reshape(-1, 3).astype(np.float32)
            pixels = pixels[~np.isnan(pixels).any(axis=1)]
            if len(pixels) < k:
                return []
            # 随机采样 2000 个像素（足够代表主色调）
            idx = np.random.choice(len(pixels), min(2000, len(pixels)), replace=False)
            pixels = pixels[idx]
            centers = pixels[:k].copy()
            for _ in range(10):
                dists  = np.sqrt(((pixels[:, None] - centers) ** 2).sum(axis=2))
                labels = np.argmin(dists, axis=1)
                new_c  = np.array([
                    pixels[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
                    for i in range(k)
                ])
                if np.allclose(centers, new_c, atol=1.0):
                    break
                centers = new_c
            result = []
            for c in centers:
                if np.isnan(c).any():
                    continue
                r, g, b = (int(np.clip(c[i], 0, 255)) for i in range(3))
                result.append({"r": r, "g": g, "b": b, "hex": f"#{r:02x}{g:02x}{b:02x}"})
            return result
        except Exception as e:
            logger.error(f"dominant colors error: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # 视觉模型：结构化输出
    # ──────────────────────────────────────────────────────────────

    async def _generate_image_info(self, image: "PILImage.Image") -> Dict[str, Any]:
        """
        调用视觉大模型，输出结构化 JSON。

        返回格式：
          {
            "type": "lipstick",           # 英文产品类型
            "keywords": ["lip", "matte"], # 英文关键词（用于 SQL LIKE 查询）
            "zh": "红色哑光唇膏"           # 中文简短描述（供前端展示）
          }
        """
        _vp = _VP
        try:
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

            prompt = (
                "你是一个电商商品识别助手，请分析图中的美妆/个护商品，"
                "只输出以下 JSON，不要任何额外说明：\n"
                '{"type":"英文产品类型(如lipstick/shampoo/cream/perfume等单词)",'
                '"keywords":["英文关键词1","英文关键词2","英文关键词3"],'
                '"zh":"10字内中文描述"}'
            )

            print(f"  {_vp}[vs]{_R} 👁️  调用视觉模型…")
            result = await self.llm_service.analyze_image(image_url=data_url, prompt=prompt)
            raw = result.get("analysis", "") if isinstance(result, dict) else str(result)
            print(f"  {_vp}[vs]{_R} 📝 视觉模型原始输出: {raw[:120]!r}")

            parsed = self._parse_json_response(raw)
            # 确保字段类型正确
            parsed.setdefault("type", "")
            parsed.setdefault("keywords", [])
            parsed.setdefault("zh", raw[:50])
            if isinstance(parsed["keywords"], str):
                parsed["keywords"] = [k.strip() for k in parsed["keywords"].split(",") if k.strip()]
            return parsed

        except Exception as e:
            logger.error(f"_generate_image_info error: {e}")
            print(f"  \033[31m[vs]{_R} ❌ 视觉模型失败: {e}")
            return {"type": "", "keywords": [], "zh": "美妆商品"}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """健壮的 JSON 解析：支持 markdown 代码块、裸 JSON、或回退"""
        # 1. 直接解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2. 从 ```json ... ``` 提取
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 3. 找第一个 {...}
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # 4. 回退
        return {"type": "", "keywords": text.split()[:4], "zh": text[:50]}

    # ──────────────────────────────────────────────────────────────
    # 关键词 SQL 检索（替换原先的随机 200 条）
    # ──────────────────────────────────────────────────────────────

    def _search_by_keywords(
        self,
        keywords: List[str],
        product_type: str,
        search_options: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        用英文关键词在 title / description 上做 LIKE 查询。

        策略：
          1. title LIKE '%type%' OR title LIKE '%kw%' → 最多 80 条（type 匹配最优先）
          2. 如果 title 匹配不足 5 条，追加 description LIKE 匹配
          3. 如果总量仍不足 5 条，追加随机 50 条作为兜底
        """
        from sqlalchemy import text as sa_text

        all_terms = [t.strip().lower() for t in ([product_type] + keywords) if t.strip()]
        all_terms = list(dict.fromkeys(all_terms))[:6]  # 去重，最多 6 个词

        base_filter = "image_url IS NOT NULL"
        if search_options:
            if search_options.get("price_range"):
                mn, mx = search_options["price_range"]
                base_filter += f" AND price BETWEEN {mn} AND {mx}"

        product_ids_seen = set()
        rows = []

        # ── 阶段 1：title 匹配（同品类过滤关键一步）──
        if all_terms:
            title_conds = " OR ".join(f"LOWER(title) LIKE :t{i}" for i in range(len(all_terms)))
            params = {f"t{i}": f"%{kw}%" for i, kw in enumerate(all_terms)}
            sql = (
                f"SELECT product_id, title, brand, category, price, image_url, product_url, "
                f"description, rating, review_count FROM products "
                f"WHERE {base_filter} AND ({title_conds}) LIMIT 80"
            )
            rows = self.db.execute(sa_text(sql), params).fetchall()
            product_ids_seen = {r[0] for r in rows}

        # ── 阶段 2：description 补充匹配（当 title 匹配太少时）──
        if len(rows) < 5 and all_terms:
            exc = ",".join(f"'{pid}'" for pid in product_ids_seen) or "''"
            desc_conds = " OR ".join(f"LOWER(description) LIKE :d{i}" for i in range(len(all_terms)))
            params2 = {f"d{i}": f"%{kw}%" for i, kw in enumerate(all_terms)}
            sql2 = (
                f"SELECT product_id, title, brand, category, price, image_url, product_url, "
                f"description, rating, review_count FROM products "
                f"WHERE {base_filter} AND product_id NOT IN ({exc}) AND ({desc_conds}) LIMIT 50"
            )
            extra = self.db.execute(sa_text(sql2), params2).fetchall()
            rows = list(rows) + list(extra)
            product_ids_seen = {r[0] for r in rows}

        # ── 阶段 3：随机兜底（关键词完全没匹配时）──
        if len(rows) < 5:
            exc = ",".join(f"'{pid}'" for pid in product_ids_seen) or "''"
            sql3 = (
                f"SELECT product_id, title, brand, category, price, image_url, product_url, "
                f"description, rating, review_count FROM products "
                f"WHERE {base_filter} AND product_id NOT IN ({exc}) ORDER BY RANDOM() LIMIT 30"
            )
            rows = list(rows) + list(self.db.execute(sa_text(sql3)).fetchall())

        return self._rows_to_products(rows)

    def _rows_to_products(self, rows) -> List[Dict[str, Any]]:
        return [
            {
                "product_id":  r[0],
                "title":       r[1] or "",
                "brand":       r[2] or "",
                "category":    r[3] or "",
                "price":       r[4] or 0.0,
                "image_url":   r[5] or "",
                "product_url": r[6] or "",
                "description": r[7] or "",
                "rating":      r[8] or 0.0,
                "review_count":r[9] or 0,
            }
            for r in rows
        ]

    # ──────────────────────────────────────────────────────────────
    # 量化评分排序
    # ──────────────────────────────────────────────────────────────

    def _rank_and_score(
        self,
        products: List[Dict[str, Any]],
        keywords: List[str],
        product_type: str,
    ) -> List[Dict[str, Any]]:
        """
        量化评分公式（满分 1.0）：

          type_match    × 0.50  ← 产品类型关键词命中 title（核心：保证同品类）
          kw_title      × 0.20  ← 其他关键词命中 title
          kw_desc       × 0.15  ← 其他关键词命中 description
          rating_score  × 0.10  ← Amazon 评分（/5.0）
          pop_score     × 0.05  ← 热度（review_count 归一化）

        设计思路：
        - product_type 单独给 50% 权重，使「同品类」成为最强信号
        - 只有 product_type 在 title 中命中，才能获得高分
        - 避免 'lip' 匹配到 'lipstick' 内部导致无关产品高分
        """
        ptype = product_type.lower().strip()

        # 其他关键词（去重，排除 product_type）
        seen: set = {ptype} if ptype else set()
        other_kws: List[str] = []
        for kw in keywords:
            kl = kw.lower().strip()
            if kl and kl not in seen:
                seen.add(kl)
                other_kws.append(kl)
        n_other = max(len(other_kws), 1)

        max_reviews = max((p["review_count"] or 0 for p in products), default=1) or 1

        scored = []
        for p in products:
            title_lower = p["title"].lower()
            desc_lower  = (p["description"] or "").lower()

            # 产品类型是否命中 title（精确子串匹配）
            type_hit = 1.0 if ptype and ptype in title_lower else (
                0.4 if ptype and ptype in desc_lower else 0.0
            )

            # 其他关键词命中率
            other_title_hits = sum(1 for kw in other_kws if kw in title_lower)
            other_desc_hits  = sum(1 for kw in other_kws if kw in desc_lower)
            kw_title = other_title_hits / n_other
            kw_desc  = other_desc_hits  / n_other

            rating_score = min((p["rating"] or 0) / 5.0, 1.0)
            pop_score    = min((p["review_count"] or 0) / max_reviews, 1.0)

            total = round(
                type_hit     * 0.50
                + kw_title   * 0.20
                + kw_desc    * 0.15
                + rating_score * 0.10
                + pop_score  * 0.05,
                4,
            )

            pc = p.copy()
            pc["similarity_score"] = total
            pc["score_breakdown"] = {
                "type_match":    round(type_hit     * 0.50, 3),
                "kw_title":      round(kw_title     * 0.20, 3),
                "kw_desc":       round(kw_desc      * 0.15, 3),
                "rating":        round(rating_score * 0.10, 3),
                "popularity":    round(pop_score    * 0.05, 3),
            }
            pc["match_keywords"] = [kw for kw in ([ptype] + other_kws) if kw and kw in title_lower]
            scored.append(pc)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored

    def _print_score_breakdown(self, top: List[Dict[str, Any]], keywords: List[str]) -> None:
        """在终端打印 Top-N 评分明细"""
        if not top:
            print(f"  {_Y}[vs]{_R} ⚠️  无匹配结果")
            return
        print(f"  {_VP}[vs]{_R} 📈 Top-{len(top)} 相似度明细 (keywords={keywords}):")
        for i, p in enumerate(top, 1):
            bd = p.get("score_breakdown", {})
            pct = int(p['similarity_score'] * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(
                f"      #{i} [{bar}] {pct:3d}%  "
                f"类型={bd.get('type_match',0):.2f} "
                f"标题词={bd.get('kw_title',0):.2f} "
                f"描述词={bd.get('kw_desc',0):.2f} "
                f"⭐={bd.get('rating',0):.2f} "
                f"热度={bd.get('popularity',0):.2f}  "
                f"「{p['title'][:38]}」"
            )

    # ──────────────────────────────────────────────────────────────
    # 以下方法保持兼容（供 API 路由调用）
    # ──────────────────────────────────────────────────────────────

    async def recognize_product(self, image_data: bytes) -> Dict[str, Any]:
        """商品识别（复用 search_by_image）"""
        result = await self.search_by_image(image_data)
        if not result["success"]:
            return result
        products = result["data"].get("similar_products", [])
        if products and products[0]["similarity_score"] > 0.6:
            return {
                "success": True,
                "data": {
                    "recognized": True,
                    "product": products[0],
                    "confidence": products[0]["similarity_score"],
                    "alternative_matches": products[1:4],
                },
            }
        return {
            "success": True,
            "data": {
                "recognized": False,
                "similar_products": products[:5],
                "suggestions": ["尝试更清晰的图片", "确保商品主体居中"],
            },
        }

    async def create_visual_search_index(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建视觉搜索索引（当前版本无需预计算，直接走关键词查询）"""
        return {
            "success": True,
            "data": {
                "indexed_products": len(products),
                "total_products": len(products),
                "indexing_completed": datetime.utcnow().isoformat(),
                "note": "当前版本使用关键词检索，无需预建索引",
            },
        }

    async def get_visual_search_statistics(self) -> Dict[str, Any]:
        from sqlalchemy import text as sa_text
        total    = self.db.execute(sa_text("SELECT COUNT(*) FROM products")).scalar()
        w_images = self.db.execute(sa_text("SELECT COUNT(*) FROM products WHERE image_url IS NOT NULL")).scalar()
        return {
            "success": True,
            "data": {
                "total_products": total,
                "products_with_images": w_images,
                "coverage_rate": round(w_images / total * 100, 2) if total else 0,
                "supported_formats": self.supported_formats,
                "max_image_size_mb": self.max_image_size // (1024 * 1024),
            },
        }

    async def _extract_image_features_from_data(self, image_data: bytes) -> Dict[str, Any]:
        image = await self._preprocess_image(image_data)
        return await self._extract_image_features(image)


# 全局工厂函数
def get_visual_search_service(db: Session) -> VisualSearchEngine:
    return VisualSearchEngine(db)
