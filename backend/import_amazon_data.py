"""
Amazon Product Data (McAuley Lab) 导入脚本
==========================================

支持格式：
  - .parquet      （McAuley Lab 2023 版推荐格式）
  - .jsonl        （明文 JSONL）
  - .jsonl.gz     （gzip 压缩 JSONL）

使用：
  cd backend
  conda activate Shopping_agent

  # 先测试 1000 条
  python import_amazon_data.py --file meta_All_Beauty.jsonl.gz.parquet --limit 1000

  # 正式导入 5 万条
  python import_amazon_data.py --file meta_All_Beauty.jsonl.gz.parquet --limit 50000
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
from typing import Iterator, Optional

import pandas as pd

# ── 路径修正 ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.core.database import SessionLocal, engine
from app.models.ecommerce_models import Base, Product
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _parse_price(raw) -> float:
    """将 '$12.99' / 12.99 / None 统一转为 float，无价格返回 0.0。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else 0.0
    s = str(raw).replace(",", "")
    m = re.search(r"\d+\.?\d*", s)
    return float(m.group()) if m else 0.0


def _best_image(images) -> Optional[str]:
    """
    从 images 字段取最高分辨率 URL。

    McAuley 2023 parquet 格式：
      images = {'hi_res': array([None, 'https://...', ...]),
                'large':  array([...]),
                'thumb':  array([...])}

    旧版 jsonl 格式：
      images = [{"hi_res": "...", "large": "..."}, ...]
    """
    if images is None:
        return None
    try:
        if pd.isna(images):
            return None
    except (TypeError, ValueError):
        pass

    # ── parquet 格式：dict of arrays ───────────────────────────────
    if isinstance(images, dict):
        for key in ("hi_res", "large", "thumb"):
            arr = images.get(key)
            if arr is None:
                continue
            # arr 可能是 np.ndarray 或 list
            for url in (arr if hasattr(arr, '__iter__') else [arr]):
                if url and isinstance(url, str) and url.startswith("http"):
                    return url
        return None

    # ── jsonl 格式：list of dicts 或 list of strings ────────────────
    if not hasattr(images, '__iter__') or isinstance(images, str):
        return None
    for img in images:
        if isinstance(img, dict):
            for key in ("hi_res", "large", "thumb"):
                url = img.get(key)
                if url and isinstance(url, str) and url.startswith("http"):
                    return url
        elif isinstance(img, str) and img.startswith("http"):
            return img
    return None


def _to_list(val) -> list:
    """将 numpy array / list / None 统一转为 Python list。"""
    if val is None:
        return []
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return [x for x in val.tolist() if x is not None]
    except ImportError:
        pass
    if isinstance(val, list):
        return [x for x in val if x is not None]
    return []


def _to_str(val, max_len=500) -> Optional[str]:
    """将 list / str / None 转为截断字符串。"""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, list):
        val = " ".join(str(x) for x in val if x)
    return str(val)[:max_len] if val else None


def _map_row(row: dict) -> Optional[dict]:
    """
    将一行 Amazon parquet/jsonl 数据映射到 Product 字段。
    返回 None 表示该行应跳过（id 或 title 缺失）。

    McAuley 2023 parquet：主键用 parent_asin（无独立 asin 列）
    McAuley 旧版 jsonl  ：主键用 asin
    """
    # 兼容两种格式的 ID 字段
    asin = str(row.get("parent_asin") or row.get("asin") or "").strip()
    if not asin:
        return None

    title = _to_str(row.get("title"), 300)
    if not title:
        return None

    price = _parse_price(row.get("price"))
    image_url = _best_image(row.get("images"))

    # 类目层级
    cats = _to_list(row.get("categories"))
    sub = _to_str(cats[1] if len(cats) > 1 else None, 100)

    meta = {
        "asin":         asin,
        "rating":       row.get("average_rating"),
        "rating_count": row.get("rating_number"),
        "features":     _to_list(row.get("features")),
        "description":  _to_list(row.get("description")),
        "details":      row.get("details") or {},
        "categories":   cats,
        "store":        row.get("store") or "",
    }

    # 用实际 DB 列名（title / description / rating / review_count）
    features_text = " | ".join(_to_list(row.get("features"))[:5])
    desc_text     = " ".join(_to_list(row.get("description"))[:3])
    full_desc     = "; ".join(filter(None, [features_text, desc_text]))

    return dict(
        product_id    = f"amazon_{asin}",
        title         = title,                              # DB 列名是 title
        description   = _to_str(full_desc, 1000),
        brand         = _to_str(row.get("store"), 100),
        category      = _to_str(row.get("main_category"), 100),
        price         = price,
        original_price= price,
        discount_rate = 0.0,
        platform      = "amazon",
        product_url   = f"https://www.amazon.com/dp/{asin}",
        image_url     = image_url,
        stock_status  = "有货",
        rating        = row.get("average_rating"),
        review_count  = row.get("rating_number"),
    )


# ── 文件读取器（返回 dict 迭代器）───────────────────────────────────────────────

def _iter_parquet(path: str, limit: int) -> Iterator[dict]:
    """分块读取 parquet，内存友好。"""
    logger.info(f"读取 parquet 文件：{path}")
    # 先探测总行数
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    logger.info(f"文件共 {total:,} 行，将读取前 {min(limit, total):,} 行")

    read = 0
    for batch in pf.iter_batches(batch_size=1000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            yield row.to_dict()
            read += 1
            if read >= limit:
                return


def _iter_jsonl(path: str, limit: int) -> Iterator[dict]:
    """读取 .jsonl 或 .jsonl.gz。"""
    opener = gzip.open(path, "rt", encoding="utf-8", errors="ignore") \
             if path.endswith(".gz") else \
             open(path, "r", encoding="utf-8", errors="ignore")
    read = 0
    with opener as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            read += 1
            if read >= limit:
                return


def _get_iterator(path: str, limit: int) -> Iterator[dict]:
    if path.endswith(".parquet"):
        return _iter_parquet(path, limit)
    return _iter_jsonl(path, limit)


# ── 主导入流程 ────────────────────────────────────────────────────────────────

def import_file(
    filepath: str,
    limit: int = 1_000,
    batch_size: int = 200,
    category_filter: Optional[str] = None,
    require_image: bool = True,
) -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    existing_ids: set = {
        row[0] for row in db.execute(text("SELECT product_id FROM products")).fetchall()
    }
    logger.info(f"数据库现有商品：{len(existing_ids)} 条")

    inserted = skipped_dup = skipped_no_img = skipped_parse = 0
    batch: list = []

    try:
        for raw in _get_iterator(filepath, limit * 3):  # 多读以补偿过滤损耗
            # 类目过滤
            if category_filter:
                cat = str(raw.get("main_category") or "")
                if category_filter.lower() not in cat.lower():
                    continue

            mapped = _map_row(raw)
            if mapped is None:
                skipped_parse += 1
                continue

            if require_image and not mapped["image_url"]:
                skipped_no_img += 1
                continue

            if mapped["product_id"] in existing_ids:
                skipped_dup += 1
                continue

            existing_ids.add(mapped["product_id"])
            batch.append(mapped)

            if len(batch) >= batch_size:
                db.execute(text("""
                    INSERT OR IGNORE INTO products
                      (product_id, title, description, brand, category,
                       price, original_price, discount_rate, platform,
                       product_url, image_url, stock_status, rating, review_count,
                       created_at, updated_at)
                    VALUES
                      (:product_id, :title, :description, :brand, :category,
                       :price, :original_price, :discount_rate, :platform,
                       :product_url, :image_url, :stock_status, :rating, :review_count,
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """), batch)
                db.commit()
                inserted += len(batch)
                batch.clear()
                logger.info(
                    f"  已导入 {inserted} 条"
                    f" | 跳过: 无图={skipped_no_img} 重复={skipped_dup}"
                )

            if inserted >= limit:
                logger.info(f"已达 limit={limit}，停止。")
                break

        if batch:
            db.bulk_save_objects(batch)
            db.commit()
            inserted += len(batch)

        if batch:
            db.execute(text("""
                INSERT OR IGNORE INTO products
                  (product_id, title, description, brand, category,
                   price, original_price, discount_rate, platform,
                   product_url, image_url, stock_status, rating, review_count,
                   created_at, updated_at)
                VALUES
                  (:product_id, :title, :description, :brand, :category,
                   :price, :original_price, :discount_rate, :platform,
                   :product_url, :image_url, :stock_status, :rating, :review_count,
                   CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """), batch)
            db.commit()
            inserted += len(batch)

    except KeyboardInterrupt:
        logger.warning("用户中断，保存已读数据…")
        if batch:
            db.execute(text("""
                INSERT OR IGNORE INTO products
                  (product_id, title, description, brand, category,
                   price, original_price, discount_rate, platform,
                   product_url, image_url, stock_status, rating, review_count,
                   created_at, updated_at)
                VALUES
                  (:product_id, :title, :description, :brand, :category,
                   :price, :original_price, :discount_rate, :platform,
                   :product_url, :image_url, :stock_status, :rating, :review_count,
                   CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """), batch)
            db.commit()
            inserted += len(batch)
    finally:
        db.close()

    logger.info("=" * 60)
    logger.info(f"✅ 导入完成：成功={inserted}  无图跳过={skipped_no_img}"
                f"  重复跳过={skipped_dup}  解析失败={skipped_parse}")


def _print_next_steps():
    print("""
═══════════════════════════════════════════════════════════
  下一步：触发视觉索引构建（后端需正在运行）
═══════════════════════════════════════════════════════════

  curl -X POST http://localhost:8000/api/visual-search/index/create \\
    -H "Content-Type: application/json" \\
    -d '{"force_reindex": false}'

  索引构建为后台任务，需下载每张图片提取颜色/纹理特征。
  1000 条约需 2~5 分钟（取决于网速）。
═══════════════════════════════════════════════════════════
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导入 McAuley Lab Amazon 商品数据（支持 parquet / jsonl / jsonl.gz）"
    )
    parser.add_argument("--file",     required=True, help="数据文件路径")
    parser.add_argument("--limit",    type=int, default=1_000, help="最多导入条数（默认 1000）")
    parser.add_argument("--batch",    type=int, default=200,   help="每批 commit 大小（默认 200）")
    parser.add_argument("--category", type=str, default=None,  help="仅导入该 main_category（模糊匹配）")
    parser.add_argument("--allow-no-image", action="store_true", help="允许无图商品（默认跳过）")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"文件不存在：{args.file}")
        sys.exit(1)

    logger.info(f"文件：{args.file}")
    logger.info(f"参数：limit={args.limit}  batch={args.batch}"
                f"  category={args.category or '全部'}"
                f"  require_image={not args.allow_no_image}")

    import_file(
        filepath        = args.file,
        limit           = args.limit,
        batch_size      = args.batch,
        category_filter = args.category,
        require_image   = not args.allow_no_image,
    )
    _print_next_steps()
