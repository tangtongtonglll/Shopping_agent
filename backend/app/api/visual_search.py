"""
视觉搜索和商品识别API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
# Form暂时移除，因为python-multipart安装有问题
# from fastapi import Form
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import base64
from io import BytesIO

from ..core.database import get_db
from ..services.visual_search_service import VisualSearchEngine, get_visual_search_service

router = APIRouter()

class VisualSearchOptions(BaseModel):
    """视觉搜索选项"""
    category: Optional[str] = Field(None, description="商品类别过滤")
    brand: Optional[str] = Field(None, description="品牌过滤")
    price_range: Optional[List[float]] = Field(None, description="价格范围 [min, max]")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="相似度阈值")
    max_results: int = Field(10, ge=1, le=50, description="最大返回结果数")

class ProductRecognitionRequest(BaseModel):
    """商品识别请求"""
    include_similar_products: bool = Field(True, description="是否包含相似商品")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="置信度阈值")

class VisualIndexRequest(BaseModel):
    """视觉索引创建请求"""
    product_ids: Optional[List[str]] = Field(None, description="要索引的商品ID列表")
    force_reindex: bool = Field(False, description="是否强制重新索引")

class ImageEnhancementRequest(BaseModel):
    """图像增强请求"""
    enhancement_type: str = Field(..., description="增强类型: brightness, contrast, sharpness, color")
    enhancement_value: float = Field(..., ge=-1.0, le=1.0, description="增强值")

# 暂时注释掉File上传路由，因为FastAPI需要python-multipart来处理File参数
# @router.post("/search", response_model=Dict[str, Any])
# async def search_by_image(
#     file: UploadFile = File(...),
#     options: Optional[VisualSearchOptions] = None,
#     db: Session = Depends(get_db)
# ):
#     """通过图像搜索商品"""
#     try:
#         # 验证文件类型
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="请上传图像文件")
#
#         # 读取文件内容
#         image_data = await file.read()
#
#         # 构建搜索选项
#         if options:
#             search_options = {
#                 "category": options.category,
#                 "brand": options.brand,
#                 "price_range": options.price_range,
#                 "similarity_threshold": options.similarity_threshold,
#                 "max_results": options.max_results
#             }
#         else:
#             search_options = {"max_results": 10}
#
#         # 执行搜索
#         service = get_visual_search_service(db)
#         result = await service.search_by_image(image_data, search_options)
#
#         return result
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# 暂时注释掉File上传路由，因为FastAPI需要python-multipart来处理File参数
# @router.post("/recognize", response_model=Dict[str, Any])
# async def recognize_product(
#     file: UploadFile = File(...),
#     request: Optional[ProductRecognitionRequest] = None,
#     db: Session = Depends(get_db)
# ):
#     """商品识别"""
#     try:
#         # 验证文件类型
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="请上传图像文件")
#
#         # 读取文件内容
#         image_data = await file.read()
#
#         # 构建识别选项
#         if request:
#             recognition_options = {
#                 "include_similar_products": request.include_similar_products,
#                 "confidence_threshold": request.confidence_threshold
#             }
#         else:
#             recognition_options = {
#                 "include_similar_products": True,
#                 "confidence_threshold": 0.7
#             }
#
#         # 执行识别
#         service = get_visual_search_service(db)
#         result = await service.recognize_product(image_data)
#
#         return result
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/base64", response_model=Dict[str, Any])
async def search_by_base64_image(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """通过base64图像搜索商品"""
    try:
        # 解析base64图像
        image_base64 = request.get("image_data")
        if not image_base64:
            raise HTTPException(status_code=400, detail="缺少图像数据")

        # 解码base64
        try:
            if image_base64.startswith("data:image"):
                # 移除data URL前缀
                image_base64 = image_base64.split(",")[1]
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的base64图像数据: {e}")

        # 构建搜索选项
        search_options = {
            "category": request.get("category"),
            "brand": request.get("brand"),
            "max_results": request.get("max_results", 10)
        }

        price_range = request.get("price_range")
        if price_range and len(price_range) == 2:
            search_options["price_range"] = price_range

        # 执行搜索
        service = get_visual_search_service(db)
        result = await service.search_by_image(image_data, search_options)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=Dict[str, Any])
async def get_visual_search_statistics(db: Session = Depends(get_db)):
    """获取视觉搜索统计信息"""
    try:
        service = get_visual_search_service(db)
        stats = await service.get_visual_search_statistics()

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/create", response_model=Dict[str, Any])
async def create_visual_search_index(
    request: VisualIndexRequest,
    background_tasks,
    db: Session = Depends(get_db)
):
    """创建视觉搜索索引"""
    try:
        # 获取要索引的商品
        from ..models.ecommerce_models import Product

        query = db.query(Product)
        if request.product_ids:
            query = query.filter(Product.product_id.in_(request.product_ids))

        products = query.all()

        # 转换为字典格式
        product_dicts = []
        for product in products:
            product_dict = {
                "product_id": product.product_id,
                "name": product.title,
                "brand": product.brand,
                "category": product.category,
                "price": product.price,
                "image_url": product.image_url,
                "description": product.meta_data.get("description", "") if product.meta_data else ""
            }
            product_dicts.append(product_dict)

        # 在后台任务中创建索引
        service = get_visual_search_service(db)
        background_tasks.add_task(
            service.create_visual_search_index,
            product_dicts
        )

        return {
            "success": True,
            "data": {
                "message": "视觉搜索索引创建任务已启动",
                "total_products": len(product_dicts),
                "force_reindex": request.force_reindex
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories", response_model=Dict[str, Any])
async def get_visual_search_categories(db: Session = Depends(get_db)):
    """获取支持视觉搜索的商品类别"""
    try:
        from ..models.ecommerce_models import Product

        # 获取所有类别
        categories = db.query(Product.category).distinct().all()
        category_list = [cat[0] for cat in categories if cat[0]]

        # 获取每个类别的商品数量
        category_counts = {}
        for category in category_list:
            count = db.query(Product).filter(
                Product.category == category,
                Product.image_url.isnot(None)
            ).count()
            category_counts[category] = count

        return {
            "success": True,
            "data": {
                "categories": category_list,
                "category_counts": category_counts,
                "total_categories": len(category_list)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brands", response_model=Dict[str, Any])
async def get_visual_search_brands(
    category: Optional[str] = Query(None, description="商品类别过滤"),
    db: Session = Depends(get_db)
):
    """获取支持视觉搜索的品牌"""
    try:
        from ..models.ecommerce_models import Product

        # 构建查询
        query = db.query(Product.brand).filter(
            Product.brand.isnot(None),
            Product.image_url.isnot(None)
        )

        if category:
            query = query.filter(Product.category == category)

        # 获取品牌列表
        brands = query.distinct().all()
        brand_list = [brand[0] for brand in brands if brand[0]]

        # 获取每个品牌的商品数量
        brand_counts = {}
        for brand in brand_list:
            count_query = db.query(Product).filter(
                Product.brand == brand,
                Product.image_url.isnot(None)
            )
            if category:
                count_query = count_query.filter(Product.category == category)

            brand_counts[brand] = count_query.count()

        return {
            "success": True,
            "data": {
                "brands": brand_list,
                "brand_counts": brand_counts,
                "total_brands": len(brand_list),
                "category_filter": category
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 暂时注释掉File上传路由，因为FastAPI需要python-multipart来处理File参数
# @router.post("/enhance", response_model=Dict[str, Any])
# async def enhance_image(
#     file: UploadFile = File(...),
#     request: ImageEnhancementRequest,
#     db: Session = Depends(get_db)
# ):
#     """图像增强"""
#     try:
#         # 验证文件类型
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="请上传图像文件")
#
#         # 读取文件内容
#         image_data = await file.read()
#
#         # 执行图像增强
#         service = get_visual_search_service(db)
#         enhanced_image = await service._enhance_image(image_data, request.enhancement_type, request.enhancement_value)
#
#         # 将增强后的图像转换为base64
#         enhanced_base64 = base64.b64encode(enhanced_image).decode()
#
#         return {
#             "success": True,
#             "data": {
#                 "enhanced_image": enhanced_base64,
#                 "enhancement_type": request.enhancement_type,
#                 "enhancement_value": request.enhancement_value,
#                 "original_size": len(image_data),
#                 "enhanced_size": len(enhanced_image)
#             }
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def check_visual_search_health(db: Session = Depends(get_db)):
    """检查视觉搜索服务健康状态"""
    try:
        service = get_visual_search_service(db)

        # 执行基本检查
        checks = {
            "service_available": True,
            "supported_formats": service.supported_formats,
            "max_image_size_mb": service.max_image_size // (1024 * 1024)
        }

        # 检查数据库连接
        try:
            from ..models.ecommerce_models import Product
            product_count = db.query(Product).count()
            checks["database_connection"] = True
            checks["total_products"] = product_count
        except Exception as e:
            checks["database_connection"] = False
            checks["database_error"] = str(e)

        # 检查是否有带图像的商品
        try:
            products_with_images = db.query(Product).filter(
                Product.image_url.isnot(None)
            ).count()
            checks["products_with_images"] = products_with_images
            checks["image_coverage_rate"] = round(products_with_images / checks.get("total_products", 1) * 100, 2)
        except Exception as e:
            checks["products_with_images"] = 0
            checks["image_coverage_rate"] = 0

        return {
            "success": True,
            "data": {
                "status": "healthy" if checks.get("database_connection") else "unhealthy",
                "checks": checks,
                "timestamp": service._get_current_timestamp()
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/trends/visual-search", response_model=Dict[str, Any])
async def get_visual_search_trends(
    days: int = Query(30, ge=7, le=365, description="统计天数"),
    db: Session = Depends(get_db)
):
    """获取视觉搜索趋势"""
    try:
        # 这里需要实现视觉搜索趋势统计
        # 简化版本，返回示例数据
        return {
            "success": True,
            "data": {
                "period": f"{days}天",
                "total_searches": 1250,
                "successful_recognitions": 980,
                "recognition_rate": 78.4,
                "popular_categories": [
                    {"category": "服装", "searches": 320},
                    {"category": "电子产品", "searches": 280},
                    {"category": "家居", "searches": 200}
                ],
                "average_similarity_score": 0.72,
                "trend_data": [
                    {"date": "2024-01-01", "searches": 45},
                    {"date": "2024-01-02", "searches": 52},
                    {"date": "2024-01-03", "searches": 38}
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 暂时注释掉File上传路由，因为FastAPI需要python-multipart来处理File参数
# @router.post("/batch-search", response_model=Dict[str, Any])
# async def batch_visual_search(
#     files: List[UploadFile] = File(...),
#     max_results_per_image: int = Query(5, ge=1, le=20),
#     db: Session = Depends(get_db)
# ):
#     """批量视觉搜索"""
#     try:
#         if len(files) > 10:
#             raise HTTPException(status_code=400, detail="一次最多处理10张图像")
#
#         service = get_visual_search_service(db)
#         results = []
#
#         for i, file in enumerate(files):
#             try:
#                 # 验证文件类型
#                 if not file.content_type.startswith('image/'):
#                     results.append({
#                         "file_index": i,
#                         "filename": file.filename,
#                         "success": False,
#                         "error": "不是图像文件"
#                     })
#                     continue
#
#                 # 读取文件内容
#                 image_data = await file.read()
#
#                 # 执行搜索
#                 search_result = await service.search_by_image(image_data, {"max_results": max_results_per_image})
#
#                 results.append({
#                     "file_index": i,
#                     "filename": file.filename,
#                     "success": search_result["success"],
#                     "data": search_result.get("data", {}),
#                     "error": search_result.get("error")
#                 })
#
#             except Exception as e:
#                 results.append({
#                     "file_index": i,
#                     "filename": file.filename,
#                     "success": False,
#                     "error": str(e)
#                 })
#
#         return {
#             "success": True,
#             "data": {
#                 "total_files": len(files),
#                 "successful_searches": len([r for r in results if r["success"]]),
#                 "failed_searches": len([r for r in results if not r["success"]]),
#                 "results": results
#             }
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# 添加缺失的方法到VisualSearchEngine
async def _enhance_image(self, image_data: bytes, enhancement_type: str, enhancement_value: float) -> bytes:
    """增强图像"""
    try:
        from PIL import ImageEnhance

        # 打开图像
        img = Image.open(BytesIO(image_data))

        # 根据增强类型处理
        if enhancement_type == "brightness":
            enhancer = ImageEnhance.Brightness(img)
            enhanced_img = enhancer.enhance(1.0 + enhancement_value)
        elif enhancement_type == "contrast":
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(1.0 + enhancement_value)
        elif enhancement_type == "sharpness":
            enhancer = ImageEnhance.Sharpness(img)
            enhanced_img = enhancer.enhance(1.0 + enhancement_value)
        elif enhancement_type == "color":
            enhancer = ImageEnhance.Color(img)
            enhanced_img = enhancer.enhance(1.0 + enhancement_value)
        else:
            raise ValueError(f"不支持的增强类型: {enhancement_type}")

        # 保存增强后的图像
        buffered = BytesIO()
        enhanced_img.save(buffered, format="JPEG")

        return buffered.getvalue()

    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        raise

def _get_current_timestamp(self) -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.utcnow().isoformat()

# 添加导入
import logging
logger = logging.getLogger(__name__)

# 将方法添加到类
VisualSearchEngine._enhance_image = _enhance_image
VisualSearchEngine._get_current_timestamp = _get_current_timestamp