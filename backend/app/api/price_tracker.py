"""
价格跟踪API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.database import get_db
from ..services.price_tracker_service import PriceTrackerService
from ..services.llm_service import LLMService

router = APIRouter()

class PriceAlertCreate(BaseModel):
    """创建价格提醒请求"""
    product_id: str = Field(..., description="商品ID")
    target_price: float = Field(..., gt=0, description="目标价格")
    alert_type: str = Field("below_target", description="提醒类型: below_target, percentage_drop")
    threshold_percentage: Optional[float] = Field(None, ge=0, le=100, description="降价百分比阈值")
    notification_method: str = Field("app", description="通知方式: app, email, sms")

class PriceAlertResponse(BaseModel):
    """价格提醒响应"""
    alert_id: int
    product_id: str
    product_name: str
    target_price: float
    current_price: float
    alert_type: str
    threshold_percentage: Optional[float]
    is_active: bool
    is_triggered: bool
    created_at: str
    triggered_at: Optional[str]

class PriceTrackingRequest(BaseModel):
    """价格跟踪请求"""
    product_ids: Optional[List[str]] = Field(None, description="要跟踪的商品ID列表")
    force_update: bool = Field(False, description="是否强制更新")

class PriceAnalysisRequest(BaseModel):
    """价格分析请求"""
    product_id: str = Field(..., description="商品ID")
    days: int = Field(30, ge=1, le=365, description="分析天数")

@router.post("/track", response_model=Dict[str, Any])
async def track_prices(
    request: PriceTrackingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """触发价格跟踪"""
    try:
        llm_service = LLMService()

        async def track():
            async with PriceTrackerService(db, llm_service) as tracker:
                return await tracker.track_product_prices(request.product_ids)

        # 在后台执行价格跟踪
        background_tasks.add_task(asyncio.run, track())

        return {
            "success": True,
            "message": "价格跟踪已启动",
            "product_ids": request.product_ids or "所有活跃提醒商品",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts", response_model=Dict[str, Any])
async def create_price_alert(
    request: PriceAlertCreate,
    user_id: str = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """创建价格提醒"""
    try:
        llm_service = LLMService()
        async with PriceTrackerService(db, llm_service) as tracker:
            result = await tracker.create_price_alert(
                user_id=user_id,
                product_id=request.product_id,
                target_price=request.target_price,
                alert_type=request.alert_type,
                threshold_percentage=request.threshold_percentage,
                notification_method=request.notification_method
            )

            return {
                "success": True,
                "data": result,
                "message": f"价格提醒已{result['action']}"
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[PriceAlertResponse])
async def get_user_alerts(
    user_id: str = Query(..., description="用户ID"),
    include_triggered: bool = Query(False, description="包含已触发的提醒"),
    db: Session = Depends(get_db)
):
    """获取用户的价格提醒"""
    try:
        llm_service = LLMService()
        async with PriceTrackerService(db, llm_service) as tracker:
            alerts = await tracker.get_user_alerts(user_id, include_triggered)
            return [PriceAlertResponse(**alert) for alert in alerts]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/alerts/{alert_id}", response_model=Dict[str, Any])
async def delete_price_alert(
    alert_id: int,
    user_id: str = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """删除价格提醒"""
    try:
        llm_service = LLMService()
        async with PriceTrackerService(db, llm_service) as tracker:
            success = await tracker.delete_alert(alert_id, user_id)

            if not success:
                raise HTTPException(status_code=404, detail="价格提醒不存在")

            return {
                "success": True,
                "message": "价格提醒已删除"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{product_id}", response_model=Dict[str, Any])
async def get_price_analysis(
    product_id: str,
    days: int = Query(30, ge=1, le=365, description="分析天数"),
    db: Session = Depends(get_db)
):
    """获取价格分析报告"""
    try:
        llm_service = LLMService()
        async with PriceTrackerService(db, llm_service) as tracker:
            analysis = await tracker.get_price_analysis(product_id, days)

            if "error" in analysis:
                raise HTTPException(status_code=404, detail=analysis["error"])

            return {
                "success": True,
                "data": analysis
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/{alert_id}/toggle", response_model=Dict[str, Any])
async def toggle_alert_status(
    alert_id: int,
    user_id: str = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """切换价格提醒状态"""
    try:
        from ..models.ecommerce_models import PriceAlert

        alert = db.query(PriceAlert).filter(
            PriceAlert.id == alert_id,
            PriceAlert.user_id == user_id
        ).first()

        if not alert:
            raise HTTPException(status_code=404, detail="价格提醒不存在")

        alert.is_active = not alert.is_active
        alert.updated_at = datetime.now()
        db.commit()

        return {
            "success": True,
            "message": f"价格提醒已{'启用' if alert.is_active else '禁用'}",
            "is_active": alert.is_active
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/{user_id}", response_model=Dict[str, Any])
async def get_price_dashboard(
    user_id: str,
    db: Session = Depends(get_db)
):
    """获取用户价格跟踪仪表板"""
    try:
        from ..models.ecommerce_models import PriceAlert, PriceHistory

        # 统计数据
        total_alerts = db.query(PriceAlert).filter(
            PriceAlert.user_id == user_id
        ).count()

        active_alerts = db.query(PriceAlert).filter(
            PriceAlert.user_id == user_id,
            PriceAlert.is_active == True,
            PriceAlert.is_triggered == False
        ).count()

        triggered_alerts = db.query(PriceAlert).filter(
            PriceAlert.user_id == user_id,
            PriceAlert.is_triggered == True
        ).count()

        # 最近的价格变化
        recent_changes = db.query(PriceHistory).join(
            PriceAlert, PriceHistory.product_id == PriceAlert.product_id
        ).filter(
            PriceAlert.user_id == user_id,
            PriceHistory.date >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ).order_by(PriceHistory.date.desc()).limit(10).all()

        # 预计节省金额
        estimated_savings = 0
        if triggered_alerts > 0:
            # 简单估算：假设每个触发的提醒平均节省10%
            avg_target_price = db.query(PriceAlert).filter(
                PriceAlert.user_id == user_id,
                PriceAlert.is_triggered == True
            ).with_entities(func.avg(PriceAlert.target_price)).scalar() or 0
            estimated_savings = triggered_alerts * avg_target_price * 0.1

        return {
            "success": True,
            "data": {
                "statistics": {
                    "total_alerts": total_alerts,
                    "active_alerts": active_alerts,
                    "triggered_alerts": triggered_alerts,
                    "estimated_savings": round(estimated_savings, 2)
                },
                "recent_changes": [
                    {
                        "product_id": change.product_id,
                        "price": change.price,
                        "date": change.date.isoformat(),
                        "platform": change.platform
                    }
                    for change in recent_changes
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/batch", response_model=Dict[str, Any])
async def batch_create_alerts(
    requests: List[PriceAlertCreate],
    user_id: str = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """批量创建价格提醒"""
    try:
        llm_service = LLMService()
        async with PriceTrackerService(db, llm_service) as tracker:
            results = []
            for request in requests:
                try:
                    result = await tracker.create_price_alert(
                        user_id=user_id,
                        product_id=request.product_id,
                        target_price=request.target_price,
                        alert_type=request.alert_type,
                        threshold_percentage=request.threshold_percentage,
                        notification_method=request.notification_method
                    )
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            return {
                "success": True,
                "data": results,
                "total": len(requests),
                "successful": len([r for r in results if "error" not in r])
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending", response_model=Dict[str, Any])
async def get_trending_price_changes(
    limit: int = Query(10, le=50, description="返回数量"),
    hours: int = Query(24, ge=1, le=168, description="时间范围（小时）"),
    db: Session = Depends(get_db)
):
    """获取热门价格变化"""
    try:
        from ..models.ecommerce_models import PriceHistory, Product

        since_date = datetime.now() - timedelta(hours=hours)

        # 获取最近有价格变化的商品
        subquery = db.query(
            PriceHistory.product_id,
            func.max(PriceHistory.date).label('max_date')
        ).filter(
            PriceHistory.date >= since_date
        ).group_by(PriceHistory.product_id).subquery()

        # 获取最新的价格记录
        latest_prices = db.query(PriceHistory).join(
            subquery,
            and_(
                PriceHistory.product_id == subquery.c.product_id,
                PriceHistory.date == subquery.c.max_date
            )
        ).all()

        # 计算价格变化
        trending_items = []
        for price_record in latest_prices:
            # 获取前一天的价格
            prev_date = price_record.date - timedelta(hours=24)
            prev_price = db.query(PriceHistory).filter(
                PriceHistory.product_id == price_record.product_id,
                PriceHistory.date <= prev_date
            ).order_by(PriceHistory.date.desc()).first()

            if prev_price and prev_price.price != price_record.price:
                change_percent = (price_record.price - prev_price.price) / prev_price.price * 100

                product = db.query(Product).filter(
                    Product.product_id == price_record.product_id
                ).first()

                if product:
                    trending_items.append({
                        "product_id": price_record.product_id,
                        "product_name": product.title,
                        "brand": product.brand,
                        "category": product.category,
                        "current_price": price_record.price,
                        "previous_price": prev_price.price,
                        "change_percent": round(change_percent, 2),
                        "change_direction": "up" if change_percent > 0 else "down",
                        "platform": price_record.platform,
                        "last_updated": price_record.date.isoformat()
                    })

        # 按变化幅度排序
        trending_items.sort(key=lambda x: abs(x['change_percent']), reverse=True)

        return {
            "success": True,
            "data": {
                "trending_items": trending_items[:limit],
                "time_range": f"{hours}小时",
                "total_changes": len(trending_items)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 需要在文件开头添加
import asyncio
from datetime import timedelta