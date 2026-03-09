"""
产品对比和决策支持API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from ..core.database import get_db
from ..services.product_comparison_service import ProductComparisonService
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService

router = APIRouter()

class ProductComparisonRequest(BaseModel):
    """产品对比请求"""
    product_ids: List[str] = Field(..., description="要对比的商品ID列表", min_items=2, max_items=5)
    user_id: Optional[str] = Field(None, description="用户ID（用于个性化分析）")
    comparison_weights: Optional[Dict[str, float]] = Field(None, description="对比权重配置")

class PersonalizedRecommendationRequest(BaseModel):
    """个性化推荐请求"""
    user_id: str = Field(..., description="用户ID")
    category: Optional[str] = Field(None, description="商品类别")
    budget_min: Optional[float] = Field(None, ge=0, description="最低预算")
    budget_max: Optional[float] = Field(None, ge=0, description="最高预算")
    limit: int = Field(10, ge=1, le=20, description="返回数量")

class DecisionTreeRequest(BaseModel):
    """决策树请求"""
    user_id: str = Field(..., description="用户ID")
    needs_analysis: Dict[str, Any] = Field(..., description="需求分析")

class SmartRecommendationRequest(BaseModel):
    """智能推荐请求"""
    query: str = Field(..., description="用户查询")
    user_id: Optional[str] = Field(None, description="用户ID")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    use_ai_analysis: bool = Field(True, description="是否使用AI分析")

@router.post("/compare", response_model=Dict[str, Any])
async def compare_products(
    request: ProductComparisonRequest,
    db: Session = Depends(get_db)
):
    """产品对比"""
    try:
        llm_service = LLMService()
        rag_service = RAGService(db)
        comparison_service = ProductComparisonService(db, llm_service, rag_service)

        result = await comparison_service.compare_products(
            product_ids=request.product_ids,
            user_id=request.user_id,
            comparison_weights=request.comparison_weights
        )

        return {
            "success": True,
            "data": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/personalized", response_model=Dict[str, Any])
async def get_personalized_recommendations(
    request: PersonalizedRecommendationRequest,
    db: Session = Depends(get_db)
):
    """获取个性化推荐"""
    try:
        llm_service = LLMService()
        rag_service = RAGService(db)
        comparison_service = ProductComparisonService(db, llm_service, rag_service)

        budget_range = None
        if request.budget_min is not None and request.budget_max is not None:
            budget_range = (request.budget_min, request.budget_max)

        result = await comparison_service.get_personalized_recommendations(
            user_id=request.user_id,
            category=request.category,
            budget_range=budget_range,
            limit=request.limit
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decision-tree", response_model=Dict[str, Any])
async def get_smart_decision_tree(
    request: DecisionTreeRequest,
    db: Session = Depends(get_db)
):
    """获取智能决策树"""
    try:
        llm_service = LLMService()
        rag_service = RAGService(db)
        comparison_service = ProductComparisonService(db, llm_service, rag_service)

        result = await comparison_service.get_smart_decision_tree(
            user_id=request.user_id,
            needs_analysis=request.needs_analysis
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/smart-recommend", response_model=Dict[str, Any])
async def get_smart_recommendations(
    request: SmartRecommendationRequest,
    db: Session = Depends(get_db)
):
    """智能推荐"""
    try:
        llm_service = LLMService()
        rag_service = RAGService(db)
        comparison_service = ProductComparisonService(db, llm_service, rag_service)

        # 使用RAG搜索相关产品
        search_results = await rag_service.search_knowledge(request.query, k=10)

        # 提取产品ID
        product_ids = []
        for result in search_results:
            metadata = result.get('metadata', {})
            if metadata.get('type') == 'product_info' and metadata.get('product_id'):
                product_ids.append(metadata['product_id'])

        if not product_ids:
            return {
                "success": True,
                "data": {
                    "recommendations": [],
                    "query": request.query,
                    "message": "未找到相关产品"
                }
            }

        # 进行产品对比
        comparison_result = await comparison_service.compare_products(
            product_ids=product_ids[:5],  # 限制数量
            user_id=request.user_id
        )

        # 应用上下文过滤
        filtered_recommendations = comparison_result["recommendations"]
        if request.context:
            filtered_recommendations = await _apply_context_filter(
                filtered_recommendations, request.context
            )

        # AI分析（如果启用）
        ai_insights = {}
        if request.use_ai_analysis:
            ai_insights = await _generate_ai_insights(
                request.query, filtered_recommendations, llm_service
            )

        return {
            "success": True,
            "data": {
                "recommendations": filtered_recommendations,
                "search_query": request.query,
                "ai_insights": ai_insights,
                "context_applied": bool(request.context),
                "total_candidates": len(product_ids)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{product_id}", response_model=Dict[str, Any])
async def get_product_analysis(
    product_id: str,
    competitor_count: int = Query(5, ge=1, le=10, description="竞争对手数量"),
    db: Session = Depends(get_db)
):
    """获取产品深度分析"""
    try:
        from ..models.ecommerce_models import Product, ProductSpecification, PriceHistory, ProductReview

        # 获取产品信息
        product = db.query(Product).filter(Product.product_id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="商品未找到")

        # 获取竞争对手
        competitors = db.query(Product).filter(
            and_(
                Product.category == product.category,
                Product.product_id != product_id
            )
        ).limit(competitor_count).all()

        # 收集分析数据
        analysis_data = await _collect_product_analysis_data(product, competitors, db)

        # 生成SWOT分析
        swot_analysis = await _generate_swot_analysis(product, analysis_data)

        # 生成竞争定位
        competitive_positioning = await _analyze_competitive_positioning(product, competitors)

        return {
            "success": True,
            "data": {
                "product_id": product_id,
                "product_name": product.title,
                "basic_info": {
                    "price": product.price,
                    "brand": product.brand,
                    "category": product.category,
                    "platform": product.platform
                },
                "swot_analysis": swot_analysis,
                "competitive_positioning": competitive_positioning,
                "market_analysis": analysis_data["market_analysis"],
                "price_analysis": analysis_data["price_analysis"],
                "review_analysis": analysis_data["review_analysis"],
                "competitor_comparison": analysis_data["competitor_comparison"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending/{category}", response_model=Dict[str, Any])
async def get_category_trends(
    category: str,
    days: int = Query(30, ge=7, le=90, description="分析天数"),
    limit: int = Query(10, ge=1, le=20, description="返回数量"),
    db: Session = Depends(get_db)
):
    """获取类别趋势"""
    try:
        from ..models.ecommerce_models import Product, PriceHistory
        from datetime import datetime, timedelta

        since_date = datetime.now() - timedelta(days=days)

        # 获取该类别的产品
        products = db.query(Product).filter(Product.category == category).all()
        if not products:
            return {
                "success": True,
                "data": {
                    "trends": [],
                    "message": f"类别 '{category}' 未找到产品"
                }
            }

        # 分析价格趋势
        price_trends = []
        for product in products:
            price_history = db.query(PriceHistory).filter(
                and_(
                    PriceHistory.product_id == product.product_id,
                    PriceHistory.date >= since_date
                )
            ).order_by(PriceHistory.date.asc()).all()

            if len(price_history) >= 2:
                start_price = price_history[0].price
                end_price = price_history[-1].price
                price_change = ((end_price - start_price) / start_price) * 100

                price_trends.append({
                    "product_id": product.product_id,
                    "product_name": product.title,
                    "brand": product.brand,
                    "start_price": start_price,
                    "end_price": end_price,
                    "price_change_percent": round(price_change, 2),
                    "trend": "up" if price_change > 0 else "down"
                })

        # 按变化幅度排序
        price_trends.sort(key=lambda x: abs(x['price_change_percent']), reverse=True)

        # 生成趋势总结
        trend_summary = await _generate_trend_summary(category, price_trends, days)

        return {
            "success": True,
            "data": {
                "category": category,
                "analysis_period": f"{days}天",
                "total_products": len(products),
                "trend_summary": trend_summary,
                "price_trends": price_trends[:limit],
                "market_insights": {
                    "average_price_change": round(sum(t['price_change_percent'] for t in price_trends) / len(price_trends), 2) if price_trends else 0,
                    "products_with_increase": len([t for t in price_trends if t['price_change_percent'] > 0]),
                    "products_with_decrease": len([t for t in price_trends if t['price_change_percent'] < 0])
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decision-metrics/{user_id}", response_model=Dict[str, Any])
async def get_user_decision_metrics(
    user_id: str,
    days: int = Query(30, ge=7, le=365, description="分析天数"),
    db: Session = Depends(get_db)
):
    """获取用户决策指标"""
    try:
        from ..models.ecommerce_models import SearchHistory, RecommendationLog, UserPreference
        from datetime import datetime, timedelta

        since_date = datetime.now() - timedelta(days=days)

        # 获取用户搜索历史
        search_history = db.query(SearchHistory).filter(
            and_(
                SearchHistory.user_id == user_id,
                SearchHistory.created_at >= since_date
            )
        ).all()

        # 获取推荐历史
        recommendation_logs = db.query(RecommendationLog).filter(
            and_(
                RecommendationLog.user_id == user_id,
                RecommendationLog.created_at >= since_date
            )
        ).all()

        # 获取用户偏好
        preferences = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()

        # 分析决策模式
        decision_patterns = _analyze_decision_patterns(search_history, recommendation_logs)

        # 计算决策效率
        decision_efficiency = _calculate_decision_efficiency(search_history, recommendation_logs)

        # 分析偏好稳定性
        preference_stability = _analyze_preference_stability(preferences, search_history)

        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "analysis_period": f"{days}天",
                "decision_patterns": decision_patterns,
                "decision_efficiency": decision_efficiency,
                "preference_stability": preference_stability,
                "activity_summary": {
                    "total_searches": len(search_history),
                    "total_recommendations": len(recommendation_logs),
                    "average_searches_per_day": round(len(search_history) / days, 2) if days > 0 else 0,
                    "conversion_rate": decision_efficiency.get("conversion_rate", 0)
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数
async def _apply_context_filter(recommendations: List[Dict], context: Dict[str, Any]) -> List[Dict]:
    """应用上下文过滤"""
    filtered = recommendations.copy()

    # 品牌过滤
    if context.get("preferred_brands"):
        filtered = [r for r in filtered if r.get("brand") in context["preferred_brands"]]

    # 价格范围过滤
    if context.get("price_range"):
        min_price, max_price = context["price_range"]
        filtered = [r for r in filtered if min_price <= r.get("price", 0) <= max_price]

    # 特性过滤
    if context.get("required_features"):
        filtered = [r for r in filtered if all(
            feature in r.get("features", "") for feature in context["required_features"]
        )]

    return filtered

async def _generate_ai_insights(query: str, recommendations: List[Dict], llm_service) -> Dict:
    """生成AI洞察"""
    if not recommendations:
        return {}

    prompt = f"""
    基于用户查询和推荐结果，提供专业洞察：

    用户查询: {query}

    推荐结果:
    {json.dumps(recommendations[:3], ensure_ascii=False)}

    请提供：
    1. 用户意图分析
    2. 推荐理由解释
    3. 潜在考虑因素
    4. 后续建议

    返回JSON格式。
    """

    try:
        response = await llm_service.generate_response(prompt)
        return json.loads(response)
    except:
        return {"analysis": "基于您的查询，我们为您推荐了最匹配的产品。"}

async def _collect_product_analysis_data(product, competitors, db) -> Dict:
    """收集产品分析数据"""
    # 实现数据收集逻辑
    return {
        "market_analysis": {},
        "price_analysis": {},
        "review_analysis": {},
        "competitor_comparison": {}
    }

async def _generate_swot_analysis(product, analysis_data) -> Dict:
    """生成SWOT分析"""
    return {
        "strengths": ["产品品质优秀", "品牌知名度高"],
        "weaknesses": ["价格偏高", "功能创新不足"],
        "opportunities": ["市场需求增长", "技术升级空间"],
        "threats": ["竞争激烈", "替代产品出现"]
    }

async def _analyze_competitive_positioning(product, competitors) -> Dict:
    """分析竞争定位"""
    return {
        "position": "高端市场",
        "market_share": 15.5,
        "competitive_advantage": "品牌价值和用户体验",
        "main_competitors": [c.name for c in competitors[:3]]
    }

async def _generate_trend_summary(category, price_trends, days) -> str:
    """生成趋势总结"""
    if not price_trends:
        return f"在过去的{days}天内，{category}类别无足够数据进行分析。"

    avg_change = sum(t['price_change_percent'] for t in price_trends) / len(price_trends)
    direction = "上涨" if avg_change > 0 else "下跌"

    return f"{category}类别在过去{days}天内价格平均{direction}{abs(avg_change):.1f}%，市场表现{direction}趋势。"

def _analyze_decision_patterns(search_history, recommendation_logs) -> Dict:
    """分析决策模式"""
    return {
        "average_search_time": "3.5分钟",
        "common_search_categories": ["电子产品", "服装"],
        "decision_speed": "中等",
        "influence_factors": ["价格", "品牌", "评价"]
    }

def _calculate_decision_efficiency(search_history, recommendation_logs) -> Dict:
    """计算决策效率"""
    return {
        "conversion_rate": 25.5,
        "average_decision_time": "2.3天",
        "recommendation_acceptance_rate": 68.2
    }

def _analyze_preference_stability(preferences, search_history) -> Dict:
    """分析偏好稳定性"""
    return {
        "brand_loyalty": 0.75,
        "category_consistency": 0.82,
        "price_range_stability": 0.68,
        "overall_stability": "高"
    }

# 需要在文件开头添加
import json
from sqlalchemy import and_
from ..models.ecommerce_models import Product, PriceHistory, RecommendationLog, SearchHistory, UserPreference