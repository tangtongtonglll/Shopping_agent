"""
个性化购物行为分析服务
跟踪和分析用户购物行为，提供个性化推荐和体验优化
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy import func, desc, and_, or_
from ..models.models import User, Memory
from ..models.ecommerce_models import (
    Product, ProductReview, SearchHistory, RecommendationLog,
    UserPreference, PriceAlert, Product, PriceHistory
)
from ..services.llm_service import LLMService
from ..services.vector_service import vector_service
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class UserBehaviorTracker:
    """用户行为追踪器"""

    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()

    def track_user_action(self, user_id: str, action_type: str, action_data: Dict[str, Any]):
        """追踪用户行为"""
        try:
            # 创建行为记录
            behavior_record = {
                "user_id": user_id,
                "action_type": action_type,
                "action_data": action_data,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": action_data.get("session_id", "default"),
                "device_info": action_data.get("device_info", {}),
                "context": action_data.get("context", {})
            }

            # 存储到内存系统（短期）
            self._store_short_term_memory(behavior_record)

            # 更新用户画像
            self._update_user_profile(user_id, action_type, action_data)

            # 触发实时分析
            if self._should_trigger_analysis(user_id):
                self._analyze_behavior_patterns(user_id)

            return {"success": True, "message": "行为记录已保存"}

        except Exception as e:
            logger.error(f"Error tracking user action: {e}")
            return {"success": False, "error": str(e)}

    def _store_short_term_memory(self, record: Dict[str, Any]):
        """存储短期记忆"""
        try:
            memory = Memory(
                user_id=int(record["user_id"]) if record["user_id"].isdigit() else 0,
                content=json.dumps(record, ensure_ascii=False),
                memory_type="user_behavior",
                importance_score=0.5,
                metadata={
                    "action_type": record["action_type"],
                    "timestamp": record["timestamp"],
                    "session_id": record["session_id"]
                }
            )

            self.db.add(memory)
            self.db.commit()

        except Exception as e:
            logger.error(f"Error storing short term memory: {e}")

    def _update_user_profile(self, user_id: str, action_type: str, action_data: Dict[str, Any]):
        """更新用户画像"""
        try:
            # 获取或创建用户偏好
            preference = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).first()

            if not preference:
                preference = UserPreference(
                    user_id=user_id,
                    preferred_brands=[],
                    preferred_categories=[],
                    price_range_min=0,
                    price_range_max=10000,
                    preferred_features=[],
                    purchase_history=[],
                    browsing_history=[],
                    meta_data={}
                )
                self.db.add(preference)

            # 根据行为类型更新偏好
            if action_type == "search":
                self._update_search_preferences(preference, action_data)
            elif action_type == "view_product":
                self._update_view_preferences(preference, action_data)
            elif action_type == "purchase":
                self._update_purchase_preferences(preference, action_data)
            elif action_type == "add_to_cart":
                self._update_cart_preferences(preference, action_data)
            elif action_type == "review":
                self._update_review_preferences(preference, action_data)

            # 更新浏览历史
            browsing_history = preference.browsing_history or []
            browsing_history.append({
                "action": action_type,
                "data": action_data,
                "timestamp": datetime.utcnow().isoformat()
            })

            # 限制历史记录长度
            if len(browsing_history) > 1000:
                browsing_history = browsing_history[-500:]

            preference.browsing_history = browsing_history
            preference.updated_at = datetime.utcnow()

            self.db.commit()

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            self.db.rollback()

    def _update_search_preferences(self, preference: UserPreference, data: Dict[str, Any]):
        """更新搜索偏好"""
        query = data.get("query", "")
        filters = data.get("filters", {})

        # 分析搜索词中的类别和品牌
        if query:
            search_terms = query.lower().split()

            # 更新类别偏好
            categories = preference.preferred_categories or []
            for term in search_terms:
                if term in ["手机", "电脑", "服装", "家电", "美妆", "食品"]:
                    if term not in categories:
                        categories.append(term)

            # 更新品牌偏好
            brands = preference.preferred_brands or []
            common_brands = ["苹果", "华为", "小米", "三星", "耐克", "阿迪达斯"]
            for brand in common_brands:
                if brand in query and brand not in brands:
                    brands.append(brand)

            preference.preferred_categories = categories
            preference.preferred_brands = brands

        # 更新价格范围偏好
        min_price = filters.get("min_price")
        max_price = filters.get("max_price")
        if min_price and max_price:
            preference.price_range_min = min(preference.price_range_min, min_price)
            preference.price_range_max = max(preference.price_range_max, max_price)

    def _update_view_preferences(self, preference: UserPreference, data: Dict[str, Any]):
        """更新浏览偏好"""
        product_id = data.get("product_id")
        if product_id:
            # 获取产品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()

            if product:
                # 更新品牌偏好
                brands = preference.preferred_brands or []
                if product.brand and product.brand not in brands:
                    brands.append(product.brand)

                # 更新类别偏好
                categories = preference.preferred_categories or []
                if product.category and product.category not in categories:
                    categories.append(product.category)

                preference.preferred_brands = brands
                preference.preferred_categories = categories

    def _update_purchase_preferences(self, preference: UserPreference, data: Dict[str, Any]):
        """更新购买偏好"""
        product_id = data.get("product_id")
        price = data.get("price")

        if product_id:
            # 获取产品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()

            if product:
                # 更新购买历史
                purchase_history = preference.purchase_history or []
                purchase_history.append({
                    "product_id": product_id,
                    "product_name": product.title,
                    "brand": product.brand,
                    "category": product.category,
                    "price": price or product.price,
                    "purchase_date": datetime.utcnow().isoformat()
                })

                preference.purchase_history = purchase_history

                # 增强品牌和类别偏好（购买行为权重更高）
                if product.brand:
                    brands = preference.preferred_brands or []
                    if product.brand in brands:
                        # 增加权重（移到列表前面）
                        brands.remove(product.brand)
                    brands.insert(0, product.brand)
                    preference.preferred_brands = brands

                if product.category:
                    categories = preference.preferred_categories or []
                    if product.category in categories:
                        categories.remove(product.category)
                    categories.insert(0, product.category)
                    preference.preferred_categories = categories

    def _update_cart_preferences(self, preference: UserPreference, data: Dict[str, Any]):
        """更新购物车偏好"""
        product_id = data.get("product_id")
        if product_id:
            # 获取产品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()

            if product:
                # 更新品牌和类别偏好（中等权重）
                brands = preference.preferred_brands or []
                if product.brand and product.brand not in brands:
                    brands.append(product.brand)

                categories = preference.preferred_categories or []
                if product.category and product.category not in categories:
                    categories.append(product.category)

                preference.preferred_brands = brands
                preference.preferred_categories = categories

    def _update_review_preferences(self, preference: UserPreference, data: Dict[str, Any]):
        """更新评价偏好"""
        product_id = data.get("product_id")
        rating = data.get("rating")
        review_text = data.get("review_text", "")

        if product_id and rating:
            # 分析评价倾向
            sentiment = self._analyze_review_sentiment(review_text) if review_text else "neutral"

            # 更新用户特征
            features = preference.preferred_features or []
            if sentiment == "positive":
                if "质量优先" not in features:
                    features.append("质量优先")
            elif sentiment == "negative":
                if "价格敏感" not in features:
                    features.append("价格敏感")

            preference.preferred_features = features

    def _analyze_review_sentiment(self, text: str) -> str:
        """简单情感分析"""
        positive_words = ["好", "棒", "优秀", "满意", "喜欢", "推荐", "值得"]
        negative_words = ["差", "糟糕", "失望", "不满", "退货", "问题", "垃圾"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _should_trigger_analysis(self, user_id: str) -> bool:
        """判断是否触发行为分析"""
        # 简单的触发条件：用户行为达到一定数量
        recent_actions = self.db.query(Memory).filter(
            and_(
                Memory.user_id == int(user_id) if user_id.isdigit() else 0,
                Memory.memory_type == "user_behavior",
                Memory.created_at >= datetime.utcnow() - timedelta(hours=1)
            )
        ).count()

        return recent_actions >= 10  # 1小时内10个行为触发分析

    def _analyze_behavior_patterns(self, user_id: str):
        """分析用户行为模式"""
        try:
            # 获取最近的行为数据
            recent_behaviors = self.db.query(Memory).filter(
                and_(
                    Memory.user_id == int(user_id) if user_id.isdigit() else 0,
                    Memory.memory_type == "user_behavior",
                    Memory.created_at >= datetime.utcnow() - timedelta(days=7)
                )
            ).all()

            if not recent_behaviors:
                return

            # 分析行为模式
            behavior_analysis = self._perform_behavior_analysis(recent_behaviors)

            # 存储分析结果
            analysis_memory = Memory(
                user_id=int(user_id) if user_id.isdigit() else 0,
                content=json.dumps(behavior_analysis, ensure_ascii=False),
                memory_type="behavior_analysis",
                importance_score=0.8,
                metadata={
                    "analysis_date": datetime.utcnow().isoformat(),
                    "behaviors_analyzed": len(recent_behaviors)
                }
            )

            self.db.add(analysis_memory)
            self.db.commit()

        except Exception as e:
            logger.error(f"Error analyzing behavior patterns: {e}")

    def _perform_behavior_analysis(self, behaviors: List[Memory]) -> Dict[str, Any]:
        """执行行为分析"""
        try:
            # 解析行为数据
            action_data = []
            for behavior in behaviors:
                try:
                    content = json.loads(behavior.content)
                    action_data.append(content)
                except:
                    continue

            if not action_data:
                return {}

            # 统计行为类型
            action_types = Counter([item["action_type"] for item in action_data])

            # 分析时间模式
            time_patterns = self._analyze_time_patterns(action_data)

            # 分析偏好模式
            preference_patterns = self._analyze_preference_patterns(action_data)

            # 分析购物漏斗
            funnel_analysis = self._analyze_shopping_funnel(action_data)

            return {
                "action_summary": dict(action_types),
                "time_patterns": time_patterns,
                "preference_patterns": preference_patterns,
                "funnel_analysis": funnel_analysis,
                "analysis_period": "7天",
                "total_actions": len(action_data)
            }

        except Exception as e:
            logger.error(f"Error in behavior analysis: {e}")
            return {}

    def _analyze_time_patterns(self, action_data: List[Dict]) -> Dict[str, Any]:
        """分析时间模式"""
        try:
            hourly_actions = defaultdict(int)
            daily_actions = defaultdict(int)

            for action in action_data:
                timestamp = datetime.fromisoformat(action["timestamp"].replace('Z', '+00:00'))

                # 小时分布
                hour = timestamp.hour
                hourly_actions[hour] += 1

                # 星期分布
                weekday = timestamp.weekday()
                daily_actions[weekday] += 1

            return {
                "peak_hours": dict(hourly_actions),
                "peak_days": dict(daily_actions),
                "most_active_hour": max(hourly_actions, key=hourly_actions.get),
                "most_active_day": max(daily_actions, key=daily_actions.get)
            }

        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}

    def _analyze_preference_patterns(self, action_data: List[Dict]) -> Dict[str, Any]:
        """分析偏好模式"""
        try:
            categories = defaultdict(int)
            brands = defaultdict(int)
            price_ranges = defaultdict(int)

            for action in action_data:
                data = action["action_data"]

                # 提取类别信息
                if "category" in data:
                    categories[data["category"]] += 1

                # 提取品牌信息
                if "brand" in data:
                    brands[data["brand"]] += 1

                # 提取价格范围
                if "price" in data:
                    price = data["price"]
                    if price < 100:
                        price_ranges["low"] += 1
                    elif price < 500:
                        price_ranges["medium"] += 1
                    else:
                        price_ranges["high"] += 1

            return {
                "preferred_categories": dict(categories),
                "preferred_brands": dict(brands),
                "price_sensitivity": dict(price_ranges)
            }

        except Exception as e:
            logger.error(f"Error analyzing preference patterns: {e}")
            return {}

    def _analyze_shopping_funnel(self, action_data: List[Dict]) -> Dict[str, Any]:
        """分析购物漏斗"""
        try:
            funnel = {
                "search": 0,
                "view_product": 0,
                "add_to_cart": 0,
                "purchase": 0
            }

            for action in action_data:
                action_type = action["action_type"]
                if action_type in funnel:
                    funnel[action_type] += 1

            # 计算转化率
            conversion_rates = {}
            if funnel["search"] > 0:
                conversion_rates["search_to_view"] = funnel["view_product"] / funnel["search"]
            if funnel["view_product"] > 0:
                conversion_rates["view_to_cart"] = funnel["add_to_cart"] / funnel["view_product"]
            if funnel["add_to_cart"] > 0:
                conversion_rates["cart_to_purchase"] = funnel["purchase"] / funnel["add_to_cart"]

            return {
                "funnel_counts": funnel,
                "conversion_rates": conversion_rates
            }

        except Exception as e:
            logger.error(f"Error analyzing shopping funnel: {e}")
            return {}

    def get_user_behavior_analysis(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """获取用户行为分析报告"""
        try:
            # 获取用户偏好
            preference = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).first()

            # 获取行为分析
            behavior_analysis = self.db.query(Memory).filter(
                and_(
                    Memory.user_id == int(user_id) if user_id.isdigit() else 0,
                    Memory.memory_type == "behavior_analysis",
                    Memory.created_at >= datetime.utcnow() - timedelta(days=days)
                )
            ).order_by(desc(Memory.created_at)).first()

            # 获取搜索历史
            search_history = self.db.query(SearchHistory).filter(
                and_(
                    SearchHistory.user_id == user_id,
                    SearchHistory.created_at >= datetime.utcnow() - timedelta(days=days)
                )
            ).all()

            # 获取推荐日志
            recommendation_logs = self.db.query(RecommendationLog).filter(
                and_(
                    RecommendationLog.user_id == user_id,
                    RecommendationLog.created_at >= datetime.utcnow() - timedelta(days=days)
                )
            ).all()

            # 生成综合分析
            comprehensive_analysis = self._generate_comprehensive_analysis(
                user_id, preference, behavior_analysis, search_history, recommendation_logs
            )

            return comprehensive_analysis

        except Exception as e:
            logger.error(f"Error getting user behavior analysis: {e}")
            return {"error": str(e)}

    def _generate_comprehensive_analysis(self, user_id: str, preference, behavior_analysis,
                                       search_history, recommendation_logs) -> Dict[str, Any]:
        """生成综合分析报告"""
        analysis = {
            "user_id": user_id,
            "analysis_date": datetime.utcnow().isoformat(),
            "user_profile": {},
            "behavior_patterns": {},
            "shopping_insights": {},
            "recommendations": []
        }

        # 用户画像
        if preference:
            analysis["user_profile"] = {
                "preferred_brands": preference.preferred_brands or [],
                "preferred_categories": preference.preferred_categories or [],
                "price_range": {
                    "min": preference.price_range_min,
                    "max": preference.price_range_max
                },
                "preferred_features": preference.preferred_features or [],
                "total_purchases": len(preference.purchase_history or []),
                "total_browsing_actions": len(preference.browsing_history or [])
            }

        # 行为模式
        if behavior_analysis:
            try:
                behavior_data = json.loads(behavior_analysis.content)
                analysis["behavior_patterns"] = behavior_data
            except:
                pass

        # 购物洞察
        analysis["shopping_insights"] = self._generate_shopping_insights(
            search_history, recommendation_logs
        )

        # 个性化推荐
        analysis["recommendations"] = self._generate_personalized_recommendations(
            user_id, preference, analysis
        )

        return analysis

    def _generate_shopping_insights(self, search_history, recommendation_logs) -> Dict[str, Any]:
        """生成购物洞察"""
        insights = {
            "search_trends": {},
            "recommendation_effectiveness": {},
            "engagement_metrics": {}
        }

        # 搜索趋势
        if search_history:
            search_queries = [h.query for h in search_history]
            common_terms = Counter()
            for query in search_queries:
                terms = query.lower().split()
                common_terms.update(terms)

            insights["search_trends"] = {
                "total_searches": len(search_history),
                "common_search_terms": dict(common_terms.most_common(10)),
                "average_searches_per_day": len(search_history) / 30
            }

        # 推荐效果
        if recommendation_logs:
            total_recommendations = len(recommendation_logs)
            total_conversions = sum(log.conversion_rate or 0 for log in recommendation_logs)
            avg_conversion_rate = total_conversions / total_recommendations if total_recommendations > 0 else 0

            insights["recommendation_effectiveness"] = {
                "total_recommendations": total_recommendations,
                "average_conversion_rate": round(avg_conversion_rate * 100, 2),
                "most_recommended_categories": self._get_most_recommended_categories(recommendation_logs)
            }

        return insights

    def _get_most_recommended_categories(self, recommendation_logs) -> List[str]:
        """获取推荐最多的类别"""
        categories = defaultdict(int)
        for log in recommendation_logs:
            if log.recommended_products:
                try:
                    products = json.loads(log.recommended_products)
                    for product in products:
                        category = product.get("category", "unknown")
                        categories[category] += 1
                except:
                    pass

        return [cat for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]]

    def _generate_personalized_recommendations(self, user_id: str, preference, analysis: Dict[str, Any]) -> List[str]:
        """生成个性化推荐建议"""
        recommendations = []

        # 基于用户偏好的推荐
        if preference and preference.preferred_categories:
            top_category = preference.preferred_categories[0]
            recommendations.append(f"基于您对{top_category}的兴趣，建议为您推荐相关的新品和热销商品")

        # 基于行为模式的推荐
        behavior_patterns = analysis.get("behavior_patterns", {})
        funnel_analysis = behavior_patterns.get("funnel_analysis", {})
        conversion_rates = funnel_analysis.get("conversion_rates", {})

        if conversion_rates.get("view_to_cart", 0) < 0.3:
            recommendations.append("建议优化商品详情页信息，提高加购率")

        if conversion_rates.get("cart_to_purchase", 0) < 0.5:
            recommendations.append("可以为您提供更多优惠信息或购物建议，提高成交率")

        # 基于时间模式的推荐
        time_patterns = behavior_patterns.get("time_patterns", {})
        most_active_hour = time_patterns.get("most_active_hour")
        if most_active_hour:
            recommendations.append(f"您通常在{most_active_hour}点最活跃，建议在这个时间段为您推送优惠信息")

        return recommendations

class PersonalizedShoppingService:
    """个性化购物服务"""

    def __init__(self, db: Session):
        self.db = db
        self.behavior_tracker = UserBehaviorTracker(db)
        self.llm_service = LLMService()

    async def create_personalized_shopping_journey(self, user_id: str,
                                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建个性化购物旅程"""
        try:
            # 获取用户行为分析
            behavior_analysis = self.behavior_tracker.get_user_behavior_analysis(user_id)

            # 生成个性化旅程
            journey = await self._generate_personalized_journey(user_id, behavior_analysis, context)

            return {
                "success": True,
                "data": journey
            }

        except Exception as e:
            logger.error(f"Error creating personalized shopping journey: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_personalized_journey(self, user_id: str, behavior_analysis: Dict[str, Any],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成个性化购物旅程"""
        # 基于用户画像和行为数据生成旅程
        user_profile = behavior_analysis.get("user_profile", {})
        behavior_patterns = behavior_analysis.get("behavior_patterns", {})

        # 生成旅程阶段
        journey_stages = [
            {
                "stage": "discovery",
                "title": "发现",
                "activities": self._generate_discovery_activities(user_profile),
                "estimated_duration": "5-10分钟"
            },
            {
                "stage": "research",
                "title": "研究",
                "activities": self._generate_research_activities(user_profile),
                "estimated_duration": "15-30分钟"
            },
            {
                "stage": "comparison",
                "title": "比较",
                "activities": self._generate_comparison_activities(user_profile),
                "estimated_duration": "10-20分钟"
            },
            {
                "stage": "decision",
                "title": "决策",
                "activities": self._generate_decision_activities(user_profile, behavior_patterns),
                "estimated_duration": "5-15分钟"
            },
            {
                "stage": "purchase",
                "title": "购买",
                "activities": self._generate_purchase_activities(user_profile),
                "estimated_duration": "3-5分钟"
            }
        ]

        return {
            "user_id": user_id,
            "journey_type": "personalized_shopping",
            "created_at": datetime.utcnow().isoformat(),
            "stages": journey_stages,
            "estimated_total_time": "38-80分钟",
            "personalization_factors": self._get_personalization_factors(user_profile, behavior_patterns)
        }

    def _generate_discovery_activities(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成发现阶段活动"""
        activities = []

        # 基于偏好类别推荐
        preferred_categories = user_profile.get("preferred_categories", [])
        if preferred_categories:
            activities.append({
                "type": "category_browse",
                "title": f"浏览{preferred_categories[0]}新品",
                "description": "查看最新上架的相关商品",
                "priority": "high"
            })

        # 基于偏好品牌推荐
        preferred_brands = user_profile.get("preferred_brands", [])
        if preferred_brands:
            activities.append({
                "type": "brand_explore",
                "title": f"探索{preferred_brands[0]}品牌",
                "description": "了解该品牌的最新产品和优惠",
                "priority": "medium"
            })

        # 趋势推荐
        activities.append({
            "type": "trending_products",
            "title": "查看热门商品",
            "description": "发现当前最受欢迎的商品",
            "priority": "medium"
        })

        return activities

    def _generate_research_activities(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成研究阶段活动"""
        activities = [
            {
                "type": "product_reviews",
                "title": "阅读用户评价",
                "description": "了解其他用户的真实使用体验",
                "priority": "high"
            },
            {
                "type": "price_comparison",
                "title": "价格对比",
                "description": "比较不同平台的价格差异",
                "priority": "high"
            },
            {
                "type": "specification_analysis",
                "title": "规格参数研究",
                "description": "深入了解产品技术参数",
                "priority": "medium"
            }
        ]

        return activities

    def _generate_comparison_activities(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成比较阶段活动"""
        activities = [
            {
                "type": "side_by_side_comparison",
                "title": "商品对比",
                "description": "同时对比多个商品的详细信息",
                "priority": "high"
            },
            {
                "type": "expert_reviews",
                "title": "专业评测",
                "description": "查看专业人士的深度评测",
                "priority": "medium"
            },
            {
                "type": "video_reviews",
                "title": "视频评测",
                "description": "观看产品使用视频和评测",
                "priority": "medium"
            }
        ]

        return activities

    def _generate_decision_activities(self, user_profile: Dict[str, Any],
                                   behavior_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成决策阶段活动"""
        activities = []

        # 基于转化率调整建议
        funnel_analysis = behavior_patterns.get("funnel_analysis", {})
        conversion_rates = funnel_analysis.get("conversion_rates", {})

        if conversion_rates.get("view_to_cart", 0) < 0.3:
            activities.append({
                "type": "cart_optimization",
                "title": "购物车优化",
                "description": "查看适用优惠券和组合购买建议",
                "priority": "high"
            })

        activities.append({
            "type": "final_price_check",
            "title": "最终价格确认",
            "description": "确认最终价格和优惠信息",
            "priority": "high"
        })

        activities.append({
            "type": "shipping_info",
            "title": "配送信息",
            "description": "确认配送时间和费用",
            "priority": "medium"
        })

        return activities

    def _generate_purchase_activities(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成购买阶段活动"""
        activities = [
            {
                "type": "payment_method",
                "title": "选择支付方式",
                "description": "选择最适合的支付方式",
                "priority": "high"
            },
            {
                "type": "order_review",
                "title": "订单确认",
                "description": "确认订单信息和收货地址",
                "priority": "high"
            },
            {
                "type": "post_purchase_support",
                "title": "售后服务",
                "description": "了解售后保障和退换货政策",
                "priority": "medium"
            }
        ]

        return activities

    def _get_personalization_factors(self, user_profile: Dict[str, Any],
                                   behavior_patterns: Dict[str, Any]) -> List[str]:
        """获取个性化因素"""
        factors = []

        # 基于用户画像的因素
        preferred_categories = user_profile.get("preferred_categories", [])
        if preferred_categories:
            factors.append(f"基于{preferred_categories[0]}购物偏好")

        preferred_brands = user_profile.get("preferred_brands", [])
        if preferred_brands:
            factors.append(f"偏好{preferred_brands[0]}品牌")

        price_range = user_profile.get("price_range", {})
        if price_range:
            factors.append(f"价格区间{price_range['min']}-{price_range['max']}元")

        # 基于行为模式的因素
        funnel_analysis = behavior_patterns.get("funnel_analysis", {})
        if funnel_analysis:
            factors.append("基于历史购物行为分析")

        return factors

# 全局服务实例
def get_shopping_behavior_service(db: Session) -> PersonalizedShoppingService:
    return PersonalizedShoppingService(db)