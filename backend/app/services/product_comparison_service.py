"""
高级产品对比和决策支持服务
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import logging

from ..models.ecommerce_models import (
    Product, ProductSpecification, PriceHistory, ProductReview,
    UserPreference, SearchHistory
)
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)

class ProductComparisonService:
    def __init__(self, db: Session, llm_service: LLMService, rag_service: RAGService):
        self.db = db
        self.llm_service = llm_service
        self.rag_service = rag_service

    async def compare_products(self, product_ids: List[str], user_id: str = None,
                             comparison_weights: Dict[str, float] = None) -> Dict:
        """高级产品对比"""
        # 获取产品基本信息
        products = self.db.query(Product).filter(
            Product.product_id.in_(product_ids)
        ).all()

        if len(products) != len(product_ids):
            found_ids = [p.product_id for p in products]
            missing_ids = set(product_ids) - set(found_ids)
            raise ValueError(f"以下商品未找到: {missing_ids}")

        # 获取用户偏好（如果有）
        user_preferences = await self._get_user_preferences(user_id) if user_id else {}

        # 默认权重
        default_weights = {
            "price": 0.25,
            "quality": 0.20,
            "brand": 0.15,
            "features": 0.15,
            "reviews": 0.15,
            "popularity": 0.10
        }

        # 合并用户自定义权重
        weights = {**default_weights, **(comparison_weights or {})}

        # 收集详细数据
        comparison_data = await self._gather_comparison_data(products)

        # 计算各项指标分数
        scores = await self._calculate_category_scores(products, comparison_data)

        # 应用用户偏好权重
        weighted_scores = await self._apply_user_preferences(scores, user_preferences, weights)

        # 生成AI分析报告
        ai_analysis = await self._generate_ai_analysis(products, comparison_data, weighted_scores, user_preferences)

        # 生成最终推荐
        recommendations = await self._generate_recommendations(products, weighted_scores, ai_analysis)

        return {
            "comparison_id": self._generate_comparison_id(),
            "timestamp": datetime.now().isoformat(),
            "products": [
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "brand": p.brand,
                    "category": p.category,
                    "price": p.price,
                    "original_price": p.original_price,
                    "discount_rate": p.discount_rate,
                    "platform": p.platform,
                    "image_url": p.image_url,
                    "product_url": p.product_url
                }
                for p in products
            ],
            "detailed_comparison": comparison_data,
            "category_scores": scores,
            "weighted_scores": weighted_scores,
            "ai_analysis": ai_analysis,
            "recommendations": recommendations,
            "weights_used": weights
        }

    async def _get_user_preferences(self, user_id: str) -> Dict:
        """获取用户偏好"""
        preferences = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()

        if not preferences:
            return {}

        # 获取用户的搜索和购买历史
        search_history = self.db.query(SearchHistory).filter(
            SearchHistory.user_id == user_id
        ).order_by(SearchHistory.created_at.desc()).limit(20).all()

        # 分析用户行为模式
        behavior_patterns = self._analyze_user_behavior(search_history)

        return {
            "preferred_brands": preferences.preferred_brands or [],
            "preferred_categories": preferences.preferred_categories or [],
            "price_range": {
                "min": preferences.price_range_min,
                "max": preferences.price_range_max
            },
            "preferred_features": preferences.preferred_features or [],
            "behavior_patterns": behavior_patterns
        }

    def _analyze_user_behavior(self, search_history: List[SearchHistory]) -> Dict:
        """分析用户行为模式"""
        if not search_history:
            return {}

        # 分析价格偏好
        price_mentions = []
        for search in search_history:
            if search.filters and 'price_range' in search.filters:
                price_mentions.extend([
                    search.filters['price_range'].get('min'),
                    search.filters['price_range'].get('max')
                ])

        avg_price_preference = np.mean([p for p in price_mentions if p]) if price_mentions else 0

        # 分析品牌偏好
        brand_mentions = {}
        for search in search_history:
            if search.filters and 'brands' in search.filters:
                for brand in search.filters['brands']:
                    brand_mentions[brand] = brand_mentions.get(brand, 0) + 1

        return {
            "avg_price_preference": avg_price_preference,
            "top_brands": sorted(brand_mentions.items(), key=lambda x: x[1], reverse=True)[:3],
            "search_frequency": len(search_history)
        }

    async def _gather_comparison_data(self, products: List[Product]) -> Dict:
        """收集对比数据"""
        product_ids = [p.product_id for p in products]
        data = {}

        for product in products:
            product_id = product.product_id
            data[product_id] = {
                "specifications": await self._get_product_specs(product_id),
                "price_history": await self._get_price_history_summary(product_id),
                "reviews": await self._get_reviews_summary(product_id),
                "market_position": await self._get_market_position(product)
            }

        return data

    async def _get_product_specs(self, product_id: str) -> Dict:
        """获取产品规格"""
        specs = self.db.query(ProductSpecification).filter(
            ProductSpecification.product_id == product_id
        ).first()

        if not specs:
            return {}

        return {
            "screen_size": specs.screen_size,
            "processor": specs.processor,
            "ram": specs.ram,
            "storage": specs.storage,
            "battery": specs.battery,
            "camera": specs.camera,
            "os": specs.os,
            "weight": specs.weight,
            "colors": specs.colors,
            "network": specs.network,
            "features": specs.features
        }

    async def _get_price_history_summary(self, product_id: str) -> Dict:
        """获取价格历史摘要"""
        history = self.db.query(PriceHistory).filter(
            PriceHistory.product_id == product_id
        ).order_by(PriceHistory.date.desc()).limit(30).all()

        if not history:
            return {}

        prices = [h.price for h in history]
        return {
            "current_price": prices[0],
            "avg_price_30d": np.mean(prices[:min(30, len(prices))]),
            "min_price_30d": min(prices),
            "max_price_30d": max(prices),
            "price_trend": "up" if len(prices) >= 2 and prices[0] > prices[-1] else "down",
            "volatility": np.std(prices) / np.mean(prices) * 100 if len(prices) > 1 else 0
        }

    async def _get_reviews_summary(self, product_id: str) -> Dict:
        """获取评价摘要"""
        reviews = self.db.query(ProductReview).filter(
            ProductReview.product_id == product_id
        ).all()

        if not reviews:
            return {}

        ratings = [r.rating for r in reviews]
        verified_reviews = [r for r in reviews if r.verified_purchase]

        return {
            "avg_rating": np.mean(ratings),
            "total_reviews": len(reviews),
            "verified_reviews": len(verified_reviews),
            "rating_distribution": self._get_rating_distribution(reviews),
            "sentiment_score": np.mean([r.sentiment_score for r in reviews if r.sentiment_score]),
            "common_pros": self._extract_common_feedback(reviews, 'pros'),
            "common_cons": self._extract_common_feedback(reviews, 'cons')
        }

    def _get_rating_distribution(self, reviews: List[ProductReview]) -> Dict:
        """获取评分分布"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for review in reviews:
            rating = int(review.rating)
            if 1 <= rating <= 5:
                distribution[rating] += 1

        total = len(reviews)
        return {k: v/total*100 for k, v in distribution.items()} if total > 0 else distribution

    def _extract_common_feedback(self, reviews: List[ProductReview], feedback_type: str) -> List[str]:
        """提取常见反馈"""
        feedback_items = []
        for review in reviews:
            if feedback_type == 'pros' and review.pros:
                feedback_items.extend(review.pros.split('/'))
            elif feedback_type == 'cons' and review.cons:
                feedback_items.extend(review.cons.split('/'))

        from collections import Counter
        return [item for item, count in Counter(feedback_items).most_common(3) if item.strip()]

    async def _get_market_position(self, product: Product) -> Dict:
        """获取市场定位"""
        # 同类商品对比
        similar_products = self.db.query(Product).filter(
            and_(
                Product.category == product.category,
                Product.product_id != product.product_id
            )
        ).all()

        if not similar_products:
            return {}

        prices = [p.price for p in similar_products] + [product.price]
        ratings = []

        for p in similar_products:
            reviews = self.db.query(ProductReview).filter(
                ProductReview.product_id == p.product_id
            ).all()
            if reviews:
                ratings.append(np.mean([r.rating for r in reviews]))

        if ratings:
            product_reviews = self.db.query(ProductReview).filter(
                ProductReview.product_id == product.product_id
            ).all()
            product_rating = np.mean([r.rating for r in product_reviews]) if product_reviews else 0
            ratings.append(product_rating)

        return {
            "price_percentile": np.percentile(prices, product.price),
            "rating_percentile": np.percentile(ratings, product_rating) if ratings else 50,
            "market_share_estimate": len(similar_products) / 10,  # 简化估算
            "competitor_count": len(similar_products)
        }

    async def _calculate_category_scores(self, products: List[Product], comparison_data: Dict) -> Dict:
        """计算各类别分数"""
        scores = {}

        for product in products:
            product_id = product.product_id
            data = comparison_data[product_id]
            scores[product_id] = {}

            # 价格分数 (1-100，越低越好)
            current_price = product.price
            original_price = product.original_price or current_price
            price_score = max(0, 100 - (current_price / original_price * 100))
            scores[product_id]["price"] = round(price_score, 2)

            # 质量分数 (基于规格和评价)
            quality_score = 0
            if data["specifications"]:
                # 基于规格的综合评分
                spec_score = self._calculate_spec_score(data["specifications"])
                quality_score += spec_score * 0.6

            if data["reviews"]:
                # 基于评价的评分
                review_score = data["reviews"]["avg_rating"] * 20  # 5分制转换为100分制
                quality_score += review_score * 0.4

            scores[product_id]["quality"] = round(quality_score, 2)

            # 品牌分数 (基于品牌知名度和一致性)
            brand_score = self._calculate_brand_score(product.brand, product.category)
            scores[product_id]["brand"] = round(brand_score, 2)

            # 功能分数 (基于特性和创新性)
            feature_score = self._calculate_feature_score(data["specifications"])
            scores[product_id]["features"] = round(feature_score, 2)

            # 评价分数 (基于用户反馈)
            review_score = data["reviews"]["avg_rating"] * 20 if data["reviews"] else 50
            scores[product_id]["reviews"] = round(review_score, 2)

            # 受欢迎程度 (基于销量和关注度)
            popularity_score = min(100, data["market_position"]["market_share_estimate"] * 10 + 50)
            scores[product_id]["popularity"] = round(popularity_score, 2)

        return scores

    def _calculate_spec_score(self, specs: Dict) -> float:
        """计算规格分数"""
        score = 0
        factors = 0

        # 简化的规格评分逻辑
        if specs.get("ram"):
            ram_score = min(100, float(specs["ram"].replace('GB', '')) * 10)
            score += ram_score
            factors += 1

        if specs.get("storage"):
            storage_score = min(100, float(specs["storage"].replace('GB', '')) * 2)
            score += storage_score
            factors += 1

        if specs.get("camera"):
            camera_score = min(100, float(specs["camera"].replace('MP', '')) * 5)
            score += camera_score
            factors += 1

        return score / factors if factors > 0 else 50

    def _calculate_brand_score(self, brand: str, category: str) -> float:
        """计算品牌分数"""
        # 这里可以实现更复杂的品牌评分逻辑
        # 目前使用简化的评分
        premium_brands = ["Apple", "Samsung", "Sony", "LG", "Nike", "Adidas"]
        mid_brands = ["Xiaomi", "Huawei", "Oppo", "Vivo", "Puma"]

        if brand in premium_brands:
            return 90
        elif brand in mid_brands:
            return 75
        else:
            return 60

    def _calculate_feature_score(self, specs: Dict) -> float:
        """计算功能分数"""
        if not specs:
            return 50

        score = 50  # 基础分

        # 基于特性加分
        if specs.get("features"):
            features = specs["features"].lower()
            if "5g" in features:
                score += 15
            if "wireless" in features:
                score += 10
            if "waterproof" in features:
                score += 10
            if "fast_charge" in features:
                score += 10

        return min(100, score)

    async def _apply_user_preferences(self, scores: Dict, user_preferences: Dict, weights: Dict) -> Dict:
        """应用用户偏好权重"""
        weighted_scores = {}

        for product_id, category_scores in scores.items():
            total_score = 0
            total_weight = 0

            for category, score in category_scores.items():
                weight = weights.get(category, 0)

                # 应用用户偏好调整
                if user_preferences:
                    # 品牌偏好
                    if category == "brand" and user_preferences.get("preferred_brands"):
                        # 这里需要产品信息来检查品牌匹配
                        pass

                    # 价格范围偏好
                    if category == "price" and user_preferences.get("price_range"):
                        # 这里需要产品价格来检查范围匹配
                        pass

                total_score += score * weight
                total_weight += weight

            weighted_scores[product_id] = {
                "weighted_total": round(total_score / total_weight if total_weight > 0 else 0, 2),
                "category_scores": category_scores,
                "applied_weights": weights
            }

        return weighted_scores

    async def _generate_ai_analysis(self, products: List[Product], comparison_data: Dict,
                                   weighted_scores: Dict, user_preferences: Dict) -> Dict:
        """生成AI分析报告"""
        # 构建分析提示
        comparison_text = self._build_comparison_text(products, comparison_data, weighted_scores)

        prompt = f"""
        作为专业的产品分析师，请对以下产品对比进行深入分析：

        用户偏好: {json.dumps(user_preferences, ensure_ascii=False)}

        产品对比数据:
        {comparison_text}

        请提供以下分析：

        1. **产品优势分析**：每个产品的主要优势
        2. **产品劣势分析**：每个产品的主要不足
        3. **适用场景分析**：每个产品最适合的用户群体和使用场景
        4. **性价比分析**：价格与价值的匹配度
        5. **购买时机建议**：当前是否是购买的好时机
        6. **未来展望**：产品的升级潜力和保值性

        请以JSON格式返回分析结果。
        """

        try:
            response = await self.llm_service.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return self._get_default_analysis(products)

    def _build_comparison_text(self, products: List[Product], comparison_data: Dict, weighted_scores: Dict) -> str:
        """构建对比文本"""
        text = "产品对比信息:\n\n"

        for product in products:
            product_id = product.product_id
            data = comparison_data[product_id]
            scores = weighted_scores[product_id]

            text += f"""
产品: {product.title} ({product.brand})
价格: ¥{product.price}
评分: {scores['weighted_total']}/100
规格: {data['specifications']}
价格趋势: {data['price_history']['price_trend']}
用户评价: {data['reviews']['avg_rating']}/5 ({data['reviews']['total_reviews']}条评价)
市场定位: 价格百分位{data['market_position']['price_percentile']:.1f}%
"""

        return text

    def _get_default_analysis(self, products: List[Product]) -> Dict:
        """获取默认分析"""
        return {
            "strengths": {p.product_id: f"{p.name} 具有良好的性价比" for p in products},
            "weaknesses": {p.product_id: f"{p.name} 在某些方面还有提升空间" for p in products},
            "use_cases": {p.product_id: f"适合日常使用" for p in products},
            "value_analysis": "建议根据个人需求和预算选择",
            "timing_recommendation": "如果需求明确，可以考虑购买",
            "future_outlook": "产品具有正常的升级周期"
        }

    async def _generate_recommendations(self, products: List[Product], weighted_scores: Dict, ai_analysis: Dict) -> List[Dict]:
        """生成推荐"""
        # 按总分排序
        sorted_products = sorted(weighted_scores.items(), key=lambda x: x[1]['weighted_total'], reverse=True)

        recommendations = []
        for i, (product_id, scores) in enumerate(sorted_products):
            product = next(p for p in products if p.product_id == product_id)

            recommendation = {
                "rank": i + 1,
                "product_id": product_id,
                "product_name": product.title,
                "total_score": scores['weighted_total'],
                "match_percentage": min(100, scores['weighted_total']),
                "key_reasons": [],
                "suitable_for": []
            }

            # 基于分数生成推荐理由
            if scores['category_scores']['price'] >= 80:
                recommendation["key_reasons"].append("价格优势明显")
            if scores['category_scores']['quality'] >= 80:
                recommendation["key_reasons"].append("品质优秀")
            if scores['category_scores']['features'] >= 80:
                recommendation["key_reasons"].append("功能丰富")

            # 基于AI分析添加适用场景
            if product_id in ai_analysis.get("use_cases", {}):
                recommendation["suitable_for"].append(ai_analysis["use_cases"][product_id])

            recommendations.append(recommendation)

        return recommendations

    def _generate_comparison_id(self) -> str:
        """生成对比ID"""
        import uuid
        return str(uuid.uuid4())

    async def get_personalized_recommendations(self, user_id: str, category: str = None,
                                           budget_range: Tuple[float, float] = None,
                                           limit: int = 10) -> Dict:
        """获取个性化推荐"""
        # 获取用户偏好
        user_preferences = await self._get_user_preferences(user_id)

        # 构建查询
        query = self.db.query(Product)

        # 应用过滤器
        if category:
            query = query.filter(Product.category == category)

        if budget_range:
            query = query.filter(
                and_(
                    Product.price >= budget_range[0],
                    Product.price <= budget_range[1]
                )
            )

        # 应用用户偏好过滤
        if user_preferences.get("preferred_brands"):
            query = query.filter(Product.brand.in_(user_preferences["preferred_brands"]))

        products = query.limit(limit * 2).all()  # 获取更多候选产品

        if not products:
            return {"recommendations": [], "reason": "未找到匹配的产品"}

        # 进行产品对比和评分
        product_ids = [p.product_id for p in products]
        comparison_result = await self.compare_products(product_ids, user_id)

        # 返回排序后的推荐
        return {
            "recommendations": comparison_result["recommendations"][:limit],
            "user_preferences_applied": bool(user_preferences),
            "total_candidates": len(products)
        }

    async def get_smart_decision_tree(self, user_id: str, needs_analysis: Dict) -> Dict:
        """智能决策树"""
        # 基于用户需求生成决策问题
        decision_tree = {
            "current_question": self._get_initial_decision_question(needs_analysis),
            "options": [],
            "path": [],
            "recommendations": []
        }

        # 生成决策路径
        for question_key, question_data in self._get_decision_questions().items():
            if self._should_ask_question(question_key, needs_analysis):
                decision_tree["options"].append(question_data)

        return decision_tree

    def _get_initial_decision_question(self, needs_analysis: Dict) -> str:
        """获取初始决策问题"""
        primary_need = needs_analysis.get("primary_need", "general")

        questions = {
            "performance": "您更看重产品的性能表现还是日常使用的便利性？",
            "budget": "您的预算范围是什么？",
            "lifestyle": "您的主要使用场景是什么？",
            "general": "请告诉我您选择产品时最看重的因素是什么？"
        }

        return questions.get(primary_need, questions["general"])

    def _get_decision_questions(self) -> Dict:
        """获取决策问题集"""
        return {
            "budget": {
                "question": "您的预算范围是？",
                "options": ["3000以下", "3000-5000", "5000-8000", "8000以上"],
                "weight": 0.3
            },
            "usage": {
                "question": "主要用途是？",
                "options": ["日常使用", "工作/学习", "娱乐/游戏", "专业用途"],
                "weight": 0.25
            },
            "brand_preference": {
                "question": "对品牌有偏好吗？",
                "options": ["国际大牌", "国产精品", "无所谓"],
                "weight": 0.2
            },
            "priority": {
                "question": "最看重什么？",
                "options": ["性能", "外观", "价格", "续航"],
                "weight": 0.25
            }
        }

    def _should_ask_question(self, question_key: str, needs_analysis: Dict) -> bool:
        """判断是否需要问某个问题"""
        # 基于用户需求分析决定是否需要询问
        return True  # 简化实现