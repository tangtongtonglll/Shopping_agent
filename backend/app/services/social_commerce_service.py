"""
社交商务集成服务
整合社交媒体、用户生成内容和社交推荐功能
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import json
import logging
from datetime import datetime, timedelta
import aiohttp
import asyncio
from ..models.models import KnowledgeBase, Document, DocumentChunk, User
from ..models.ecommerce_models import Product, ProductReview, UserPreference
from ..services.llm_service import LLMService
from ..services.vector_service import vector_service
from ..core.config import settings

logger = logging.getLogger(__name__)

class SocialPlatform:
    """社交平台基类"""
    def __init__(self, platform_name: str, config: Dict[str, Any]):
        self.platform_name = platform_name
        self.config = config
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.access_token = config.get("access_token")
        self.is_active = config.get("is_active", True)

    async def fetch_trending_products(self) -> List[Dict[str, Any]]:
        """获取热门商品 - 子类需要实现"""
        raise NotImplementedError

    async def fetch_user_reviews(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户评价 - 子类需要实现"""
        raise NotImplementedError

    async def fetch_influencer_content(self, category: str = None) -> List[Dict[str, Any]]:
        """获取网红内容 - 子类需要实现"""
        raise NotImplementedError

    async def post_product_share(self, product_id: str, content: str, user_id: str) -> Dict[str, Any]:
        """分享商品 - 子类需要实现"""
        raise NotImplementedError

class WeiboPlatform(SocialPlatform):
    """微博平台集成"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("weibo", config)

    async def fetch_trending_products(self) -> List[Dict[str, Any]]:
        """获取微博热门商品"""
        try:
            trending_products = []

            # 搜索热门商品话题
            search_keywords = ["好物推荐", "种草", "购物分享", "好物测评"]
            for keyword in search_keywords:
                posts = await self._search_weibo_posts(keyword, limit=10)

                for post in posts:
                    # 提取商品信息
                    product_info = self._extract_product_from_post(post)
                    if product_info:
                        trending_products.append(product_info)

            return trending_products[:20]  # 限制返回数量

        except Exception as e:
            logger.error(f"Error fetching trending products from Weibo: {e}")
            return []

    async def _search_weibo_posts(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索微博帖子"""
        try:
            # 这里应该调用微博API
            # 简化版本，返回模拟数据
            return [
                {
                    "id": f"weibo_{hash(keyword)}_{i}",
                    "content": f"刚买了{keyword}，真的很不错！推荐给大家~",
                    "author": f"user_{i}",
                    "likes": 100 + i * 10,
                    "comments": 20 + i * 5,
                    "shares": 10 + i * 2,
                    "created_at": datetime.utcnow().isoformat()
                }
                for i in range(limit)
            ]
        except Exception as e:
            logger.error(f"Error searching Weibo posts: {e}")
            return []

    def _extract_product_from_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从帖子中提取商品信息"""
        try:
            content = post.get("content", "")

            # 简单的关键词匹配
            if any(keyword in content for keyword in ["推荐", "种草", "不错", "购买"]):
                return {
                    "platform": "weibo",
                    "post_id": post.get("id"),
                    "content": content,
                    "author": post.get("author"),
                    "engagement": {
                        "likes": post.get("likes", 0),
                        "comments": post.get("comments", 0),
                        "shares": post.get("shares", 0)
                    },
                    "extracted_at": datetime.utcnow().isoformat(),
                    "sentiment": self._analyze_sentiment(content)
                }

            return None

        except Exception as e:
            logger.error(f"Error extracting product from post: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> str:
        """简单情感分析"""
        positive_words = ["好", "不错", "推荐", "喜欢", "棒", "优秀", "满意"]
        negative_words = ["差", "糟糕", "失望", "退货", "问题", "垃圾"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

class XiaohongshuPlatform(SocialPlatform):
    """小红书平台集成"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("xiaohongshu", config)

    async def fetch_trending_products(self) -> List[Dict[str, Any]]:
        """获取小红书热门商品"""
        try:
            trending_products = []

            # 搜索热门笔记
            search_keywords = ["好物", "分享", "测评", "穿搭", "美妆"]
            for keyword in search_keywords:
                notes = await self._search_xiaohongshu_notes(keyword, limit=10)

                for note in notes:
                    product_info = self._extract_product_from_note(note)
                    if product_info:
                        trending_products.append(product_info)

            return trending_products[:20]

        except Exception as e:
            logger.error(f"Error fetching trending products from Xiaohongshu: {e}")
            return []

    async def _search_xiaohongshu_notes(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索小红书笔记"""
        try:
            # 这里应该调用小红书API
            # 简化版本，返回模拟数据
            return [
                {
                    "id": f"xhs_{hash(keyword)}_{i}",
                    "title": f"{keyword}分享，真心推荐！",
                    "content": f"今天给大家分享一个超赞的{keyword}，使用感受很不错~",
                    "author": f"xhs_user_{i}",
                    "likes": 200 + i * 15,
                    "comments": 30 + i * 8,
                    "saves": 50 + i * 10,
                    "created_at": datetime.utcnow().isoformat(),
                    "tags": [keyword, "好物", "分享"]
                }
                for i in range(limit)
            ]
        except Exception as e:
            logger.error(f"Error searching Xiaohongshu notes: {e}")
            return []

    def _extract_product_from_note(self, note: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从笔记中提取商品信息"""
        try:
            content = note.get("content", "")
            title = note.get("title", "")

            if any(keyword in content + title for keyword in ["推荐", "分享", "测评", "不错"]):
                return {
                    "platform": "xiaohongshu",
                    "note_id": note.get("id"),
                    "title": title,
                    "content": content,
                    "author": note.get("author"),
                    "engagement": {
                        "likes": note.get("likes", 0),
                        "comments": note.get("comments", 0),
                        "saves": note.get("saves", 0)
                    },
                    "tags": note.get("tags", []),
                    "extracted_at": datetime.utcnow().isoformat(),
                    "sentiment": self._analyze_sentiment(content)
                }

            return None

        except Exception as e:
            logger.error(f"Error extracting product from note: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> str:
        """简单情感分析"""
        positive_words = ["推荐", "不错", "喜欢", "棒", "超赞", "种草"]
        negative_words = ["差", "糟糕", "失望", "不推荐", "踩雷"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

class DouyinPlatform(SocialPlatform):
    """抖音平台集成"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("douyin", config)

    async def fetch_trending_products(self) -> List[Dict[str, Any]]:
        """获取抖音热门商品"""
        try:
            trending_products = []

            # 搜索热门视频
            search_keywords = ["好物", "推荐", "测评", "开箱", "种草"]
            for keyword in search_keywords:
                videos = await self._search_douyin_videos(keyword, limit=10)

                for video in videos:
                    product_info = self._extract_product_from_video(video)
                    if product_info:
                        trending_products.append(product_info)

            return trending_products[:20]

        except Exception as e:
            logger.error(f"Error fetching trending products from Douyin: {e}")
            return []

    async def _search_douyin_videos(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索抖音视频"""
        try:
            # 这里应该调用抖音API
            # 简化版本，返回模拟数据
            return [
                {
                    "id": f"douyin_{hash(keyword)}_{i}",
                    "title": f"必看！这个{keyword}真的绝了！",
                    "description": f"给大家推荐一个超好用的{keyword}，效果真的很棒~",
                    "author": f"douyin_user_{i}",
                    "views": 10000 + i * 1000,
                    "likes": 500 + i * 50,
                "comments": 100 + i * 20,
                "shares": 50 + i * 10,
                    "created_at": datetime.utcnow().isoformat()
                }
                for i in range(limit)
            ]
        except Exception as e:
            logger.error(f"Error searching Douyin videos: {e}")
            return []

    def _extract_product_from_video(self, video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从视频中提取商品信息"""
        try:
            title = video.get("title", "")
            description = video.get("description", "")

            if any(keyword in title + description for keyword in ["推荐", "必看", "绝了", "好用", "种草"]):
                return {
                    "platform": "douyin",
                    "video_id": video.get("id"),
                    "title": title,
                    "description": description,
                    "author": video.get("author"),
                    "engagement": {
                        "views": video.get("views", 0),
                        "likes": video.get("likes", 0),
                        "comments": video.get("comments", 0),
                        "shares": video.get("shares", 0)
                    },
                    "extracted_at": datetime.utcnow().isoformat(),
                    "sentiment": self._analyze_sentiment(title + " " + description)
                }

            return None

        except Exception as e:
            logger.error(f"Error extracting product from video: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> str:
        """简单情感分析"""
        positive_words = ["推荐", "必看", "绝了", "好用", "种草", "棒"]
        negative_words = ["差", "糟糕", "失望", "不推荐", "踩雷"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

class SocialCommerceService:
    """社交商务服务"""

    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
        self.platforms = {}
        self._initialize_platforms()

    def _initialize_platforms(self):
        """初始化社交平台"""
        # 从配置文件或数据库加载平台配置
        platform_configs = {
            "weibo": {
                "is_active": True,
                "api_key": "your_weibo_api_key",
                "api_secret": "your_weibo_api_secret"
            },
            "xiaohongshu": {
                "is_active": True,
                "api_key": "your_xiaohongshu_api_key",
                "api_secret": "your_xiaohongshu_api_secret"
            },
            "douyin": {
                "is_active": True,
                "api_key": "your_douyin_api_key",
                "api_secret": "your_douyin_api_secret"
            }
        }

        # 初始化平台实例
        if platform_configs["weibo"]["is_active"]:
            self.platforms["weibo"] = WeiboPlatform(platform_configs["weibo"])

        if platform_configs["xiaohongshu"]["is_active"]:
            self.platforms["xiaohongshu"] = XiaohongshuPlatform(platform_configs["xiaohongshu"])

        if platform_configs["douyin"]["is_active"]:
            self.platforms["douyin"] = DouyinPlatform(platform_configs["douyin"])

    async def get_trending_social_products(self, category: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """获取社交平台热门商品"""
        try:
            all_trending = []

            # 从各平台获取热门商品
            for platform_name, platform in self.platforms.items():
                if platform.is_active:
                    trending_products = await platform.fetch_trending_products()
                    all_trending.extend(trending_products)

            # 按参与度排序
            all_trending.sort(key=lambda x: self._calculate_engagement_score(x), reverse=True)

            # 过滤分类
            if category:
                all_trending = [item for item in all_trending if category in item.get("content", "")]

            return all_trending[:limit]

        except Exception as e:
            logger.error(f"Error getting trending social products: {e}")
            return []

    def _calculate_engagement_score(self, social_post: Dict[str, Any]) -> float:
        """计算社交参与度分数"""
        try:
            engagement = social_post.get("engagement", {})
            platform = social_post.get("platform", "")

            # 不同平台的权重
            platform_weights = {
                "weibo": 1.0,
                "xiaohongshu": 1.2,
                "douyin": 1.5
            }

            weight = platform_weights.get(platform, 1.0)

            # 计算基础分数
            likes = engagement.get("likes", 0)
            comments = engagement.get("comments", 0)
            shares = engagement.get("shares", 0)
            views = engagement.get("views", 0)
            saves = engagement.get("saves", 0)

            # 加权计算
            score = (likes * 1.0 + comments * 2.0 + shares * 3.0 + views * 0.01 + saves * 1.5) * weight

            return float(score)

        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    async def generate_social_recommendations(self, user_id: str, product_id: str) -> Dict[str, Any]:
        """生成社交推荐内容"""
        try:
            # 获取用户偏好
            user_preference = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).first()

            # 获取商品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()

            if not product:
                return {"success": False, "error": "商品不存在"}

            # 获取社交内容
            social_content = await self._get_related_social_content(product)

            # 生成推荐内容
            recommendation = await self._generate_recommendation_content(
                user_preference, product, social_content
            )

            return {
                "success": True,
                "data": {
                    "user_id": user_id,
                    "product_id": product_id,
                    "recommendation": recommendation,
                    "social_content_count": len(social_content),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error generating social recommendations: {e}")
            return {"success": False, "error": str(e)}

    async def _get_related_social_content(self, product: Product) -> List[Dict[str, Any]]:
        """获取相关社交内容"""
        try:
            related_content = []

            # 搜索相关的社交内容
            search_terms = [product.title, product.brand, product.category]

            for term in search_terms:
                for platform_name, platform in self.platforms.items():
                    if platform.is_active:
                        # 简化的搜索逻辑
                        content = await self._search_platform_content(platform, term)
                        related_content.extend(content)

            # 去重并限制数量
            unique_content = []
            seen_ids = set()
            for content in related_content:
                content_id = content.get("id")
                if content_id and content_id not in seen_ids:
                    unique_content.append(content)
                    seen_ids.add(content_id)

            return unique_content[:10]  # 限制返回数量

        except Exception as e:
            logger.error(f"Error getting related social content: {e}")
            return []

    async def _search_platform_content(self, platform: SocialPlatform, search_term: str) -> List[Dict[str, Any]]:
        """搜索平台内容"""
        try:
            # 简化的搜索实现
            # 实际应用中应该调用各平台的API
            return [
                {
                    "id": f"{platform.platform_name}_{hash(search_term)}_{i}",
                    "platform": platform.platform_name,
                    "content": f"关于{search_term}的社交内容示例{i}",
                    "author": f"user_{i}",
                    "engagement": {"likes": 100, "comments": 20},
                    "sentiment": "positive"
                }
                for i in range(3)
            ]
        except Exception as e:
            logger.error(f"Error searching platform content: {e}")
            return []

    async def _generate_recommendation_content(self, user_preference: UserPreference,
                                             product: Product, social_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成推荐内容"""
        try:
            # 分析社交内容情感
            sentiment_analysis = self._analyze_social_sentiment(social_content)

            # 生成个性化推荐
            prompt = f"""
            基于以下信息，为用户生成个性化的社交推荐内容：

            商品信息：
            - 名称：{product.title}
            - 品牌：{product.brand}
            - 类别：{product.category}
            - 价格：{product.price}

            用户偏好：
            - 偏好品牌：{user_preference.preferred_brands or []}
            - 偏好类别：{user_preference.preferred_categories or []}
            - 价格区间：{user_preference.price_range_min}-{user_preference.price_range_max}

            社交内容分析：
            - 总体情感：{sentiment_analysis['overall_sentiment']}
            - 正面评价：{sentiment_analysis['positive_count']}
            - 负面评价：{sentiment_analysis['negative_count']}

            请生成：
            1. 个性化的推荐文案
            2. 适合的社交平台建议
            3. 推荐理由和卖点
            4. 用户互动建议

            返回JSON格式的推荐内容。
            """

            recommendation = await self.llm_service.generate_response(prompt)

            try:
                return json.loads(recommendation)
            except:
                return {"recommendation_text": recommendation}

        except Exception as e:
            logger.error(f"Error generating recommendation content: {e}")
            return {"recommendation_text": "无法生成推荐内容"}

    def _analyze_social_sentiment(self, social_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析社交内容情感"""
        try:
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for content in social_content:
                sentiment = content.get("sentiment", "neutral")
                if sentiment == "positive":
                    positive_count += 1
                elif sentiment == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1

            total = len(social_content)
            if total == 0:
                return {
                    "overall_sentiment": "neutral",
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "positive_ratio": 0.0
                }

            return {
                "overall_sentiment": "positive" if positive_count > negative_count else "negative",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_ratio": round(positive_count / total, 2)
            }

        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {
                "overall_sentiment": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_ratio": 0.0
            }

    async def create_social_sharing_campaign(self, product_id: str, campaign_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建社交分享活动"""
        try:
            # 获取商品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()

            if not product:
                return {"success": False, "error": "商品不存在"}

            # 生成活动内容
            campaign_content = await self._generate_campaign_content(product, campaign_config)

            return {
                "success": True,
                "data": {
                    "campaign_id": f"campaign_{product_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    "product_id": product_id,
                    "campaign_content": campaign_content,
                    "config": campaign_config,
                    "created_at": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error creating social sharing campaign: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_campaign_content(self, product: Product, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成活动内容"""
        try:
            platforms = config.get("platforms", ["weibo", "xiaohongshu", "douyin"])
            campaign_type = config.get("campaign_type", "product_promotion")
            target_audience = config.get("target_audience", "general")

            # 为每个平台生成内容
            platform_content = {}
            for platform in platforms:
                platform_content[platform] = await self._generate_platform_content(
                    product, platform, campaign_type, target_audience
                )

            return {
                "campaign_type": campaign_type,
                "target_audience": target_audience,
                "platform_content": platform_content,
                "product_info": {
                    "name": product.title,
                    "brand": product.brand,
                    "category": product.category,
                    "price": product.price
                }
            }

        except Exception as e:
            logger.error(f"Error generating campaign content: {e}")
            return {}

    async def _generate_platform_content(self, product: Product, platform: str,
                                        campaign_type: str, target_audience: str) -> Dict[str, Any]:
        """为特定平台生成内容"""
        try:
            # 根据平台特性生成内容
            platform_configs = {
                "weibo": {
                    "max_length": 140,
                    "style": "简洁直接",
                    "hashtags": ["#好物推荐", "#购物分享"]
                },
                "xiaohongshu": {
                    "max_length": 1000,
                    "style": "详细评测",
                    "hashtags": ["#好物", "#测评", "#种草"]
                },
                "douyin": {
                    "max_length": 200,
                    "style": "生动有趣",
                    "hashtags": ["#好物推荐", "#种草", "#必买"]
                }
            }

            config = platform_configs.get(platform, platform_configs["weibo"])

            prompt = f"""
            为{platform}平台生成商品推广内容：

            商品：{product.title}
            品牌：{product.brand}
            价格：{product.price}
            类别：{product.category}

            要求：
            - 风格：{config['style']}
            - 长度限制：{config['max_length']}字
            - 标签：{', '.join(config['hashtags'])}
            - 目标受众：{target_audience}
            - 活动类型：{campaign_type}

            请生成吸引人的推广文案。
            """

            content = await self.llm_service.generate_response(prompt)

            return {
                "content": content,
                "platform": platform,
                "style": config["style"],
                "hashtags": config["hashtags"],
                "length": len(content)
            }

        except Exception as e:
            logger.error(f"Error generating platform content: {e}")
            return {"content": "无法生成内容", "platform": platform}

    async def get_social_commerce_analytics(self, product_id: str = None, days: int = 30) -> Dict[str, Any]:
        """获取社交商务分析"""
        try:
            analytics = {
                "period": f"{days}天",
                "generated_at": datetime.utcnow().isoformat(),
                "platform_analytics": {},
                "content_performance": {},
                "sentiment_analysis": {},
                "trending_topics": []
            }

            # 各平台分析
            for platform_name, platform in self.platforms.items():
                if platform.is_active:
                    platform_analytics = await self._get_platform_analytics(platform_name, days)
                    analytics["platform_analytics"][platform_name] = platform_analytics

            # 整体性能分析
            analytics["content_performance"] = await self._get_content_performance(days)
            analytics["sentiment_analysis"] = await self._get_sentiment_analysis(days)
            analytics["trending_topics"] = await self._get_trending_topics(days)

            return {
                "success": True,
                "data": analytics
            }

        except Exception as e:
            logger.error(f"Error getting social commerce analytics: {e}")
            return {"success": False, "error": str(e)}

    async def _get_platform_analytics(self, platform_name: str, days: int) -> Dict[str, Any]:
        """获取平台分析数据"""
        try:
            # 简化的分析数据
            return {
                "platform": platform_name,
                "total_posts": 150,
                "total_engagement": 15000,
                "average_engagement": 100,
                "sentiment_distribution": {
                    "positive": 0.75,
                    "neutral": 0.20,
                    "negative": 0.05
                },
                "top_categories": ["服装", "美妆", "数码"],
                "growth_trend": "upward"
            }
        except Exception as e:
            logger.error(f"Error getting platform analytics: {e}")
            return {}

    async def _get_content_performance(self, days: int) -> Dict[str, Any]:
        """获取内容性能分析"""
        try:
            return {
                "total_content": 450,
                "high_performing_content": 120,
                "average_likes": 250,
                "average_comments": 45,
                "average_shares": 30,
                "best_posting_times": ["10:00", "14:00", "20:00"],
                "top_hashtags": ["#好物推荐", "#种草", "#购物分享"]
            }
        except Exception as e:
            logger.error(f"Error getting content performance: {e}")
            return {}

    async def _get_sentiment_analysis(self, days: int) -> Dict[str, Any]:
        """获取情感分析"""
        try:
            return {
                "overall_sentiment": "positive",
                "sentiment_distribution": {
                    "positive": 0.78,
                    "neutral": 0.18,
                    "negative": 0.04
                },
                "sentiment_trend": "improving",
                "key_positive_topics": ["质量好", "价格合理", "推荐购买"],
                "key_negative_topics": ["发货慢", "包装差"]
            }
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return {}

    async def _get_trending_topics(self, days: int) -> List[Dict[str, Any]]:
        """获取热门话题"""
        try:
            return [
                {"topic": "春季新品", "mentions": 1250, "sentiment": "positive"},
                {"topic": "护肤推荐", "mentions": 980, "sentiment": "positive"},
                {"topic": "数码测评", "mentions": 750, "sentiment": "neutral"},
                {"topic": "穿搭分享", "mentions": 620, "sentiment": "positive"}
            ]
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    async def share_product_to_social(self, user_id: str, product_id: str,
                                    platform: str, content: str) -> Dict[str, Any]:
        """分享商品到社交平台"""
        try:
            # 验证平台
            if platform not in self.platforms:
                return {"success": False, "error": f"不支持的平台: {platform}"}

            # 获取用户信息
            user = self.db.query(User).filter(User.id == int(user_id)).first()
            if not user:
                return {"success": False, "error": "用户不存在"}

            # 获取商品信息
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()
            if not product:
                return {"success": False, "error": "商品不存在"}

            # 调用平台API分享
            platform_instance = self.platforms[platform]
            share_result = await platform_instance.post_product_share(product_id, content, user_id)

            # 记录分享行为
            await self._record_share_action(user_id, product_id, platform, content)

            return {
                "success": True,
                "data": {
                    "share_id": f"share_{user_id}_{product_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "user_id": user_id,
                    "product_id": product_id,
                    "platform": platform,
                    "content": content,
                    "shared_at": datetime.utcnow().isoformat(),
                    "platform_result": share_result
                }
            }

        except Exception as e:
            logger.error(f"Error sharing product to social: {e}")
            return {"success": False, "error": str(e)}

    async def _record_share_action(self, user_id: str, product_id: str, platform: str, content: str):
        """记录分享行为"""
        try:
            # 创建分享记录
            share_record = {
                "user_id": user_id,
                "product_id": product_id,
                "platform": platform,
                "content": content,
                "shared_at": datetime.utcnow().isoformat(),
                "action_type": "social_share"
            }

            # 存储到内存系统
            from ..models.models import Memory
            memory = Memory(
                user_id=int(user_id),
                content=json.dumps(share_record, ensure_ascii=False),
                memory_type="social_share",
                importance_score=0.6,
                metadata={
                    "platform": platform,
                    "product_id": product_id
                }
            )

            self.db.add(memory)
            self.db.commit()

        except Exception as e:
            logger.error(f"Error recording share action: {e}")

# 全局社交商务服务实例
def get_social_commerce_service(db: Session) -> SocialCommerceService:
    return SocialCommerceService(db)