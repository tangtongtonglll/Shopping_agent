"""
电商相关数据模型
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Product(Base):
    """商品基本信息表"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    brand = Column(String(100), index=True)
    category = Column(String(100), index=True)
    price = Column(Float, nullable=False, default=0.0)
    original_price = Column(Float)
    discount_rate = Column(Float, default=0.0)
    platform = Column(String(50), index=True)
    product_url = Column(Text)
    image_url = Column(Text)
    stock_status = Column(String(20), default="有货")
    rating = Column(Float)
    review_count = Column(Integer)
    sales_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    price_history = relationship("PriceHistory", back_populates="product")
    reviews = relationship("ProductReview", back_populates="product")
    specifications = relationship("ProductSpecification", back_populates="product")
    images = relationship("ProductImage", back_populates="product")

class ProductSpecification(Base):
    """商品规格参数表"""
    __tablename__ = "product_specifications"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.product_id"), nullable=False)
    screen_size = Column(String)
    processor = Column(String)
    ram = Column(String)
    storage = Column(String)
    battery = Column(String)
    camera = Column(String)
    os = Column(String)
    weight = Column(String)
    material = Column(String)
    colors = Column(String)
    network = Column(String)
    features = Column(Text)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    product = relationship("Product", back_populates="specifications")

class PriceHistory(Base):
    """价格历史表"""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.product_id"), nullable=False)
    price = Column(Float, nullable=False)
    platform = Column(String, index=True)
    discount_type = Column(String)
    promotion_info = Column(Text)
    date = Column(DateTime, nullable=False, index=True)
    is_stock_available = Column(Boolean, default=True)
    seller_info = Column(Text)
    monthly_sales = Column(Integer, default=0)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    product = relationship("Product", back_populates="price_history")

class ProductReview(Base):
    """商品评价表"""
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.product_id"), nullable=False)
    username = Column(String, index=True)
    rating = Column(Float, nullable=False)  # 1.0-5.0
    content = Column(Text)
    pros = Column(Text)  # 优点，用/分隔
    cons = Column(Text)  # 缺点，用/分隔
    purchase_date = Column(DateTime)
    helpful_count = Column(Integer, default=0)
    verified_purchase = Column(Boolean, default=False)
    user_level = Column(String)
    tags = Column(Text)  # 标签，用/分隔
    sentiment_score = Column(Float)  # 情感分析分数
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    product = relationship("Product", back_populates="reviews")

class UserPreference(Base):
    """用户偏好表"""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    preferred_brands = Column(JSON)  # 偏好品牌列表
    preferred_categories = Column(JSON)  # 偏好类别列表
    price_range_min = Column(Float, default=0)
    price_range_max = Column(Float)
    preferred_features = Column(JSON)  # 偏好特性列表
    purchase_history = Column(JSON)  # 购买历史
    browsing_history = Column(JSON)  # 浏览历史
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SearchHistory(Base):
    """搜索历史表"""
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    query = Column(Text, nullable=False)
    search_type = Column(String, default="product")  # product, price, review
    filters = Column(JSON)  # 搜索过滤器
    results_count = Column(Integer, default=0)
    clicked_products = Column(JSON)  # 点击的商品
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class RecommendationLog(Base):
    """推荐日志表"""
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    recommendation_type = Column(String, default="product")  # product, similar, trending
    input_query = Column(Text)
    recommended_products = Column(JSON)  # 推荐的商品列表
    user_feedback = Column(JSON)  # 用户反馈
    conversion_rate = Column(Float, default=0.0)  # 转化率
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class PriceAlert(Base):
    """价格提醒表"""
    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    product_id = Column(String, nullable=False, index=True)
    target_price = Column(Float, nullable=False)
    current_price = Column(Float)
    alert_type = Column(String, default="below_target")  # below_target, percentage_drop
    threshold_percentage = Column(Float)  # 降价百分比阈值
    is_active = Column(Boolean, default=True)
    is_triggered = Column(Boolean, default=False)
    notification_method = Column(String, default="app")  # app, email, sms
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Coupon(Base):
    """优惠券表"""
    __tablename__ = "coupons"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, nullable=False, index=True)  # 平台
    code = Column(String, nullable=False, unique=True, index=True)  # 优惠码
    title = Column(String, nullable=False)  # 标题
    description = Column(Text)  # 描述
    discount_type = Column(String, nullable=False)  # fixed, percentage
    discount_value = Column(Float, nullable=False)  # 折扣值
    min_order_amount = Column(Float, default=0.0)  # 最低消费
    max_discount = Column(Float)  # 最大折扣金额
    start_date = Column(DateTime, nullable=False)  # 开始时间
    end_date = Column(DateTime, nullable=False)  # 结束时间
    usage_limit = Column(Integer, default=0)  # 使用次数限制
    used_count = Column(Integer, default=0)  # 已使用次数
    remaining_count = Column(Integer)  # 剩余次数
    is_active = Column(Boolean, default=True)  # 是否激活
    applicable_categories = Column(JSON)  # 适用类别
    applicable_brands = Column(JSON)  # 适用品牌
    terms_conditions = Column(Text)  # 使用条款
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CouponProduct(Base):
    """优惠券商品关联表"""
    __tablename__ = "coupon_products"

    id = Column(Integer, primary_key=True, index=True)
    coupon_id = Column(Integer, ForeignKey("coupons.id"), nullable=False)
    product_id = Column(String, nullable=False)
    discount_value = Column(Float)  # 特定商品折扣
    created_at = Column(DateTime, default=datetime.utcnow)

class ProductImage(Base):
    """商品图片表"""
    __tablename__ = "product_images"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.product_id"), nullable=False)
    image_url = Column(Text, nullable=False)
    image_type = Column(String, default="main")  # main, detail, review, user_upload
    alt_text = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)
    format = Column(String)  # jpg, png, webp, etc.
    is_primary = Column(Boolean, default=False)
    sort_order = Column(Integer, default=0)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    product = relationship("Product", back_populates="images")

class CouponStrategy(Base):
    """优惠券策略表"""
    __tablename__ = "coupon_strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    strategy_name = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)  # saving, timing, stacking
    description = Column(Text)
    rules = Column(JSON)  # 策略规则
    expected_savings = Column(Float)  # 预期节省
    success_rate = Column(Float, default=0.0)  # 成功率
    usage_count = Column(Integer, default=0)  # 使用次数
    is_active = Column(Boolean, default=True)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)