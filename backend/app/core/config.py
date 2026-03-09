from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: str = "bigmodel"  # Options: "openai", "azure", "bigmodel", "deepseek", "moonshot"

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Azure OpenAI Configuration
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2025-01-01-preview"

    # BigModel Configuration
    bigmodel_api_key: Optional[str] = None  # 从环境变量 BIGMODEL_API_KEY 读取
    bigmodel_vlm_api_key: Optional[str] = None  # 从环境变量 BIGMODEL_VLM_API_KEY 读取
    bigmodel_base_url: str = "https://open.bigmodel.cn/api/paas/v4"  # 从环境变量 BIGMODEL_BASE_URL 读取

    # Model Configuration
    text_model: str = "GLM-4.6"  # Default model for BigModel, can be overridden for Azure/OpenAI
    vision_model: str = "glm-4.6v"  # Default vision model for BigModel
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

    # Database Configuration
    database_url: str = "sqlite:///./llm_agent.db"

    # Security Configuration
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # File Upload Configuration
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    # Vector and Memory Configuration
    redis_url: Optional[str] = None
    embedding_model: str = "BAAI/bge-large-zh"
    vector_db_path: str = "./vector_store"
    vector_db_type: Optional[str] = "faiss"

    # Shopping Assistant Configuration
    shopping_platforms: str = "jd,taobao,pdd,xiaohongshu,douyin"
    enable_shopping_assistant: bool = True
    enable_price_monitoring: bool = True
    enable_image_recognition: bool = True
    image_recognition_model: str = "glm-4v"
    similarity_threshold: float = 0.7

    # Price Prediction Configuration
    price_prediction_days: int = 30
    price_prediction_model: str = "ensemble"
    enable_price_alerts: bool = True

    # Risk Analysis Configuration
    risk_analysis_threshold: float = 0.7
    enable_risk_keyword_expansion: bool = True
    risk_categories: str = "quality,logistics,service,price,authenticity"

    # Decision Tool Configuration
    enable_decision_tool: bool = True
    default_weights_price: float = 0.3
    default_weights_quality: float = 0.3
    default_weights_performance: float = 0.2
    default_weights_service: float = 0.1
    default_weights_risk: float = 0.1

    # Third-party API Configuration
    jd_api_key: Optional[str] = None
    jd_api_secret: Optional[str] = None
    taobao_api_key: Optional[str] = None
    taobao_api_secret: Optional[str] = None
    pdd_api_key: Optional[str] = None
    pdd_api_secret: Optional[str] = None
    
    # Onebound API Configuration (万邦API)
    onebound_api_key: Optional[str] = None  # 从环境变量 ONEBOUND_API_KEY 读取
    onebound_api_secret: Optional[str] = None  # 从环境变量 ONEBOUND_API_SECRET 读取
    onebound_api_base_url: str = "https://api-gw.onebound.cn"  # 万邦API基础URL

    # Image Processing Configuration
    image_upload_path: str = "./uploads/images"
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    supported_image_formats: str = "jpg,jpeg,png,webp"

    # File Upload Configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10MB

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5

    # MCP (Model Context Protocol) Configuration
    mcp_enabled: bool = False
    mcp_base_url: Optional[str] = None
    mcp_api_key: Optional[str] = None
    mcp_timeout: int = 30
    mcp_resource_filters: str = "{}"  # JSON string for resource filters
    mcp_content_format: str = "text"
    mcp_max_items: int = 100

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    request_timeout: int = 30  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = 'utf-8'

settings = Settings()

# Create upload directories if they don't exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.image_upload_path, exist_ok=True)