import openai
from typing import List, Dict, Any, Optional
from ..core.config import settings
import json
import asyncio

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import zhipuai
    BIGMODEL_AVAILABLE = True
except ImportError:
    BIGMODEL_AVAILABLE = False

class LLMService:
    def __init__(self):
        self.llm_provider = settings.llm_provider

        # 设置默认模型
        self.text_model = getattr(settings, 'text_model', 'glm-4.7-flash')
        self.vision_model = getattr(settings, 'vision_model', 'glm-4.6v-flash')
        self.max_tokens = getattr(settings, 'max_tokens', 4096)
        self.temperature = getattr(settings, 'temperature', 0.7)
        self.top_p = getattr(settings, 'top_p', 0.9)

        if self.llm_provider == "azure" and AZURE_AVAILABLE:
            # 初始化Azure OpenAI客户端
            if settings.azure_endpoint and settings.azure_api_key:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=settings.azure_endpoint,
                    api_key=settings.azure_api_key,
                    api_version=settings.azure_api_version,
                )
            else:
                raise Exception("Azure OpenAI configuration incomplete: missing endpoint or API key")
        elif self.llm_provider == "bigmodel":
            if not BIGMODEL_AVAILABLE:
                raise Exception("zhipuai package not installed. Please run: pip install zhipuai")
            # 初始化BigModel客户端
            if not settings.bigmodel_api_key:
                raise Exception("BigModel API key not configured. Please set BIGMODEL_API_KEY in .env file")
            self.bigmodel_client = zhipuai.ZhipuAI(
                api_key=settings.bigmodel_api_key,
                base_url=getattr(settings, 'bigmodel_base_url', 'https://open.bigmodel.cn/api/paas/v4')
            )
            # VLM API key可以使用相同的key，如果没有单独配置
            vlm_api_key = settings.bigmodel_vlm_api_key or settings.bigmodel_api_key
            self.bigmodel_vlm_client = zhipuai.ZhipuAI(
                api_key=vlm_api_key,
                base_url=getattr(settings, 'bigmodel_base_url', 'https://open.bigmodel.cn/api/paas/v4')
            )
        elif self.llm_provider == "openai" or self.llm_provider == "azure":
            # 初始化OpenAI或Azure客户端
            if self.llm_provider == "azure" and AZURE_AVAILABLE:
                if settings.azure_endpoint and settings.azure_api_key:
                    self.azure_client = AzureOpenAI(
                        azure_endpoint=settings.azure_endpoint,
                        api_key=settings.azure_api_key,
                        api_version=settings.azure_api_version,
                    )
                else:
                    raise Exception("Azure OpenAI configuration incomplete: missing endpoint or API key")
            else:
                if not settings.openai_api_key:
                    raise Exception("OpenAI API key not configured")
                self.client = openai.OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url
                )
        else:
            raise Exception(f"Unsupported LLM provider: {self.llm_provider}. Supported providers: openai, azure, bigmodel")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        进行文本聊天完成
        """
        try:
            # 使用默认值如果未提供
            actual_model = model or self.text_model
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature or self.temperature

            if self.llm_provider == "azure" and AZURE_AVAILABLE:
                return await self._azure_chat_completion(
                    messages, actual_model, actual_max_tokens, actual_temperature, stream
                )
            elif self.llm_provider == "bigmodel":
                if not BIGMODEL_AVAILABLE:
                    raise Exception("zhipuai package not installed. Please run: pip install zhipuai")
                return await self._bigmodel_chat_completion(
                    messages, actual_model, actual_max_tokens, actual_temperature, stream
                )
            else:
                raise Exception(f"Unsupported LLM provider: {self.llm_provider}. Chat completion requires bigmodel provider.")

        except Exception as e:
            raise Exception(f"LLM API error: {str(e)}")

    async def _azure_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> Dict[str, Any]:
        """Azure OpenAI聊天完成"""
        try:
            # 使用部署名称作为模型名称
            deployment_name = getattr(settings, 'azure_deployment', 'gpt-4o')

            response = self.azure_client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

            if stream:
                return response

            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model_used": f"azure/{deployment_name}",
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            }

            return result

        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")

    async def _openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> Dict[str, Any]:
        """OpenAI聊天完成"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

            if stream:
                return response

            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model_used": model,
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            }

            return result

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    async def _bigmodel_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> Dict[str, Any]:
        """BigModel聊天完成"""
        try:
            # 如果模型名称是OpenAI的格式，转换为BigModel格式
            if model.startswith("gpt-"):
                model = self.text_model  # 使用配置的默认GLM模型

            # 转换消息格式
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # 调用BigModel API
            response = self.bigmodel_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=stream
            )

            if stream:
                return response

            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model_used": model,
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            }

            return result

        except Exception as e:
            # 更详细的错误信息
            error_str = str(e)
            # 如果错误消息中包含OpenAI相关的内容，但实际上是BigModel的错误，需要更正
            if "OpenAI" in error_str or "openai" in error_str.lower():
                # 这可能是zhipuai库内部使用了OpenAI的格式，但实际上是BigModel的错误
                error_msg = f"BigModel API error: {error_str}"
                # 替换OpenAI为BigModel
                error_msg = error_msg.replace("OpenAI API error", "BigModel API error")
                error_msg = error_msg.replace("OpenAI", "BigModel")
            else:
                error_msg = f"BigModel API error: {error_str}"
            
            if "api_key" in error_str.lower() or "401" in error_str or "unauthorized" in error_str.lower():
                error_msg += " - Please check your BigModel API key in .env file (BIGMODEL_API_KEY)"
            elif "model" in error_str.lower():
                error_msg += f" - Model '{model}' may not be available for BigModel"
            
            raise Exception(error_msg)

    async def analyze_image(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析图像内容
        """
        try:
            # 使用默认视觉模型如果未提供
            actual_model = model or self.vision_model

            if self.llm_provider == "azure" and AZURE_AVAILABLE:
                return await self._azure_analyze_image(image_url, prompt, actual_model)
            elif self.llm_provider == "bigmodel" and BIGMODEL_AVAILABLE:
                return await self._bigmodel_analyze_image(image_url, prompt, actual_model)
            else:
                return await self._openai_analyze_image(image_url, prompt, actual_model)

        except Exception as e:
            raise Exception(f"Image analysis error: {str(e)}")

    async def _azure_analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """Azure OpenAI图像分析"""
        try:
            # 使用部署名称作为模型名称
            deployment_name = getattr(settings, 'azure_deployment', 'gpt-4o')

            response = self.azure_client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            return {
                "analysis": response.choices[0].message.content,
                "model_used": f"azure/{deployment_name}",
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            raise Exception(f"Azure OpenAI image analysis error: {str(e)}")

    async def _openai_analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """OpenAI图像分析"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            return {
                "analysis": response.choices[0].message.content,
                "model_used": model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            raise Exception(f"OpenAI image analysis error: {str(e)}")

    async def _bigmodel_analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """BigModel图像分析"""
        try:
            # 如果模型名称是OpenAI的格式，转换为BigModel格式
            if model.startswith("gpt-4-vision"):
                model = self.vision_model  # 使用配置的默认GLM-4.5V模型

            # 如果模型名称是OpenAI的视觉模型格式，转换为BigModel格式
            if model.startswith("gpt-4-vision") or model.startswith("gpt-4o"):
                model = self.vision_model

            # 转换消息格式为BigModel支持的格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]

            # 使用VLM专用客户端
            response = self.bigmodel_vlm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,  # GLM-4.5V支持更多tokens
                temperature=self.temperature
            )

            return {
                "analysis": response.choices[0].message.content,
                "model_used": model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            # 更详细的错误信息
            error_msg = f"BigModel image analysis error: {str(e)}"
            if "api_key" in str(e).lower():
                error_msg += " - Please check your BigModel VLM API key"
            elif "model" in str(e).lower():
                error_msg += f" - Vision model '{model}' may not be available"
            elif "image_url" in str(e).lower():
                error_msg += " - Image URL may be invalid or inaccessible"
            raise Exception(error_msg)

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转录音频文件
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language
                )

            return {
                "text": response.text,
                "language": language or "detected"
            }

        except Exception as e:
            raise Exception(f"Audio transcription error: {str(e)}")

    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1"
    ) -> str:
        """
        文本转语音
        """
        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )

            # 保存音频文件
            output_file = f"tts_output_{hash(text)}.mp3"
            response.stream_to_file(output_file)

            return output_file

        except Exception as e:
            raise Exception(f"Text-to-speech error: {str(e)}")

    async def generate_embeddings(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """
        生成文本嵌入向量
        """
        try:
            if self.llm_provider == "azure" and AZURE_AVAILABLE:
                return await self._azure_generate_embeddings(text, model)
            elif self.llm_provider == "bigmodel" and BIGMODEL_AVAILABLE:
                return await self._bigmodel_generate_embeddings(text, model)
            else:
                return await self._openai_generate_embeddings(text, model)

        except Exception as e:
            raise Exception(f"Embedding generation error: {str(e)}")

    async def _azure_generate_embeddings(
        self,
        text: str,
        model: str
    ) -> List[float]:
        """Azure OpenAI嵌入生成"""
        try:
            # 使用部署名称作为模型名称
            deployment_name = getattr(settings, 'azure_deployment', 'gpt-4o')

            response = self.azure_client.embeddings.create(
                model=deployment_name,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"Azure OpenAI embedding error: {str(e)}")

    async def _openai_generate_embeddings(
        self,
        text: str,
        model: str
    ) -> List[float]:
        """OpenAI嵌入生成"""
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")

    async def _bigmodel_generate_embeddings(
        self,
        text: str,
        model: str
    ) -> List[float]:
        """BigModel嵌入生成"""
        try:
            # 转换模型名称
            if model.startswith("text-embedding-"):
                model = "embedding-2"  # 使用BigModel的嵌入模型

            response = self.bigmodel_client.embeddings.create(
                model=model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"BigModel embedding error: {str(e)}")

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应 - 简化接口"""
        try:
            response = await self.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.get("content", "")
        except Exception as e:
            logger.error(f"LLM response generation error: {str(e)}")
            return ""

# 全局LLM服务实例（延迟初始化）
_llm_service_instance = None

def get_llm_service(db=None) -> LLMService:
    """获取LLM服务实例 - 每次调用都重新创建以确保使用最新配置"""
    global _llm_service_instance
    try:
        # 每次都重新创建实例以确保使用最新的配置
        # 这在开发环境中特别重要，因为配置可能会改变
        _llm_service_instance = None  # 清除旧实例
        
        # 创建新实例
        new_instance = LLMService()
        
        # 验证初始化是否正确
        if settings.llm_provider == "bigmodel":
            if not hasattr(new_instance, 'bigmodel_client'):
                raise Exception("BigModel client not initialized. Please check BIGMODEL_API_KEY configuration.")
            if new_instance.bigmodel_client is None:
                raise Exception("BigModel client is None. Please check BIGMODEL_API_KEY configuration.")
        elif settings.llm_provider == "openai":
            if not hasattr(new_instance, 'client'):
                raise Exception("OpenAI client not initialized. Please check OPENAI_API_KEY configuration.")
        
        _llm_service_instance = new_instance
        return _llm_service_instance
        
    except Exception as e:
        # 如果初始化失败，清除实例并抛出更友好的错误
        _llm_service_instance = None
        error_msg = str(e)
        if "zhipuai" in error_msg.lower() or "not installed" in error_msg.lower():
            raise Exception(f"Failed to initialize LLM service: {error_msg}. Please run: pip install zhipuai")
        elif "api key" in error_msg.lower() or "API key" in error_msg:
            raise Exception(f"Failed to initialize LLM service: {error_msg}. Please check your API key in .env file.")
        else:
            raise Exception(f"Failed to initialize LLM service: {error_msg}. Please check your configuration in .env file.")