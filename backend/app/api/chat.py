from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
# Form已暂时移除，因为python-multipart安装有问题
# from fastapi import Form
from sqlalchemy.orm import Session
from typing import Optional, List
import os
import time
import logging
from datetime import datetime

from ..models.schemas import (
    ChatRequest, ChatResponse, ConversationResponse,
    MessageResponse, ConversationCreate, FileUploadResponse,
    EnhancedChatRequest, EnhancedChatResponse, RAGSearchResult
)
from ..services.conversation_service import ConversationService
from ..services.media_service import media_service
from ..services.memory_service import MemoryService, get_memory_service
from ..services.rag_service import RAGService, get_rag_service
from ..services.llm_service import LLMService, get_llm_service
from ..core.database import get_db
from ..core.config import settings
from ..graph.graph import run as graph_run
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    处理聊天消息（LangGraph 路由）

    将用户消息路由至 ShoppingAgent StateGraph：
    - search/compare 意图 → 混合检索（BM25 + FAISS + BGE Reranker）+ 生成回复
    - chat 意图         → 直接生成回复（跳过检索）
    """
    try:
        # 用 conversation_id 作为 LangGraph 的 thread_id，保持会话连续性
        thread_id = str(request.conversation_id) if request.conversation_id else f"anon-{int(time.time())}"

        # 调用 LangGraph 状态机
        result = await graph_run(
            query=request.message,
            db=db,
            thread_id=thread_id,
            knowledge_base_ids=None,  # 全库检索；如需限定知识库可在此传入 ID 列表
        )

        final_response = result.get("final_response") or "抱歉，我暂时无法回答这个问题。"
        logger.info(
            f"[/chat] intent={result.get('intent')} "
            f"confidence={result.get('confidence_score', 0):.2f} "
            f"docs={len(result.get('retrieved_docs', []))}"
        )

        return ChatResponse(
            response=final_response,
            conversation_id=request.conversation_id or 0,
            message_id=0,
            model_used=settings.text_model,
        )
    except Exception as e:
        logger.exception("[/chat] LangGraph 执行异常")
        raise HTTPException(status_code=500, detail=str(e))

# 暂时注释掉文件上传路由，因为FastAPI需要python-multipart来处理File参数
# 当python-multipart正确安装后，可以取消注释
# @router.post("/chat/upload", response_model=ChatResponse)
# async def chat_with_upload(
#     request: ChatRequest,
#     file: Optional[UploadFile] = File(None),
#     db: Session = Depends(get_db)
# ):
#     """
#     处理带文件上传的聊天消息（使用JSON body替代Form）
#     """
#     try:
#         file_path = None
#         if file:
#             # 保存文件
#             file_path = await media_service.save_upload_file(file, "chat")
# 
#             # 确定文件类型
#             file_type = "image" if file.content_type and file.content_type.startswith("image/") else "audio"
# 
#             # 处理文件
#             if file_type == "image":
#                 processed_path, _ = await media_service.process_image(file_path)
#                 request.media_url = processed_path
#                 request.message_type = file_type
#         else:
#             file_type = request.message_type or "text"
# 
#         conversation_service = ConversationService(db)
#         response = await conversation_service.process_chat_message(request)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    user_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    获取对话列表
    """
    conversation_service = ConversationService(db)
    conversations = conversation_service.get_user_conversations(user_id, limit)
    return conversations

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    获取对话详情
    """
    conversation_service = ConversationService(db)
    conversation = conversation_service.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    db: Session = Depends(get_db)
):
    """
    创建新对话
    """
    conversation_service = ConversationService(db)
    conversation = conversation_service.create_conversation(conversation_data)
    return conversation

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    删除对话
    """
    conversation_service = ConversationService(db)
    success = conversation_service.delete_conversation(conversation_id)

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted successfully"}

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    获取对话的消息历史
    """
    conversation_service = ConversationService(db)
    messages = conversation_service.get_conversation_messages(conversation_id)
    return messages

# 暂时注释掉File上传路由，因为FastAPI需要python-multipart来处理File参数
# @router.post("/upload", response_model=FileUploadResponse)
# async def upload_file(
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
    """
    上传文件
    """
    try:
        # 验证文件类型
        allowed_types = ['jpg', 'jpeg', 'png', 'gif', 'mp3', 'wav', 'mp4', 'mov']
        if not media_service.validate_file_type(file.filename, allowed_types):
            raise HTTPException(status_code=400, detail="File type not allowed")

        # 验证文件大小
        file_size = 0
        if hasattr(file, 'size'):
            file_size = file.size
        if not media_service.validate_file_size(file_size):
            raise HTTPException(status_code=400, detail="File too large")

        # 保存文件
        file_path = await media_service.save_upload_file(file)
        file_info = media_service.get_file_info(file_path)

        return FileUploadResponse(
            id=1,  # 这里应该保存到数据库并返回真实ID
            filename=os.path.basename(file_path),
            original_name=file.filename,
            file_type=file_info.get('extension', ''),
            file_size=file_info.get('size', 0),
            file_url=file_path,
            uploaded_at=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{filename}")
async def get_file(filename: str):
    """
    获取文件
    """
    file_path = os.path.join(settings.upload_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

@router.post("/enhanced", response_model=EnhancedChatResponse)
async def enhanced_chat(
    request: EnhancedChatRequest,
    db: Session = Depends(get_db)
):
    """
    增强聊天接口（LangGraph 路由）

    在基础 /chat 之上额外返回：检索文档列表、置信度、意图等调试字段。
    knowledge_base_ids 传入时限定 RAG 检索范围。
    """
    start_time = time.time()

    try:
        thread_id = str(request.conversation_id) if request.conversation_id else f"anon-{int(time.time())}"

        # 仅在 use_rag=True 时透传 knowledge_base_ids
        kb_ids = request.knowledge_base_ids if request.use_rag else None

        result = await graph_run(
            query=request.message,
            db=db,
            thread_id=thread_id,
            knowledge_base_ids=kb_ids,
        )

        final_response = result.get("final_response") or "抱歉，我暂时无法回答这个问题。"
        retrieved_docs  = result.get("retrieved_docs", [])
        confidence      = result.get("confidence_score", 0.0)
        intent          = result.get("intent", "chat")

        logger.info(
            f"[/chat/enhanced] intent={intent} confidence={confidence:.2f} "
            f"docs={len(retrieved_docs)}"
        )

        processing_time = time.time() - start_time

        # 将检索文档映射为 RAGSearchResult 格式
        rag_results_out = [
            RAGSearchResult(
                content=doc.get("content", ""),
                document_id=doc.get("document_id", 0),
                chunk_index=doc.get("chunk_index", 0),
                score=float(doc.get("rerank_score") or doc.get("retrieval_score") or 0.0),
                metadata={"document_name": doc.get("document_name", ""),
                          "retrieval_source": doc.get("retrieval_source", "")},
            )
            for doc in retrieved_docs
        ]

        return EnhancedChatResponse(
            response=final_response,
            conversation_id=request.conversation_id or 0,
            message_id=0,
            model_used=settings.text_model,
            tokens_used=None,
            memory_used=len(retrieved_docs) > 0,
            rag_results=rag_results_out,
            agent_collaboration={"intent": intent, "confidence": confidence},
            processing_time=processing_time,
        )

    except Exception as e:
        logger.exception("[/chat/enhanced] LangGraph 执行异常")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-memory")
async def extract_conversation_memory(
    conversation_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    从对话中提取重要信息到记忆系统
    """
    try:
        memory_service = get_memory_service(db)
        await memory_service.extract_and_store_conversation_memory(conversation_id, user_id)
        return {"message": "Memory extraction completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))