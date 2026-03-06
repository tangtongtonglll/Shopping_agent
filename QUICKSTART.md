# 本地快速启动指南

> 适用于在本机完整运行 Shopping Agent 项目（后端 + 前端）

---

## 一、环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端构建 |
| npm | 9+ | 随 Node.js 附带 |
| Git | 任意 | 拉取代码 |

BGE 模型（`bge-large-zh` 和 `bge-reranker-v2-m3`）会在首次运行时由 HuggingFace 自动下载，
约 1.5 GB，请确保磁盘空间充足、网络可访问 HuggingFace（或配置镜像）。

---

## 二、克隆代码

```bash
git clone https://github.com/tangtongtonglll/Shopping_agent.git
cd Shopping_agent
git checkout feat/arch-refactor-phase1   # 使用重构后的分支
```

---

## 三、后端配置

### 3.1 创建 Python 虚拟环境

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3.2 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **注意**：`FlagEmbedding` 会拉取 PyTorch 依赖，安装较慢，耐心等待。
> 如需加速，可先单独安装 PyTorch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 3.3 配置环境变量

在 `backend/` 目录下创建 `.env` 文件（从示例复制后修改）：

```bash
cp .env.example .env   # 如果没有 example 文件，手动创建
```

`.env` 最小配置（选择你持有的 LLM 服务商之一填写）：

```dotenv
# ── LLM 服务商选择（必填其中一个）──────────────────────────────────────

# 选项 A：智谱 BigModel（GLM-4，默认）
LLM_PROVIDER=bigmodel
BIGMODEL_API_KEY=your_bigmodel_api_key_here

# 选项 B：OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
# OPENAI_BASE_URL=https://api.openai.com/v1   # 中转地址可在此修改

# 选项 C：DeepSeek
# LLM_PROVIDER=deepseek
# OPENAI_API_KEY=your_deepseek_api_key
# OPENAI_BASE_URL=https://api.deepseek.com/v1

# ── 数据库（默认 SQLite，无需修改）───────────────────────────────────────
DATABASE_URL=sqlite:///./llm_agent.db

# ── 安全密钥（本地开发可保持默认）────────────────────────────────────────
SECRET_KEY=local-dev-secret-key-change-in-production

# ── 向量库路径（默认即可）───────────────────────────────────────────────
VECTOR_DB_PATH=./vector_store
EMBEDDING_MODEL=BAAI/bge-large-zh

# ── HuggingFace 镜像（国内网络必填）────────────────────────────────────
# HF_ENDPOINT=https://hf-mirror.com
```

> **API Key 获取地址**
> - 智谱 BigModel：https://open.bigmodel.cn
> - OpenAI：https://platform.openai.com
> - DeepSeek：https://platform.deepseek.com

### 3.4 初始化数据库

```bash
# 在 backend/ 目录下执行
python -c "
from app.core.database import engine
from app.models.models import Base
from app.models.ecommerce_models import Base as EBase
Base.metadata.create_all(bind=engine)
EBase.metadata.create_all(bind=engine)
print('数据库初始化完成')
"
```

### 3.5 启动后端

```bash
# 在 backend/ 目录下
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 四、验证 LangGraph 图流转

在**项目根目录**（非 backend/）运行测试脚本，验证重构后的 LangGraph 图是否正常工作：

```bash
# 确保在项目根目录
cd /path/to/Shopping_agent

# 如果上面用了 venv，需要激活
source backend/venv/bin/activate

# 安装测试所需的最小依赖（若上面已全量安装 requirements.txt 则跳过）
pip install langchain-core langgraph numpy rank-bm25 sqlalchemy openai

# 运行测试
python test_graph.py
```

预期输出（5 项全部通过）：

```
████████████████████████████████████████████████████████████
  ShoppingAgent LangGraph 图流转测试
████████████████████████████████████████████████████████████

测试 1：图编译与节点注册
  ✓ 节点已注册: comparator_node
  ✓ 节点已注册: generator_node
  ✓ 节点已注册: human_review_node
  ✓ 节点已注册: retriever_node
  ✓ 节点已注册: router_node
✅ 测试 1 通过：图编译正常，所有节点注册完毕

测试 2：搜索意图流转
  意图识别: search  检索文档数: 2  置信度: 0.9345
✅ 测试 2 通过：搜索流程节点顺序与输出正确

测试 3：对比意图流转
  意图识别: compare  对比维度数: 2
✅ 测试 3 通过：对比流程节点顺序与输出正确

测试 4：闲聊意图流转
  意图识别: chat  检索文档数: 0  置信度: 0.9000
✅ 测试 4 通过：闲聊流程正确跳过检索节点

测试 5：SqliteSaver Checkpointer
✅ 测试 5 通过（或跳过）

共 5 项，通过 5 项，失败 0 项
```

> 测试使用全 Mock，**不需要真实 API Key**，不会下载 BGE 模型。

---

## 五、前端配置与启动

```bash
cd frontend
npm install
```

创建 `frontend/.env`（如果不存在）：

```dotenv
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

启动前端开发服务器：

```bash
npm start
```

浏览器自动打开 http://localhost:3000

---

## 六、完整启动顺序（两个终端）

**终端 1 — 后端：**

```bash
cd Shopping_agent/backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**终端 2 — 前端：**

```bash
cd Shopping_agent/frontend
npm start
```

---

## 七、HuggingFace 模型下载说明

首次使用向量检索或重排序功能时，系统会自动下载两个 BGE 模型：

| 模型 | 用途 | 大小 |
|------|------|------|
| `BAAI/bge-large-zh` | 文本向量化（1024 维） | ~1.3 GB |
| `BAAI/bge-reranker-v2-m3` | Cross-Encoder 重排序 | ~1.1 GB |

模型缓存位置：`~/.cache/huggingface/hub/`

**国内网络加速（二选一）**：

方案 A — 环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

方案 B — 提前手动下载：
```bash
pip install huggingface_hub
huggingface-cli download BAAI/bge-large-zh --local-dir ./models/bge-large-zh
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3
```

然后修改 `backend/.env`：
```dotenv
EMBEDDING_MODEL=./models/bge-large-zh
```

---

## 八、Docker 一键启动（可选）

如果不想手动配置环境，可使用 Docker Compose：

```bash
# 在项目根目录
cp .env.example .env       # 填写 API Key
docker-compose up --build
```

服务启动后：
- 后端：http://localhost:8000
- 前端：http://localhost:3000

---

## 九、常见问题

### Q：启动后端时报 `ModuleNotFoundError: No module named 'app'`

确认是在 `backend/` 目录下执行 `uvicorn`，不要在项目根目录执行。

### Q：`FlagEmbedding` 安装失败

尝试：
```bash
pip install FlagEmbedding --no-deps
pip install torch transformers
```

### Q：模型下载超时

设置镜像后重试：
```bash
HF_ENDPOINT=https://hf-mirror.com pip install FlagEmbedding
```

### Q：运行 `test_graph.py` 报 `ModuleNotFoundError: No module named 'langchain_core'`

需要先安装核心依赖：
```bash
pip install langchain-core langgraph
```

### Q：前端报 `CORS error` 或无法连接后端

检查后端是否在运行（访问 http://localhost:8000/health），
并确认 `frontend/.env` 中 `REACT_APP_API_URL=http://localhost:8000`。

### Q：想用 OpenAI 兼容的国内中转 API

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-proxy.com/v1
```

---

## 十、项目结构速览

```
Shopping_agent/
├── backend/                  # FastAPI 后端
│   ├── requirements.txt      # Python 依赖
│   ├── .env                  # 环境变量（需自行创建）
│   └── app/
│       ├── main.py           # FastAPI 入口
│       ├── core/config.py    # 配置（读取 .env）
│       ├── graph/            # LangGraph 图（Phase 3 & 4）
│       │   ├── state.py      # ShoppingAgentState
│       │   ├── nodes.py      # 5 个节点函数
│       │   └── graph.py      # StateGraph 装配
│       └── services/
│           ├── vector_service.py        # FAISS + BGE 向量服务
│           └── hybrid_search_service.py # 混合检索（Phase 2）
│
├── frontend/                 # React 前端
│   ├── package.json
│   └── src/
│
├── test_graph.py             # LangGraph 图流转测试（根目录）
├── ARCHITECTURE_REDESIGN.md  # 架构设计文档
└── docker-compose.yml        # Docker 一键启动
```
