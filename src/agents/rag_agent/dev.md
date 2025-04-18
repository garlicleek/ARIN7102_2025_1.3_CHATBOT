# RAG多智能体系统开发指南

## 系统概述

本系统是一个基于LangGraph的多智能体RAG（检索增强生成）系统，由以下组件组成：

1. **路由智能体（Auth Agent）**
   - 负责请求分发和流程控制
   - 集成安全检查机制（使用LlamaGuard）
   - 根据用户需求选择合适的检索方式

2. **内部检索智能体（Internal Retrieval Agent）**
   - 负责检索内部文档库
   - 使用向量数据库进行相似度搜索
   - 格式化检索结果

3. **网络检索智能体（Web Retrieval Agent）**
   - 负责执行网络搜索
   - 整合网络搜索结果
   - 格式化网络信息

4. **评估智能体（Evaluation Agent）**
   - 评估内容安全性
   - 评估内容质量
   - 生成评估报告

## 文件结构
```
src/
├── agents/          # 智能体相关代码
├── service/         # 服务层代码
├── schema/          # 数据模型定义
├── pages/          # 页面组件
├── memory/         # 记忆管理
├── core/           # 核心功能
├── components/     # 通用组件
├── client/         # 客户端代码
├── streamlit_app.py  # Streamlit应用入口
├── run_agent.py    # 智能体运行入口
├── run_client.py   # 客户端运行入口
└── run_service.py  # 服务运行入口
```
```
agents/
├── __init__.py
├── agents.py       # 智能体注册和管理
├── tools.py        # 通用工具函数
├── llama_guard.py  # 安全检查模块
└── rag_agent/      # RAG多智能体系统
    ├── __init__.py
    ├── rag_agent.py        # 主智能体
    ├── auth_agent.py       # 路由智能体
    ├── internal_retrieval_agent.py  # 内部检索智能体
    ├── web_retrieval_agent.py      # 网络检索智能体
    ├── evaluation_agent.py         # 评估智能体
    └── tools.py                    # RAG专用工具
```
```
rag_agent/
├── __init__.py          # 包初始化文件
├── rag_agent.py         # 主智能体
├── auth_agent.py        # 路由智能体
├── internal_retrieval_agent.py  # 内部检索智能体
├── web_retrieval_agent.py      # 网络检索智能体
└── evaluation_agent.py         # 评估智能体
```

## 开发规范

### 1. 状态定义

每个智能体都需要定义自己的状态类，继承自 `MessagesState`：

```python
class AgentState(MessagesState, total=False):
    """智能体状态定义"""
    # 自定义状态字段
    custom_field: str | None
```

### 2. 节点函数

节点函数需要遵循以下模式：

```python
async def node_function(state: AgentState, config: RunnableConfig) -> AgentState:
    """节点函数说明"""
    # 处理逻辑
    return {"messages": [AIMessage(content="处理结果")]}
```

### 3. 图构建

每个智能体都需要构建自己的图结构：

```python
# 创建图
agent = StateGraph(AgentState)

# 添加节点
agent.add_node("node_name", node_function)

# 设置入口点
agent.set_entry_point("node_name")

# 添加边
agent.add_edge("node_name", END)
```

### 4. 工具集成

使用 `ToolNode` 集成工具：

```python
from langgraph.prebuilt import ToolNode

# 定义工具
tools = [tool1, tool2]

# 添加工具节点
agent.add_node("tools", ToolNode(tools))
```

### 5. 安全检查

集成 LlamaGuard 进行安全检查：

```python
from agents.llama_guard import LlamaGuard

async def check_safety(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}
```

目前本项目未实现Llama_guard,需要自行实现

## 开发流程

1. **创建新智能体**
   - 在 `rag_agent` 目录下创建新文件
   - 定义状态类
   - 实现节点函数
   - 构建图结构

2. **集成到主系统**
   - 在 `auth_agent.py` 中添加路由逻辑
   - 更新 `rag_agent.py` 中的主图结构
   - 集成 LlamaGuard 进行安全检查（重要：不要忘记实现安全检查功能）

3. **部署上线**
   - 更新 `agents.py` 注册新智能体
   - 更新前端配置
   - 进行集成测试

## 安全检查实现说明

### LlamaGuard 集成

LlamaGuard 是本系统的重要组成部分，用于确保内容的安全性。在实现具体功能时，必须集成 LlamaGuard：

1. **依赖安装**
   ```bash
   pip install llama-guard
   ```

2. **配置要求**
   - 需要配置 LlamaGuard 模型路径或 API 密钥
   - 在环境变量中设置必要的配置项

3. **实现位置**
   - 主要在 `auth_agent.py` 中实现安全检查
   - 在 `evaluation_agent.py` 中实现内容评估

4. **安全检查流程**
   ```python
   from agents.llama_guard import LlamaGuard

   async def check_safety(state: AgentState, config: RunnableConfig) -> AgentState:
       llama_guard = LlamaGuard()
       safety_output = await llama_guard.ainvoke("User", state["messages"])
       return {"safety": safety_output}
   ```

5. **注意事项**
   - 确保在开发环境中正确配置 LlamaGuard
   - 在生产环境中使用适当的模型和配置
   - 定期更新安全检查规则
   - 记录安全检查结果以便审计

### 安全检查状态定义

在状态类中需要包含安全检查相关的字段：

```python
class AgentState(MessagesState, total=False):
    """智能体状态定义"""
    safety: LlamaGuardOutput  # 安全检查结果
    # 其他状态字段
```

### 安全检查结果处理

在节点函数中需要处理安全检查结果：

```python
async def process_safety_result(state: AgentState, config: RunnableConfig) -> AgentState:
    """处理安全检查结果"""
    safety = state["safety"]
    if safety.safety_assessment != "SAFE":
        return {"messages": [AIMessage(content="内容安全检查未通过")]}
    return state
```
