from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph


class InternalRetrievalState(MessagesState, total=False):
    """内部检索智能体状态"""
    retrieved_docs: List[Document] | None


async def retrieve_docs(state: InternalRetrievalState, config: RunnableConfig) -> InternalRetrievalState:
    """检索内部文档"""
    # 框架阶段，只返回模拟数据
    return {"retrieved_docs": [Document(page_content="模拟的内部文档内容")]}


async def format_results(state: InternalRetrievalState, config: RunnableConfig) -> InternalRetrievalState:
    """格式化检索结果"""
    docs = state["retrieved_docs"]
    if not docs:
        return {"messages": [AIMessage(content="未找到相关内部文档")]}
    
    formatted_content = "找到以下相关内部文档：\n\n"
    for i, doc in enumerate(docs, 1):
        formatted_content += f"{i}. {doc.page_content}\n\n"
    
    return {"messages": [AIMessage(content=formatted_content)]}


# 构建内部检索智能体图
internal_retrieval_agent = StateGraph(InternalRetrievalState)

# 添加节点
internal_retrieval_agent.add_node("retrieve_docs", retrieve_docs)
internal_retrieval_agent.add_node("format_results", format_results)

# 设置入口点
internal_retrieval_agent.set_entry_point("retrieve_docs")

# 添加边
internal_retrieval_agent.add_edge("retrieve_docs", "format_results")
internal_retrieval_agent.add_edge("format_results", END)

# 编译图并添加MemorySaver
internal_retrieval_agent = internal_retrieval_agent.compile(checkpointer=MemorySaver()) 