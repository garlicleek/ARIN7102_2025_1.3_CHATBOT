from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph


class WebRetrievalState(MessagesState, total=False):
    """网络检索智能体状态"""
    search_results: str | None


async def search_web(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
    """执行网络搜索"""
    # 框架阶段，只返回模拟数据
    return {"search_results": "模拟的网络搜索结果"}


async def format_web_results(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
    """格式化网络搜索结果"""
    results = state["search_results"]
    if not results:
        return {"messages": [AIMessage(content="未找到相关网络信息")]}
    
    formatted_content = "网络搜索结果：\n\n"
    formatted_content += results
    
    return {"messages": [AIMessage(content=formatted_content)]}


# 构建网络检索智能体图
web_retrieval_agent = StateGraph(WebRetrievalState)

# 添加节点
web_retrieval_agent.add_node("search_web", search_web)
web_retrieval_agent.add_node("format_web_results", format_web_results)

# 设置入口点
web_retrieval_agent.set_entry_point("search_web")

# 添加边
web_retrieval_agent.add_edge("search_web", "format_web_results")
web_retrieval_agent.add_edge("format_web_results", END)

# 编译图并添加MemorySaver
web_retrieval_agent = web_retrieval_agent.compile(checkpointer=MemorySaver()) 