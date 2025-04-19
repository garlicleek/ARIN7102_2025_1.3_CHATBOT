from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent


class AuthAgentState(MessagesState, total=False):
    """路由智能体状态"""
    retrieval_type: Literal["internal", "web", "both"] | None


async def check_safety(state: AuthAgentState, config: RunnableConfig) -> AuthAgentState:
    """安全检查"""
    # 框架阶段，直接返回安全状态
    return state


def determine_retrieval_type(state: AuthAgentState) -> Literal["internal", "web", "both"]:
    """确定检索类型"""
    # 框架阶段，默认返回both
    return "both"


async def route_to_retrieval(state: AuthAgentState, config: RunnableConfig) -> AuthAgentState:
    """路由到相应的检索智能体"""
    retrieval_type = determine_retrieval_type(state)
    
    if retrieval_type == "internal":
        result = await internal_retrieval_agent.ainvoke(state, config)
    elif retrieval_type == "web":
        result = await web_retrieval_agent.ainvoke(state, config)
    else:
        # 并行执行两种检索
        internal_result = await internal_retrieval_agent.ainvoke(state, config)
        web_result = await web_retrieval_agent.ainvoke(state, config)
        result = {
            "messages": internal_result["messages"] + web_result["messages"]
        }
    
    return result


async def evaluate_results(state: AuthAgentState, config: RunnableConfig) -> AuthAgentState:
    """评估检索结果"""
    return await evaluation_agent.ainvoke(state, config)


# 构建路由智能体图
auth_agent = StateGraph(AuthAgentState)

# 添加节点
auth_agent.add_node("check_safety", check_safety)
auth_agent.add_node("route_to_retrieval", route_to_retrieval)
auth_agent.add_node("evaluate_results", evaluate_results)

# 设置入口点
auth_agent.set_entry_point("check_safety")

# 添加边
auth_agent.add_edge("check_safety", "route_to_retrieval")
auth_agent.add_edge("route_to_retrieval", "evaluate_results")
auth_agent.add_edge("evaluate_results", END)

# 编译图并添加MemorySaver
auth_agent = auth_agent.compile(checkpointer=MemorySaver()) 