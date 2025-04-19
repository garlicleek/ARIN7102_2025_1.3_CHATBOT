from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.llama_guard import LlamaGuard, LlamaGuardOutput


class EvaluationState(MessagesState, total=False):
    """评估智能体状态"""
    safety_score: float | None
    quality_score: float | None


async def evaluate_safety(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
    """评估内容安全性"""
    # 框架阶段，只返回模拟数据
    return {"safety_score": 0.9}


async def evaluate_quality(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
    """评估内容质量"""
    # 框架阶段，只返回模拟数据
    return {"quality_score": 0.8}


async def format_evaluation(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
    """格式化评估结果"""
    safety = state.get("safety_score", 0)
    quality = state.get("quality_score", 0)
    
    content = f"评估结果：\n"
    content += f"安全性评分：{safety:.2f}\n"
    content += f"质量评分：{quality:.2f}\n"
    
    return {"messages": [AIMessage(content=content)]}


# 构建评估智能体图
evaluation_agent = StateGraph(EvaluationState)

# 添加节点
evaluation_agent.add_node("evaluate_safety", evaluate_safety)
evaluation_agent.add_node("evaluate_quality", evaluate_quality)
evaluation_agent.add_node("format_evaluation", format_evaluation)

# 设置入口点
evaluation_agent.set_entry_point("evaluate_safety")

# 添加边
evaluation_agent.add_edge("evaluate_safety", "evaluate_quality")
evaluation_agent.add_edge("evaluate_quality", "format_evaluation")
evaluation_agent.add_edge("format_evaluation", END)

# 编译图并添加MemorySaver
evaluation_agent = evaluation_agent.compile(checkpointer=MemorySaver()) 