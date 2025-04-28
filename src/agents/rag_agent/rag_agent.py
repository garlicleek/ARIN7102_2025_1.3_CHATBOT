from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent


class RAGAgentState(MessagesState, total=False):
	"""RAG智能体状态"""
	pass


async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""处理用户请求
	作为RAG系统的统一接口，负责：
	1. 并行调用内部和网络检索
	2. 组合检索结果
	3. 评估结果
	4. 返回最终响应
	"""
	# 并行执行两种检索
	internal_result = await internal_retrieval_agent.ainvoke(state, config)
	web_result = await web_retrieval_agent.ainvoke(state, config)

	# internal_result = {"messages": [AIMessage("测试接口:internal_result")]}
	# web_result = {"messages": [AIMessage("测试接口:web_result")]}

	# 组合结果
	result = {
		"messages": [
			AIMessage(content="测试RAG系统接口:"),
			*internal_result["messages"],
			*web_result["messages"]
		]
	}

	# 评估结果
	evaluated_result = await evaluation_agent.ainvoke(result, config)
	return evaluated_result


# 构建主智能体图
rag_agent = StateGraph(RAGAgentState)

# 添加节点
rag_agent.add_node("process_request", process_request)

# 设置入口点
rag_agent.set_entry_point("process_request")

# 添加边
rag_agent.add_edge("process_request", END)

# 编译图并添加MemorySaver
rag_agent = rag_agent.compile(checkpointer=MemorySaver())
