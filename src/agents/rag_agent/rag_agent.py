
from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent
from typing import Optional
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph


class RAGAgentState(MessagesState, total=False):
	"""增强的RAG智能体状态"""
	valid_response: bool  # 验证状态
	internal_safety: float  # 内部检索安全分
	internal_quality: float  # 内部检索质量分
	web_safety: float  # 网络检索安全分
	web_quality: float  # 网络检索质量分


async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""改进的请求处理流程"""
	# 清空前次无效响应
	if not state.get("valid_response", True):
		state["messages"] = []

	# 并行获取原始响应
	internal_response = await internal_retrieval_agent.ainvoke(state, config)
	web_response = await web_retrieval_agent.ainvoke(state, config)

	# 分离评估各来源
	internal_eval = await evaluation_agent.ainvoke(
		{"messages": internal_response["messages"]},
		config
	)
	web_eval = await evaluation_agent.ainvoke(
		{"messages": web_response["messages"]},
		config
	)

	# 计算综合评分
	total_score = (
			internal_eval.get("safety_score", 0)
			+ internal_eval.get("quality_score", 0)
			+ web_eval.get("safety_score", 0)
			+ web_eval.get("quality_score", 0)
	)

	# 构建新状态
	return {
		"valid_response": total_score > 3,
		"internal_safety": internal_eval.get("safety_score", 0),
		"internal_quality": internal_eval.get("quality_score", 0),
		"web_safety": web_eval.get("safety_score", 0),
		"web_quality": web_eval.get("quality_score", 0),
		"messages": [
			*internal_response["messages"],
			*web_response["messages"],
			AIMessage(content=f"当前总分: {total_score:.2f}")
		]
	}


# 构建带循环的状态图
rag_agent = StateGraph(RAGAgentState)
rag_agent.add_node("process_request", process_request)
rag_agent.set_entry_point("process_request")

# 添加条件循环逻辑
rag_agent.add_conditional_edges(
	"process_request",
	lambda state: "process_request" if not state.get("valid_response") else END
)

# 最终编译
rag_agent = rag_agent.compile(
	checkpointer=MemorySaver(),
	# 启用调试追踪
	debug=True
)