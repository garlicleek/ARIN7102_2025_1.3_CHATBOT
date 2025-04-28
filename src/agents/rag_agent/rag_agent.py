from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent
from core import get_model, settings

from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import aiohttp


class RAGAgentState(MessagesState, total=False):
	"""RAG智能体状态"""
	question: HumanMessage
	internal_text: str
	web_text: str

	valid_response: bool
	internal_safety: float
	internal_quality: float
	web_safety: float
	web_quality: float
	retry_count: int


def wrap_model(model, instructions) -> RunnableSerializable[RAGAgentState, AIMessage]:
	preprocessor = RunnableLambda(
		lambda state: [SystemMessage(content=instructions)] + state["messages"],
		name="StateModifier",
	)
	return preprocessor | model


async def summarize_responses(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""生成总结"""
	prompt = f"""
	请根据以下两个来源的内容生成简明的总结：
	[内部知识库]
	[网络资源]
	用中文输出3-5句话的总结，保持客观中立。"""

	model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
	model_runnable = wrap_model(model, prompt)
	response = await model_runnable.ainvoke(
		{"messages": [HumanMessage(content=state["internal_text"] + state["web_text"])]},
		config
	)

	# 添加总结内容
	return {
		"messages": [AIMessage(
			content=response.content + f"\n\ntotal score: {state.get('internal_safety', 0) + state.get('internal_quality', 0) + state.get('web_safety', 0) + state.get('web_quality', 0):.2f} (retry count: {state.get('retry_count', 0)})")]
	}


async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	print("Retrieval information...")
	internal_response = await internal_retrieval_agent.ainvoke(state, config)
	internal_text = "\n".join([m.content for m in internal_response["messages"] if isinstance(m, AIMessage)])

	web_response = await web_retrieval_agent.ainvoke(state, config)
	web_text = "\n".join([m.content for m in web_response["messages"] if isinstance(m, AIMessage)])
	print("Retrieval information finished")

	# 更新状态并返回
	return {
		"question": state["messages"][-1],
		"internal_text": internal_text,
		"web_text": web_text,
	}


async def evaluate(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""评估响应质量"""
	print("Evaluate information...")

	internal_eval = await evaluation_agent.ainvoke(
		{"messages": [state["question"], AIMessage(content=state["internal_text"])]},
		config
	)
	web_eval = await evaluation_agent.ainvoke(
		{"messages": [state["question"], AIMessage(content=state["web_text"])]},
		config
	)

	# 完整评分计算
	total_score = (
			internal_eval.get("safety_score", 0)
			+ internal_eval.get("quality_score", 0)
			+ web_eval.get("safety_score", 0)
			+ web_eval.get("quality_score", 0)
	)
	is_valid = total_score > 3
	print("Evaluate information finished")

	# 更新状态
	current_retry = state.get("retry_count", 0)
	new_retry = current_retry + 1 if not is_valid else 0
	return {
		"valid_response": is_valid or new_retry >= 2,
		"retry_count": new_retry,
		"internal_safety": internal_eval.get("safety_score", 0),
		"internal_quality": internal_eval.get("quality_score", 0),
		"web_safety": web_eval.get("safety_score", 0),
		"web_quality": web_eval.get("quality_score", 0)
	}


# 构建状态图
rag_agent = StateGraph(RAGAgentState)
rag_agent.add_node("process_request", process_request)
rag_agent.add_node("evaluate", evaluate)
rag_agent.add_node("summarize", summarize_responses)

rag_agent.set_entry_point("process_request")
rag_agent.add_edge("process_request", "evaluate")
rag_agent.add_edge("evaluate", "summarize")
rag_agent.add_conditional_edges(
	"summarize",
	lambda state: "evaluate" if (
			not state.get("valid_response")
			and state.get("retry_count", 0) < 2
	) else END
)

rag_agent = rag_agent.compile(
	checkpointer=MemorySaver(),
	debug=False
)
