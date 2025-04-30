import re
from typing import List, Dict, Any
from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent
from core import get_model, settings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json


class RAGAgentState(MessagesState, total=False):
	"""RAG智能体状态"""
	question: HumanMessage
	enhanced_question: str

	retrieved_docs: List[Dict[str, Any]] | None
	docs_text: str

	valid_response: bool
	avg_item_score: float
	web_safety: float
	web_quality: float
	retry_count: int


def wrap_model(model, instructions) -> RunnableSerializable[RAGAgentState, AIMessage]:
	preprocessor = RunnableLambda(
		lambda state: [SystemMessage(content=instructions)] + state["messages"],
		name="StateModifier",
	)
	return preprocessor | model


summary_prompt = f"""
You are a professional consultant. 
Please follow these steps to analyze and summarize the content, then give the answer for users:

1. Content Analysis
   - Carefully review the content from both internal knowledge base and web resources
   - Identify key information and main points
   - Evaluate information reliability and relevance

2. Information Integration
   - Compare information from different sources
   - Identify connections and differences between information
   - Eliminate duplicate content
   - Handle potential conflicting information

3. Structured Thinking
   - Determine the main framework for the summary
   - Prioritize information based on importance
   - Ensure logical coherence

4. Summary Generation
   - Use clear and concise language
   - Maintain an objective and neutral stance
   - Highlight key information and core viewpoints
   - Ensure completeness and accuracy of the summary

5. Oral language
   - Use oral language in case your customer dont understand
   - still keep precise summary

Please output in the following format:
[Analysis Process]
1. Key Information Identification:
   - Internal Knowledge Base: <list key points>
   - Web Resources: <list key points>

2. Information Integration Results:
   - Common Points: <list common points>
   - Differences: <list differences>

[Final Summary]
Provide a comprehensive summary that:
- Uses clear and concise language
- Emphasizes key points
- Maintains logical flow
- Stays objective and neutral
- Covers all important aspects of the content

[Quality Assessment]
- Information Completeness: <score>
- Logical Coherence: <score>
- Language Standardization: <score>

Please begin your analysis and summarization. 
(important) please do not display your [Analysis Process] an [Quality Assessment], return final summary only"""


async def summarize_responses(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""生成总结"""
	model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
	model_runnable = wrap_model(model, summary_prompt)
	response = await model_runnable.ainvoke(
		{"messages": [HumanMessage(content=state["docs_text"])]},
		config
	)

	# 添加总结内容
	return {
		"messages": [AIMessage(
			content=response.content + f"\n\ntotal score: {state.get('avg_item_score', 0):.2f} (retry count: {state.get('retry_count', 0)})")]
	}


semantic_enhancement_prompt = """
You are a pharmaceutical data specialist with expertise in healthcare and financial domains. 
Enhance user questions while strictly preserving technical terms and optimizing for medication-related data retrieval. Follow this protocol:

1. **Terminology Protection**
   - Tag identified terms:
     - Medical: <medical>{term}</medical>
     - Financial: <finance>{term}</finance>
   - Create RAG mapping:
     Example: 
     Metformin → <medical>Metformin HCl</medical> (CAS:1115-70-4)

3. **Intent Inference**
   - For non-medical questions containing:
     - Drug names → Add therapeutic context 
       (e.g., "side effects of <medical>Metformin</medical>" → add "in Type 2 Diabetes management")
     - Financial terms → Link to medical entities
       (e.g., "reimbursement rate" → associate with <medical>PCI stent</medical> procedures)
   - For ambiguous queries → Activate hypothesis engine to generate a guess in pharmaceutical domain:
		(e.g., original:"cost trend" -> enhanced:"cost trend in <guess>medicine for Type 2 Diabetes</guess>")

Please return the result in the following JSON format:
{
    "original_question": "Original question",
    "enhanced_question": "The question after augment, include tag <medical> and <finance>",
}
IMPORTANT: You must return ONLY a JSON object, without any markdown formatting or additional text.
"""


# TODO 生成过程过长, 用户体验差, 可以考虑改成deepseekR1一样, 将思考过程流式输出
async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	print("Enhancing question semantics...")
	# 获取原始问题
	original_question = state["messages"][-1].content

	# 使用LLM进行语义增强
	model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL)).with_config(tags=["skip_stream"])
	model_runnable = wrap_model(model, semantic_enhancement_prompt)
	enhanced_response = await model_runnable.ainvoke(
		{"messages": [HumanMessage(content=original_question)]},
		config
	)

	# 解析JSON响应
	json_pattern = r'\{[^{}]*\}'
	match = re.search(json_pattern, enhanced_response.content)
	if not match:
		enhanced_question = enhanced_response.content
	enhanced_question = json.loads(match.group(0))["enhanced_question"]
	print(enhanced_question)

	print("Retrieval information...")
	internal_response = await internal_retrieval_agent.ainvoke(
		{"messages": [HumanMessage(content=enhanced_question)]},
		config
	)
	internal_results = internal_response.get("retrieved_docs", [])
	print("\n".join([json.dumps(result, ensure_ascii=False) for result in internal_results]))

	web_response = await web_retrieval_agent.ainvoke(
		{"messages": [HumanMessage(content=enhanced_question)]},
		config
	)
	web_results = web_response.get("retrieved_docs", [])
	print("\n".join([json.dumps(result, ensure_ascii=False) for result in web_results]))

	# 合并检索结果
	all_results = []
	if internal_results:
		all_results.extend(internal_results)
	if web_results:
		all_results.extend(web_results)

	print("Retrieval information finished")

	return {
		"question": state["messages"][-1],
		"enhanced_question": enhanced_question,
		"retrieved_docs": all_results,
		"docs_text": json.dumps(all_results, ensure_ascii=False)
	}


# TODO 低质量评估结果的内容除了删除, 也应该让LLM做对应的调整, 第二次以上的evaluate应该调整策略
async def evaluate(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
	"""评估响应质量"""
	print("Evaluate information...")

	evaluation_response = await evaluation_agent.ainvoke(
		{"messages": [HumanMessage(content=state["enhanced_question"]), AIMessage(content=state["docs_text"])]},
		config
	)
	safety_score = evaluation_response.get("safety_score", 1.0)
	quality_scores = evaluation_response.get("quality_score", [])

	# 更新retrieved_docs中的confidence
	if state.get("retrieved_docs") and quality_scores:
		updated_docs = []
		for doc in state["retrieved_docs"]:
			# 在quality_scores中查找匹配的id和source
			matching_score = next(
				(score for score in quality_scores
				 if score.get("id") == doc.get("id")
				 and score.get("source") == doc.get("source")),
				None
			)
			if matching_score:
				# 更新所有评分相关字段
				doc["confidence"] = matching_score["final_confidence"]
				doc["relevance"] = matching_score["relevance"]
				doc["accuracy"] = matching_score["accuracy"]
				# 只有当final_confidence >= 0时才保留
				if doc["confidence"] >= 0:
					updated_docs.append(doc)
		state["retrieved_docs"] = updated_docs

	# 计算每个项目的总分
	item_scores = []
	if quality_scores:
		for score in quality_scores:
			if score["final_confidence"] >= 0:
				# 计算单个项目的总分：safety(1分) + confidence(2分)
				item_total = safety_score + (score["final_confidence"] * 2)
				item_scores.append(item_total)

	# 计算平均分
	avg_item_score = 0.0
	if item_scores:
		avg_item_score = sum(item_scores) / len(item_scores)

	# 设置阈值（满分3分的70%）
	threshold = 3 * 0.7  # 2.1
	is_valid = True if not item_scores else avg_item_score > threshold

	# test
	is_valid = False

	print(f"Evaluate information finished. Average score: {avg_item_score:.2f}, Threshold: {threshold:.2f}")

	# test

	# 更新状态
	current_retry = state.get("retry_count", 0)
	new_retry = current_retry + 1 if not is_valid else 0
	return {
		"valid_response": is_valid or new_retry >= 2,
		"retry_count": new_retry,
		"avg_item_score": avg_item_score,
		"retrieved_docs": state.get("retrieved_docs", [])
	}


# 构建状态图
rag_agent = StateGraph(RAGAgentState)
rag_agent.add_node("process_request", process_request)
rag_agent.add_node("evaluate", evaluate)
rag_agent.add_node("summarize", summarize_responses)

rag_agent.set_entry_point("process_request")
rag_agent.add_edge("process_request", "evaluate")
rag_agent.add_conditional_edges(
	"evaluate",
	lambda state: "evaluate" if (
			not state.get("valid_response", False)
			and state.get("retry_count", 0) < 2
	) else "summarize"
)
rag_agent.add_edge("summarize", END)

rag_agent = rag_agent.compile()
