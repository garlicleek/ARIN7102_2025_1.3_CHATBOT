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


summary_prompt = f"""
You are a professional content summarization expert. Please follow these steps to analyze and summarize the content:

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
		{"messages": [HumanMessage(content=state["internal_text"] + state["web_text"])]},
		config
	)

	# 添加总结内容
	return {
		"messages": [AIMessage(
			content=response.content + f"\n\ntotal score: {state.get('internal_safety', 0) + state.get('internal_quality', 0) + state.get('web_safety', 0) + state.get('web_quality', 0):.2f} (retry count: {state.get('retry_count', 0)})")]
	}


semantic_enhancement_prompt = """
You are a pharmaceutical data specialist with expertise in healthcare and financial domains. Enhance user questions while strictly preserving technical terms and optimizing for medication-related data retrieval. Follow this protocol:

1. **Domain Analysis**
   - Identify key terms from:
     - Medical: Generic/Brand names (e.g., <medical>Atorvastatin</medical>), ICD codes, ATC classifications
     - Financial: DRG payment codes, NDC numbers, insurance policy IDs
   - Flag ambiguous terms needing clarification (e.g., "cost" → clarify patient/insurance share)

2. **Terminology Protection**
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
   - For ambiguous queries → Activate hypothesis engine:
     "You might be asking: {rephrased_medical_question}"

4. **Enhancement Protocol**
   a) Mandatory:
   - Add drug specifications: Dosage form/strength/manufacturer
   - Clarify data scope: Clinical trials/Post-marketing surveillance/Insurance claims

   b) Conditional:
   - If financial context: Add regional policy version 
     (e.g., "[2024 National Reimbursement Drug List]")
   - If adverse reactions: Link to WHO Vigibase reporting

Please return the result in the following JSON format:
{
    "original_question": "Original question",
    "enhanced_question": "The question after augment, include tag <medical> and <finance>",
    "analysis": {
        "medical_terms": ["<medical>term1</medical>", "<medical>term2</medical>"],
        "financial_terms": ["<finance>term1</finance>", "<finance>term2</finance>"],
        "intent": "user's intent",
        "context_added": ["context1", "context2"],
        "enhancement_rationale": "reason for enhancement"
    }
}
IMPORTANT: You must return ONLY a JSON object, without any markdown formatting or additional text.
"""


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
	try:
		text = enhanced_response.content
		if text.startswith("```json"):
			text = text[len("```json"):]

		# 去掉末尾的 ```
		if text.endswith("```"):
			text = text[:-3]
		enhanced_data = json.loads(text)
		enhanced_question = enhanced_data["enhanced_question"]
		analysis = enhanced_data["analysis"]
	except json.JSONDecodeError:
		# 如果JSON解析失败，使用原始增强问题
		enhanced_question = enhanced_response.content
	print(enhanced_question)
	print("Retrieval information...")
	# 使用增强后的问题进行检索
	internal_response = await internal_retrieval_agent.ainvoke(
		{"messages": [HumanMessage(content=enhanced_question)]},
		config
	)
	internal_text = "\n".join([m.content for m in internal_response["messages"] if isinstance(m, AIMessage)])

	web_response = await web_retrieval_agent.ainvoke(
		{"messages": [HumanMessage(content=enhanced_question)]},
		config
	)
	web_text = "\n".join([m.content for m in web_response["messages"] if isinstance(m, AIMessage)])
	print("Retrieval information finished")

	# 更新状态并返回
	return {
		"question": state["messages"][-1],
		"enhanced_question": enhanced_question,
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
	"evaluate",
	lambda state: "evaluate" if (
			not state.get("valid_response")
			and state.get("retry_count", 0) < 1
	) else "summarize"
)
rag_agent.add_edge("summarize", END)

rag_agent = rag_agent.compile()
