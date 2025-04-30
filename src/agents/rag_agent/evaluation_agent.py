import json
import math
from typing import Literal, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
import os
import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.llama_guard import LlamaGuard, SafetyAssessment
from langchain_core.documents import Document
from core import get_model, settings
import ast


class EvaluationState(MessagesState, total=False):
	"""评估智能体状态"""
	safety_score: float | None
	quality_score: List[Dict[
		str, Any]] | None  # 每个评分项包含id, source, original_confidence, relevance, accuracy, final_confidence


def wrap_model(model, instructions) -> RunnableSerializable[EvaluationState, AIMessage]:
	preprocessor = RunnableLambda(
		lambda state: [SystemMessage(content=instructions)] + state["messages"],
		name="StateModifier",
	)
	return preprocessor | model


async def evaluate_safety(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
	"""使用LlamaGuard评估内容安全性, 只返回安全评分"""
	try:
		# 初始化LlamaGuard
		guard = LlamaGuard()

		# 获取对话历史中的最后一条AI消息
		messages = state.get("messages", [])
		last_ai_message = next(
			(msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
			None
		)

		if not last_ai_message:
			return {"safety_score": 1.0}  # 没有AI消息视为安全

		# 构建包含上下文的对话历史（最后3条消息）
		context_messages = messages[-3:] if len(messages) >= 3 else messages

		# 调用LlamaGuard进行评估
		assessment = await guard.ainvoke(
			role="Agent",  # 评估AI回复
			messages=context_messages
		)

		# 计算安全评分 (1.0=安全, 0.0=不安全)
		safety_score = 1.0 if assessment.safety_assessment == SafetyAssessment.SAFE else 0.0

		return {"safety_score": safety_score}
	except Exception as e:
		print(f"LlamaGuard evaluation failed: {str(e)}")
		# 如果LlamaGuard评估失败，默认返回安全
		return {"safety_score": 1.0}


async def evaluate_quality(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
	"""使用DeepSeek API评估内容质量"""
	# 获取对话历史
	messages = state.get("messages", [])

	# 提取最近3条消息作为上下文（可根据API要求调整）
	question_messages = messages[0]
	answer_messages = messages[1]

	# 准备API请求
	api_key = os.getenv("DEEPSEEK_API_KEY")
	url = "https://api.deepseek.com/v1/chat/completions"
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}

	# 构建评估指令
	system_prompt = f"""
You are a quality pharmacy expert and retrieval-augmentation expert. Follow the instructions carefully.

I have a list of retrieved documents or content segments. Each item has its original confidence score(based on embeddings or retrieval system). 
I want you to evaluate each item's relevance and accuracy.

** criteria instruction **:
1. Parse if the response relevant to user's current question (0-1)
2. Parse the information Accuracy/Correctness (0-1)

**Steps instruction** :
1. Read the user's question: {question_messages.content}
2. Read each retrieved content item and its original confidence.
3. Apply the **criteria instruction**.
Please provide the chain-of-thought reasoning implicitly (internally) but do not reveal the full chain-of-thought in the final output. Instead, produce only the score that you think is appropriate.

** Return Format **
A JSON of list structure. For each content item, show the id, source, relevance score and accuracy score:
[
	{{"id": id_number, "source": the source of response, "relevance": relevance_score, "accuracy": accuracy_score}},
	...
]

Example: 
	- Input: [
		{{"id": 1, "content": "relevant knowledge 1", "source": "sql", "confidence": 0.6}}, 
		{{"id": 2, "content": {{"title": "title 2", "content": "relevant knowledge 2"}}, "source": "web", "confidence": 0.3}},
		{{"id": 3, "content": "irrelevant knowledge 3", "source": "faiss", "confidence": 0.2}}
	]
    - Output: [
        {{"id": 1, "source": "sql", "relevance": 0.8, "accuracy": 0.9}}, 
        {{"id": 2, "source": "web", "relevance": 0.7, "accuracy": 0.8}}, 
        {{"id": 3, "source": "faiss", "relevance": 0.2, "accuracy": 0.6}}
    ]
"""

	llm_model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
	model_runnable = wrap_model(llm_model, system_prompt).with_config(tags=["skip_stream"])
	response = await model_runnable.ainvoke({"messages": [HumanMessage(content=answer_messages.content)]}, config)

	try:
		# 解析LLM返回的评分JSON
		# 清理响应内容中的Markdown代码块标记
		cleaned_content = response.content.strip()
		if cleaned_content.startswith('```json'):
			cleaned_content = cleaned_content[7:]
		if cleaned_content.endswith('```'):
			cleaned_content = cleaned_content[:-3]
		cleaned_content = cleaned_content.strip()

		llm_scores = json.loads(cleaned_content)  # 解析清理后的内容
		print(llm_scores)

		# 解析原始答案JSON以获取原始confidence
		answer_json = json.loads(answer_messages.content)
		confidence_dict = {(item['id'], item['source']): item['confidence'] for item in answer_json}
		print("original confidence:", confidence_dict)

		# 构建只包含评分信息的输出
		quality_score = []
		for score in llm_scores:
			key = (score['id'], score['source'])
			if key in confidence_dict:
				original_confidence = confidence_dict[key]
				# 计算最终分数
				final_score = (
						original_confidence * 0.6 +  # 原始置信度权重
						score['relevance'] * 0.3 +  # 相关性权重
						score['accuracy'] * 0.1  # 准确性权重
				)

				# 如果最终分数小于0.3，设置为-1
				if final_score < 0.3:
					final_score = -1

				quality_score.append({
					"id": score['id'],
					"source": score['source'],
					"original_confidence": original_confidence,
					"relevance": score['relevance'],
					"accuracy": score['accuracy'],
					"final_confidence": final_score
				})

	except (SyntaxError, ValueError, json.JSONDecodeError) as e:
		print(f"Error processing JSON: {str(e)}")
		quality_score = []

	print("final confidence:", quality_score)

	return {
		"quality_score": quality_score
	}


# 构建评估智能体图
evaluation_agent = StateGraph(EvaluationState)

# 添加节点
evaluation_agent.add_node("evaluate_safety", evaluate_safety)
evaluation_agent.add_node("evaluate_quality", evaluate_quality)

# 设置入口点
evaluation_agent.set_entry_point("evaluate_safety")

# 添加边
evaluation_agent.add_edge("evaluate_safety", "evaluate_quality")
evaluation_agent.add_edge("evaluate_quality", END)

# 编译图并添加MemorySaver
evaluation_agent = evaluation_agent.compile(checkpointer=MemorySaver())
