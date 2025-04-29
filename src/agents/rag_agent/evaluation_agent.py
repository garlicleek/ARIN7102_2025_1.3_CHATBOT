from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
import os
import aiohttp
from langchain_core.messages import AIMessage, HumanMessage
from agents.llama_guard import LlamaGuard, SafetyAssessment


class EvaluationState(MessagesState, total=False):
	"""评估智能体状态"""
	safety_score: float | None
	quality_score: float | None


async def evaluate_safety(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
	"""使用LlamaGuard评估内容安全性，只返回安全评分"""
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
	context_messages = messages[-3:] if len(messages) >= 3 else messages

	# 准备API请求
	api_key = os.getenv("DEEPSEEK_API_KEY")
	url = "https://api.deepseek.com/v1/chat/completions"
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}

	# 构建评估指令
	system_prompt = """
You are a quality assessment expert. Please evaluate the AI response based on the following criteria and provide a weighted score:
1. Information Accuracy (Weight: 20%)
   - Score: 0-1
   - Criteria: Factual correctness, data reliability, source credibility
2. Logical Coherence (Weight: 10%)
   - Score: 0-1
   - Criteria: Flow of ideas, argument structure, reasoning clarity
3. User Need Match (Weight: 60%)
   - Score: 0-1
   - Criteria: Relevance to user's question, completeness of answer, practical value
4. Language Standardization (Weight: 10%)
   - Score: 0-1
   - Criteria: Grammar, clarity, professional tone
Calculation Method:
1. First, score each criterion separately (0-1)
2. Then apply weights:
   - Information Accuracy * 0.2
   - Logical Coherence * 0.1
   - User Need Match * 0.6
   - Language Standardization * 0.1
3. Sum the weighted scores to get the final score (0-1)

Return only the final weighted score as a number between 0 and 1."""

	# 构造对话历史
	formatted_messages = [{"role": "system", "content": system_prompt}]
	for msg in context_messages:
		role = "assistant" if isinstance(msg, AIMessage) else "user"
		formatted_messages.append({"role": role, "content": msg.content})

	payload = {
		"model": "deepseek-chat",
		"messages": formatted_messages,
		"temperature": 0.1,
		"max_tokens": 4
	}

	# 调用API
	async with aiohttp.ClientSession() as session:
		async with session.post(url, json=payload, headers=headers) as response:
			if response.status == 200:
				result = await response.json()
				response_text = result['choices'][0]['message']['content'].strip()

				# 解析评分
				try:
					quality_score = min(max(float(response_text), 0.0), 1.0)
				except ValueError:
					quality_score = 0.7  # 默认值
			else:
				quality_score = 0.7  # 失败时默认值

	return {"quality_score": quality_score}


async def format_evaluation(state: EvaluationState, config: RunnableConfig) -> EvaluationState:
	"""格式化评估结果，只显示分数"""
	safety = state.get("safety_score", 0)
	quality = state.get("quality_score", 0)

	content = f"Evaluation Result:\n"
	content += f"Safety Score: {safety:.2f}\n"
	content += f"Quality Score: {quality:.2f}\n"

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
