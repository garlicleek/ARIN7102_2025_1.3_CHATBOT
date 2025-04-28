from agents.rag_agent.internal_retrieval_agent import internal_retrieval_agent
from agents.rag_agent.web_retrieval_agent import web_retrieval_agent
from agents.rag_agent.evaluation_agent import evaluation_agent
from typing import Optional
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
import os
import aiohttp


class RAGAgentState(MessagesState, total=False):
    """增强的RAG智能体状态"""
    valid_response: bool
    internal_safety: float
    internal_quality: float
    web_safety: float
    web_quality: float
    retry_count: int


async def summarize_responses(internal_content: str, web_content: str) -> str:
    """DeepSeek总结生成"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "API密钥未配置"

    url = "https://api.deepseek.com/v1/chat/completions"
    prompt = f"""请根据以下两个来源的内容生成简明的总结：

    [内部知识库]
    {internal_content}

    [网络资源]
    {web_content}

    用中文输出3-5句话的总结，保持客观中立。"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                return f"API请求失败：{response.status}"
    except Exception as e:
        return f"总结生成异常：{str(e)}"


async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
    """完整处理流程"""
    # 初始化重试计数器
    current_retry = state.get("retry_count", 0)

    # 强制结束条件
    if current_retry >= 2:
        return {
            "valid_response": True,
            "retry_count": current_retry,
            "messages": [AIMessage(content="已达到最大重试次数，返回最终结果")]
        }

    # 清空前次无效响应
    if not state.get("valid_response", True):
        state["messages"] = []

    # 并行获取响应
    internal_response = await internal_retrieval_agent.ainvoke(state, config)
    web_response = await web_retrieval_agent.ainvoke(state, config)

    # 评估逻辑
    internal_eval = await evaluation_agent.ainvoke(
        {"messages": internal_response["messages"]},
        config
    )
    web_eval = await evaluation_agent.ainvoke(
        {"messages": web_response["messages"]},
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

    # 生成总结条件
    summary = ""
    if is_valid or current_retry >= 1:  # 最后一次强制生成
        internal_text = "\n".join([
            m.content for m in internal_response["messages"]
            if isinstance(m, AIMessage)
        ])
        web_text = "\n".join([
            m.content for m in web_response["messages"]
            if isinstance(m, AIMessage)
        ])
        summary = await summarize_responses(internal_text, web_text)

    # 构建消息
    messages = [
        *internal_response["messages"],
        *web_response["messages"],
        AIMessage(content=f"评估总分: {total_score:.2f} (重试次数: {current_retry})")
    ]
    if summary:
        messages.append(AIMessage(content=f"【最终总结】\n{summary}"))

    # 更新状态
    new_retry = current_retry + 1 if not is_valid else 0
    return {
        "valid_response": is_valid or new_retry >= 2,
        "retry_count": new_retry,
        "internal_safety": internal_eval.get("safety_score", 0),
        "internal_quality": internal_eval.get("quality_score", 0),
        "web_safety": web_eval.get("safety_score", 0),
        "web_quality": web_eval.get("quality_score", 0),
        "messages": messages
    }


# 构建状态图
rag_agent = StateGraph(RAGAgentState)
rag_agent.add_node("process_request", process_request)
rag_agent.set_entry_point("process_request")

rag_agent.add_conditional_edges(
    "process_request",
    lambda state: "process_request" if (
            not state.get("valid_response")
            and state.get("retry_count", 0) < 2
    ) else END
)

rag_agent = rag_agent.compile(
    checkpointer=MemorySaver(),
    debug=False
)