import os
import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from core import settings

api_key = settings.TAVILY_API_KEY


# Test API
# response = requests.post("https://api.tavily.com/search",json={"query":"test", "api_key":api_key}, timeout=5)
# print(response.status_code)

class WebRetrievalState(MessagesState, total=False):
	"""网络检索智能体状态"""
	search_results: str | None


async def search_web(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
	"""执行网络搜索"""
	# 框架阶段，只返回模拟数据
	query = state["messages"][-1].content
	print(f"question is {query}")
	# initialization of tools of search (environmental variable 'TAVILY_API_KEY' should be set)
	try:
		search = TavilySearchResults()
		results = await search.ainvoke({"query": query})
		return {"search_results": str(results)}
	except Exception as e:
		return {"search_results": [AIMessage(content=f"Search Failed: {str(e)}")]}


async def format_web_results(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
	"""格式化网络搜索结果"""
	results = state["search_results"]
	if not results:
		return {"messages": [AIMessage(content="Relevant information not found on website")]}

	# 检查是否是错误消息
	if isinstance(results, str) and results.startswith("Search Failed:"):
		return {"messages": [AIMessage(content=results)]}

	formatted_content = "Found the following relevant websearch results:\n\n"
	try:
		# 尝试将字符串转换为列表
		results_list = eval(results) if isinstance(results, str) else results
		
		# 确保results_list是列表
		if not isinstance(results_list, list):
			results_list = [results_list]
			
		# 格式化每个结果
		for i, result in enumerate(results_list, 1):
			if isinstance(result, dict):
				# 处理字典格式的结果
				title = result.get("title", "No Title")
				content = result.get("content", "No Content")
				url = result.get("url", "No URL")
				formatted_content += f"{i}. {title}\n"
				formatted_content += f"Content: {content}\n"
				formatted_content += f"URL: {url}\n\n"
			else:
				# 处理其他格式的结果
				formatted_content += f"{i}. {str(result)}\n\n"
	except Exception as e:
		# 如果解析失败，直接输出原始结果
		formatted_content += str(results)

	return {"messages": [AIMessage(content=formatted_content)]}


# 构建网络检索智能体图
web_retrieval_agent = StateGraph(WebRetrievalState)

# 添加节点
web_retrieval_agent.add_node("search_web", search_web)
web_retrieval_agent.add_node("format_web_results", format_web_results)

# 设置入口点
web_retrieval_agent.set_entry_point("search_web")

# 添加边
web_retrieval_agent.add_edge("search_web", "format_web_results")
web_retrieval_agent.add_edge("format_web_results", END)

# 编译图并添加MemorySaver
web_retrieval_agent = web_retrieval_agent.compile(checkpointer=MemorySaver())

# TEST API #####
search = TavilySearchResults(tavily_api_key=api_key)
results = search.invoke({"query": "test"})
print(results)
