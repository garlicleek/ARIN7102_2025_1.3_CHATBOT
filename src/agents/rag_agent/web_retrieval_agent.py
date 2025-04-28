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
	query = state["messages"][-1].content  # obtain the newest information of users for search
	# initialization of tools of search (environmental variable 'TAVILY_API_KEY' should be set)
	try:
		search = TavilySearchResults()
		results = await search.ainvoke({"query": query})
		return {"search_results": str(results)}
	except Exception as e:
		return {"messages": [AIMessage(content=f"Search Failed: {str(e)}")]}


async def format_web_results(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
	"""格式化网络搜索结果"""
	results = state["search_results"]
	if not results:
		return {"messages": [AIMessage(content="未找到相关网络信息")]}

	formatted_content = "网络搜索结果：\n\n"
	formatted_content += results

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
