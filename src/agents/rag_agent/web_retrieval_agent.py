import os
import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from typing import List, Dict, Any
import json
import re

from core import settings

api_key = settings.TAVILY_API_KEY


# Test API
# response = requests.post("https://api.tavily.com/search",json={"query":"test", "api_key":api_key}, timeout=5)
# print(response.status_code)

class WebRetrievalState(MessagesState, total=False):
	"""网络检索智能体状态"""
	retrieved_docs: List[Dict[str, Any]] | None


async def search_web(state: WebRetrievalState, config: RunnableConfig) -> WebRetrievalState:
	"""执行网络搜索"""
	query = state["messages"][-1].content
	# 清理查询中的语义增强标签
	query = re.sub(r'<[^>]+>', '', query).strip()
	print(f"question is {query}")
	
	try:
		# initialization of tools of search (environmental variable 'TAVILY_API_KEY' should be set)
		search = TavilySearchResults()
		results_list = await search.ainvoke({"query": query})
		print("Tavily Search Results is", results_list)

		# 格式化每个结果为JSON对象
		formatted_results = []
		for i, result in enumerate(results_list, 1):
			formatted_result = {
				"id": i,
				"content": {
					"title": result.get("title", "No Title"),
					"content": result.get("content", "No Content"),
					"url": result.get("url", "No URL"),
					"score": result.get("score", 0.7),
					"raw_content": result.get("raw_content", ""),
					"images": result.get("images", []),
					"answer": result.get("answer", "")
				},
				"source": "web",
				"confidence": result.get("score", 0.7)  # Tavily默认不打分, 所以这里全是默认值0.7
			}
			formatted_results.append(formatted_result)
		
		return {
			"retrieved_docs": formatted_results
		}
	except Exception as e:
		return {
			"retrieved_docs": []
		}


# 构建网络检索智能体图
web_retrieval_agent = StateGraph(WebRetrievalState)

# 添加节点
web_retrieval_agent.add_node("search_web", search_web)

# 设置入口点
web_retrieval_agent.set_entry_point("search_web")
web_retrieval_agent.add_edge("search_web", END)

# 编译图并添加MemorySaver
web_retrieval_agent = web_retrieval_agent.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
	# TEST API #####
	search = TavilySearchResults(tavily_api_key=api_key)
	results = search.invoke({"query": "test"})
	print(results)
