from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph




class RAGAgentState(MessagesState, total=False):
    """RAG智能体状态"""
    pass


async def process_request(state: RAGAgentState, config: RunnableConfig) -> RAGAgentState:
    """处理用户请求"""
    pass


# 构建主智能体图
rag_agent = StateGraph(RAGAgentState)

# 添加节点
rag_agent.add_node("process_request", process_request)

# 设置入口点
rag_agent.set_entry_point("process_request")

# 添加边
rag_agent.add_edge("process_request", END)

# 编译图并添加MemorySaver
rag_agent = rag_agent.compile(checkpointer=MemorySaver()) 