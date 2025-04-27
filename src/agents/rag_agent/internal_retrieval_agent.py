from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

import os
import numpy as np
from time import time
import csv
import faiss
from sentence_transformers import SentenceTransformer

# 配置参数
current_dir = os.path.dirname(os.path.abspath(__file__))
# 回退三级目录到项目根目录，再进入data
target_path = os.path.join(current_dir, '..', '..', '..', 'data')
# # 标准化路径（解决诸如../等问题）
# file_path = os.path.normpath(target_path)

DATA_PATH = "../../../data"
# TEXT_FILE = "../../../data/drugsComSentiment.tsv"
TEXT_FILE = os.path.normpath(os.path.join(target_path, 'drugsComSentiment.tsv'))
# EMBEDDING_FILE = f"{DATA_PATH}/embeddings.npy"
EMBEDDING_FILE = os.path.normpath(os.path.join(target_path, 'embeddings.npy'))
# TEXT_CACHE_FILE = f"{DATA_PATH}/text_cache.npy"
TEXT_CACHE_FILE = os.path.normpath(os.path.join(target_path, 'text_cache.npy'))
# INDEX_FILE = f"{DATA_PATH}/faiss_index.index"
INDEX_FILE = os.path.normpath(os.path.join(target_path, 'faiss_index.index'))
MODEL_NAME = 'all-MiniLM-L6-v2'  # 平衡速度和精度的模型
BATCH_SIZE = 128  # 批处理加速编码

class InternalRetrievalState(MessagesState, total=False):
    """内部检索智能体状态"""
    retrieved_docs: List[Document] | None


def load_or_build():
    # 如果已有缓存直接加载
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE):
        print("Loading cached embeddings...")
        embeddings = np.load(EMBEDDING_FILE)
        texts = np.load(TEXT_CACHE_FILE, allow_pickle=True)
        index = faiss.read_index(INDEX_FILE)
        return texts, embeddings, index

    # 需要重新生成
    else:
        print(f"找不到文件,路径{EMBEDDING_FILE}和{INDEX_FILE}")

    print("Processing text file...")
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # 跳过标题行
        texts = [f"drugName:{row[1].strip()}; useCondition:{row[2].strip()}; review:{row[3].strip()}" for row in
                 reader]

    print(f"Encoding {len(texts)} texts...")
    model = SentenceTransformer(MODEL_NAME)

    # 批量编码提升速度
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # 保存缓存
    np.save(EMBEDDING_FILE, embeddings)
    np.save(TEXT_CACHE_FILE, texts)

    # 构建Faiss索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 使用内积（余弦相似度）
    index.add(embeddings.astype('float32'))  # Faiss需要float32
    faiss.write_index(index, INDEX_FILE)

    return texts, embeddings, index

texts, embeddings, index = load_or_build()
model = SentenceTransformer(MODEL_NAME)  # 单独加载模型用于查询编码


def search(question, top_k=5):
    # 编码问题
    query_embedding = model.encode([question], convert_to_tensor=False)

    # Faiss搜索
    start = time()
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    print(f"Search time: {time() - start:.4f}s")

    # 提取结果
    return [texts[i] for i in indices[0]]


async def retrieve_docs(state: InternalRetrievalState, config: RunnableConfig) -> InternalRetrievalState:
    """检索内部文档"""
    # 在这一部分，只需要找到问题对应的文档内容即可
    # 框架阶段，只返回模拟数据
    print(f"state is {state['messages']}")
    question = state['messages'][0].content
    print(f"question is {question}")

    results = search(question, top_k=3)
    print("Top 5 results:")
    for i, res in enumerate(results):
        print(f"{i + 1}. {res}")

    return {"retrieved_docs": [Document(page_content=f"{content}") for content in results]}


async def format_results(state: InternalRetrievalState, config: RunnableConfig) -> InternalRetrievalState:
    """格式化检索结果"""
    # docs 是上一步retrieve_docs函数中得到的"retrieved_docs"部分字段，也就是list[page_content, page_content...]
    docs = state["retrieved_docs"]
    if not docs:
        return {"messages": [AIMessage(content="未找到相关内部文档")]}
    
    formatted_content = "找到以下相关内部文档：\n\n"
    for i, doc in enumerate(docs, 1):
        formatted_content += f"{i}. {doc.page_content}\n\n"
    
    return {"messages": [AIMessage(content=formatted_content)]}


# 构建内部检索智能体图
internal_retrieval_agent = StateGraph(InternalRetrievalState)

# 添加节点
internal_retrieval_agent.add_node("retrieve_docs", retrieve_docs)       # 搜索文档
internal_retrieval_agent.add_node("format_results", format_results)     # 格式化

# 设置入口点
internal_retrieval_agent.set_entry_point("retrieve_docs")

# 添加边
internal_retrieval_agent.add_edge("retrieve_docs", "format_results")
internal_retrieval_agent.add_edge("format_results", END)

# 编译图并添加MemorySaver
internal_retrieval_agent = internal_retrieval_agent.compile(checkpointer=MemorySaver()) 