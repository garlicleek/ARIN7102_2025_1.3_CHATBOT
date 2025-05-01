import traceback
from typing import List, Dict, Any
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

import os
import numpy as np
from time import time
import csv
import faiss
from sentence_transformers import SentenceTransformer
from core import get_model, settings
import json
import re

# 配置参数
current_dir = os.path.dirname(os.path.abspath(__file__))
# 回退三级目录到项目根目录，再进入data
target_path = os.path.join(current_dir, '..', '..', '..', 'data')
# # 标准化路径（解决诸如../等问题）
# file_path = os.path.normpath(target_path)

DATA_PATH = "../../../data"
TEXT_FILE = os.path.normpath(os.path.join(target_path, 'drugsComSentiment.tsv'))
EMBEDDING_FILE = os.path.normpath(os.path.join(target_path, 'embeddings.npy'))
TEXT_CACHE_FILE = os.path.normpath(os.path.join(target_path, 'text_cache.npy'))
INDEX_FILE = os.path.normpath(os.path.join(target_path, 'faiss_index.index'))
MODEL_NAME = 'all-MiniLM-L6-v2'  # 平衡速度和精度的模型
BATCH_SIZE = 128  # 批处理加速编码ChatPromptTemplate

# 药品销量数据
SALES_DATA_FILE = os.path.normpath(os.path.join(target_path, 'LSTM_result.csv'))
sales_df = pd.read_csv(SALES_DATA_FILE)
drug_names = sales_df.columns.tolist()


class InternalRetrievalState(MessagesState, total=False):
	"""内部检索智能体状态"""
	retrieved_docs: List[Dict[str, Any]] | None


def wrap_model(model, instructions) -> RunnableSerializable[InternalRetrievalState, AIMessage]:
	preprocessor = RunnableLambda(
		lambda state: [SystemMessage(content=instructions)] + state["messages"],
		name="StateModifier",
	)
	return preprocessor | model


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
		texts = [f"drugName:{row[1].strip()}; useCondition:{row[2].strip()}; review:{row[3].strip()}; sentiment:{row[7].strip()}" for row in reader]

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

	# 提取结果和距离
	results = []
	for idx, dist in zip(indices[0], distances[0]):
		results.append({
			"text": texts[idx],
			"distance": float(dist)
		})
	return results


sales_instructions = f"""
You are a professional pharmaceutical sales data analysis assistant. Your tasks are:
1. Determine if the user is asking about drug sales
2. Extract drug names from the user's question (can be 0 to N drugs)

IMPORTANT: When determining drug relevance, consider the following aspects:
1. Disease-related drugs: Drugs that are commonly used to treat the mentioned disease or condition
2. Similar-effect drugs: Drugs that have similar therapeutic effects or belong to the same drug class
3. Alternative drugs: Drugs that can be used as alternatives or substitutes
4. Combination drugs: Drugs that are often used in combination with the mentioned drug

Available drug list: {drug_names}

IMPORTANT: You must return ONLY a JSON object, without any markdown formatting or additional text.
The JSON object must have the following structure:
{{
	"is_sales_related": boolean,  // Whether the question is related to drug sales
	"drug_names": [string],      // Array of extracted drug names, including both directly mentioned and relevant drugs
	"confidence": float,         // Confidence score between 0 and 1
}}

Example response (return exactly this format, nothing else):
{{
	"is_sales_related": true,
	"drug_names": ["DrugA", "DrugB", "DrugC"],
	"confidence": 0.95,
}}
"""


async def analyze_intent_and_extract_drugs(state: InternalRetrievalState, config: RunnableConfig) -> Dict:
	"""Use LLM to analyze user intent and extract drug names"""
	# Get model and analyze intent
	question = state["messages"][-1].content
	llm_model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
	model_runnable = wrap_model(llm_model, sales_instructions).with_config(tags=["skip_stream"])
	response = await model_runnable.ainvoke({"messages": [HumanMessage(content=question)]}, config)

	try:
		json_pattern = r'\{[^{}]*\}'
		match = re.search(json_pattern, response.content)
		if not match:
			raise ValueError("No JSON object found in response")

		result = json.loads(match.group(0))
		print(result)
		# 如果置信度大于0.7且是销售相关的问题，添加药品销售数据到retrieved_docs
		if result["is_sales_related"] and result["confidence"] > 0.7:
			# 获取所有置信度大于0.7的药品的销售数据
			high_confidence_drugs = result["drug_names"]
			results = []

			time_data = sales_df['ds'].tolist()

			for i, drug in enumerate(high_confidence_drugs, 1):
				if drug in sales_df.columns:
					sales_data = [
						{"date": date, "sales": sales}
						for date, sales in zip(time_data, sales_df[drug].tolist())
					]

					drug_result = {
						"id": i,
						"content": f"Sales data for {drug}: {json.dumps(sales_data, ensure_ascii=False)}",
						"source": "SQL",
						"confidence": result["confidence"]
					}
					results.append(drug_result)

			return {
				"retrieved_docs": results
			}
		else:
			return {
				"retrieved_docs": []
			}
	except Exception as e:
		print("=== Complete err info ===")
		traceback.print_exc()
		print("=================")
		return {
			"retrieved_docs": []
		}


def normalize_confidence(question: str, distances: List[float]) -> List[float]:
	"""将匹配的知识根据内积计算置信度"""
	if not distances:
		return []

	# 设置最小置信度阈值
	MIN_CONFIDENCE = 0.1

	# 使用问题与自身的相似度作为理论最大值
	question_embedding = model.encode([question], convert_to_tensor=False)[0]
	theoretical_max = np.dot(question_embedding, question_embedding)

	# 将内积值归一化到[0, 1]区间
	normalized_distances = [d / theoretical_max for d in distances]

	# 如果所有值都相同，返回最小置信度
	if len(set(normalized_distances)) == 1:
		return [MIN_CONFIDENCE] * len(distances)

	# 使用softmax进行归一化，保留相对关系
	exp_distances = np.exp(np.array(normalized_distances))
	softmax_distances = exp_distances / exp_distances.sum()

	# 将softmax结果映射到[MIN_CONFIDENCE, 1.0]区间
	confidence_scores = MIN_CONFIDENCE + (1.0 - MIN_CONFIDENCE) * softmax_distances

	return confidence_scores.tolist()


async def retrieve_docs(state: InternalRetrievalState, config: RunnableConfig) -> InternalRetrievalState:
	"""Retrieve internal documents"""
	question = state["messages"][-1].content

	search_results = search(question, top_k=3)

	distances = [result["distance"] for result in search_results]
	normalized_confidences = normalize_confidence(question, distances)

	existing_docs = state.get("retrieved_docs", [])
	formatted_results = []
	for i, (result, confidence) in enumerate(zip(search_results, normalized_confidences), 1):
		formatted_result = {
			"id": i,
			"content": result["text"],
			"source": "faiss",
			"confidence": float(confidence)
		}
		formatted_results.append(formatted_result)

	# 将结果添加到retrieved_docs
	docs = existing_docs + formatted_results

	return {"retrieved_docs": docs}


# 构建内部检索智能体图
internal_retrieval_agent = StateGraph(InternalRetrievalState)

# 添加节点
internal_retrieval_agent.add_node("analyze_intent_and_extract_drugs", analyze_intent_and_extract_drugs)
internal_retrieval_agent.add_node("retrieve_docs", retrieve_docs)

# 设置入口点
internal_retrieval_agent.set_entry_point("analyze_intent_and_extract_drugs")

# 添加边
internal_retrieval_agent.add_edge("analyze_intent_and_extract_drugs", "retrieve_docs")
internal_retrieval_agent.add_edge("retrieve_docs", END)

# 编译图并添加MemorySaver
internal_retrieval_agent = internal_retrieval_agent.compile(checkpointer=MemorySaver())
