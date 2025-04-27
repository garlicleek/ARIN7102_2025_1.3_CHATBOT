import json
import traceback
from tqdm import tqdm

import pandas as pd
from sqlalchemy import create_engine
from neo4j import GraphDatabase
from datetime import datetime
from sqlalchemy import text


def load_config():
	"""加载配置文件"""
	with open('config.json', 'r') as f:
		return json.load(f)


def test_neo4j_connection(driver):
	"""测试Neo4j连接"""
	try:
		with driver.session() as session:
			result = session.run("RETURN 1 as test")
			print("Neo4j连接测试成功！")
	except Exception as e:
		print(f"Neo4j连接测试失败: {str(e.__traceback__)}")
		traceback.print_exc()
		return False
	return True


def test_sql_connection(engine):
	"""测试SQL连接"""
	try:
		with engine.connect() as conn:
			result = conn.execute(text("SHOW TABLES"))
			print("MySQL连接测试成功！")
	except Exception as e:
		print(f"SQL连接测试失败: {str(e)}")
		traceback.print_exc()
		return False
	return True


def get_sql_connection(config):
	"""创建SQL连接"""
	sql_config = config['sql']
	connection_string = f"mysql+pymysql://{sql_config['user']}:{sql_config['password']}@{sql_config['host']}:{sql_config['port']}/{sql_config['database']}"
	engine = create_engine(connection_string)
	if not test_sql_connection(engine):
		raise Exception("SQL连接测试失败，请检查配置")
	return engine


def get_neo4j_driver(config):
	"""创建Neo4j驱动"""
	neo4j_config = config['neo4j']
	driver = GraphDatabase.driver(
		neo4j_config['uri'],
		auth=(neo4j_config['user'], neo4j_config['password'])
	)
	if not test_neo4j_connection(driver):
		raise Exception("Neo4j连接测试失败，请检查配置")
	return driver


def import_lstm_result(engine, driver):
	"""导入lstm_result表数据到Neo4j"""
	# 从SQL读取数据
	query = "SELECT * FROM lstm_result"
	df = pd.read_sql(query, engine)

	with driver.session() as session:
		# 清空LSTM相关数据
		session.run("""
            MATCH (n:Sales_Entity)
            DETACH DELETE n
        """)
		session.run("""
            MATCH (n:Sales)
            DETACH DELETE n
        """)

		# 获取所有药品列名（除了ds列）
		drug_columns = [col for col in df.columns if col != 'ds']

		# 为每个药品创建节点和关系
		for drug in drug_columns:
			# 创建药品节点
			session.run("""
                MERGE (e:Sales_Entity {name: $name})
                SET e.source = 'lstm_result'
            """, {
				'name': drug
			})

			# 为每个时间点的销量创建节点和关系
			for _, row in df.iterrows():
				date_str = row['ds']
				value = row[drug]

				# 创建销量节点
				session.run("""
                    MERGE (s:Sales {date: $date})
                    SET s.value = $value,
                        s.source = 'lstm_result'
                """, {
					'date': date_str,
					'value': value
				})

				# 创建关系
				session.run("""
                    MATCH (e:Sales_Entity {name: $drug_name})
                    MATCH (s:Sales {date: $date})
                    MERGE (e)-[r:HAS_SALES]->(s)
                    SET r.value = $value
                """, {
					'drug_name': drug,
					'date': date_str,
					'value': value
				})


def import_drugs_comments(engine, driver):
	"""导入drugscomsentiment表数据到Neo4j"""
	print("⏳ 正在从数据库读取数据...")
	query = "SELECT * FROM drugscomsentiment"
	df = pd.read_sql(query, engine)
	total_records = len(df)
	print(f"✅ 成功读取 {total_records} 条记录")

	with driver.session() as session:
		# 清空药品评论相关数据
		print("⚠️ 正在清空旧数据...")
		session.run("""
            MATCH (n:Drug)
            DETACH DELETE n
        """)
		session.run("""
            MATCH (n:Comment)
            DETACH DELETE n
        """)
		print("🗑️ 旧数据清理完成")

		# 首先创建所有药品节点
		print("💊 开始创建药品节点...")
		unique_drugs = df['drugName'].unique()
		for drug in unique_drugs:
			session.run("""
                MERGE (d:Drug {name: $name})
                SET d.source = 'drugscomsentiment'
            """, {
				'name': drug
			})
		print(f"✅ 成功创建 {len(unique_drugs)} 个药品节点")

		# 创建评论节点并建立关系
		print("📝 开始处理评论及关系...")
		for _, row in df.iterrows():
			# 创建评论节点
			session.run("""
                MERGE (c:Comment {id: $id})
                SET c.content = $content,
                    c.condition = $condition,
                    c.rating = $rating,
                    c.date = $date,
                    c.usefulCount = $usefulCount,
                    c.sentiment = $sentiment,
                    c.source = 'drugscomsentiment'
            """, {
				'id': row['id'],
				'content': row['review'],
				'condition': row['condition'],
				'rating': row['rating'],
				'date': row['date'],
				'usefulCount': row['usefulCount'],
				'sentiment': row['sentiment']
			})

			# 创建药品和评论之间的关系
			session.run("""
                MATCH (d:Drug {name: $drug_name})
                MATCH (c:Comment {id: $comment_id})
                MERGE (d)-[r:HAS_COMMENT]->(c)
            """, {
				'drug_name': row['drugName'],
				'comment_id': row['id']
			})
			print(f"📊 导入完成！")


if __name__ == "__main__":
	config = load_config()
	neo4jDriver = get_neo4j_driver(config)
	sqlDriver = get_sql_connection(config)

	# print("开始导入lstm_result数据...")
	# import_lstm_result(sqlDriver, neo4jDriver)
	# print("lstm_result数据导入完成")

	print("开始导入药品评论数据...")
	import_drugs_comments(sqlDriver, neo4jDriver)
	print("药品评论数据导入完成")

	neo4jDriver.close()
