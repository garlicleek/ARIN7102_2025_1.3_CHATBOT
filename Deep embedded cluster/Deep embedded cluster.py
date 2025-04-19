import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# 新增设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 数据预处理模块（保持不变）
def load_data(file_path, text_column='review'):
    df = pd.read_csv(file_path,sep='\t')
    texts = df[text_column].tolist()
    return texts


# 2. 深度嵌入聚类模型（添加设备支持）
class DeepEmbeddedClustering(nn.Module):
    def __init__(self, input_dim, n_clusters, df=3.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.df = df

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 384)
        ).to(device)  # 将编码器移动到设备

        # 聚类中心参数
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, 384).to(device))  # 设备初始化

    def forward(self, x):
        x = x.to(device)  # 确保输入在设备上
        z = self.encoder(x)
        q = self._student_t_distribution(z)
        return q

    def _student_t_distribution(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        numerator = (1 + dist / self.df) ** (- (self.df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        return numerator / denominator


# 3. KL散度优化函数（保持不变）
def kl_loss(q, target_q):
    return (target_q * torch.log(target_q / (q + 1e-8))).sum(dim=1).mean()


# 4. 聚类训练流程（设备兼容改造）
def train_dec(embeddings, n_clusters, max_iters=1000):
    # 数据预处理
    data = normalize(embeddings)
    data = torch.Tensor(data).to(device)  # 数据送设备

    # 模型初始化
    model = DeepEmbeddedClustering(input_dim=data.shape[1], n_clusters=n_clusters)
    model = model.to(device)  # 模型送设备

    # 初始化聚类中心（设备兼容）
    with torch.no_grad():
        # 编码数据并移至CPU进行K-means
        encoded = model.encoder(data).cpu().detach().numpy()  # 重要修改！

        # K-means初始化
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(encoded)

        # 设备一致性处理
        model.cluster_centers.data.copy_(
            torch.Tensor(kmeans.cluster_centers_).to(device)
        )

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(max_iters):
        # 前向传播
        q = model(data)

        # 目标分布计算
        p = (q ** 2 / torch.sum(q, dim=0))
        p = (p.T / torch.sum(p, dim=1)).T

        # 损失计算
        loss = kl_loss(q, p.detach())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 获取结果（设备兼容）
    with torch.no_grad():
        q = model(data)
        labels = torch.argmax(q, dim=1).cpu().numpy()  # 移至CPU

    return labels, model.cluster_centers.cpu().detach().numpy()  # 移至CPU


# 5. 主题关键词提取（保持不变）
def extract_topic_keywords(texts, labels, centers, embedding_model, n_words=15):
    vectorizer = CountVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    word_embeddings = embedding_model.encode(words, show_progress_bar=False)

    topics = {}
    for cluster_id in range(centers.shape[0]):
        center = centers[cluster_id]
        sim_scores = word_embeddings.dot(center)
        top_indices = sim_scores.argsort()[-n_words:][::-1]
        topic_keywords = [words[i] for i in top_indices]

        topics[f"Topic_{cluster_id}"] = {
            'keywords': topic_keywords,
            'representative_vector': center
        }
    return topics


# 主流程（保持不变）
def main():
    tsv_path = "../drugsComSentiment.tsv"
    text_column = "review"
    n_clusters = 5

    texts = load_data(tsv_path, text_column)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(texts)

    labels, centers = train_dec(embeddings, n_clusters)

    topics = extract_topic_keywords(texts, labels, centers, embed_model)

    reducer = UMAP(n_components=2)
    vis_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(vis_embeddings[:, 0], vis_embeddings[:, 1], c=labels, cmap='Spectral', alpha=0.6)
    plt.colorbar()
    plt.title("Deep Embedded Clustering with Student-t Distribution")
    plt.show()

    print("Discovered Topics:")
    for topic_id, info in topics.items():
        print(f"\n{topic_id}: {', '.join(info['keywords'][:10])}...")


if __name__ == "__main__":
    main()