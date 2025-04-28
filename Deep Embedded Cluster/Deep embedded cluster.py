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

# enable cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data preprocessing
def load_data(file_path, text_column='review'):
    df = pd.read_csv(file_path,sep='\t')
    texts = df[text_column].tolist()
    return texts


# Deep Embedded Cluster
class DeepEmbeddedClustering(nn.Module):
    def __init__(self, input_dim, n_clusters, df=3.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.df = df

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 384)
        ).to(device)

        # cluster centers
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, 384).to(device))  # 设备初始化

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        q = self._student_t_distribution(z)
        return q

    def _student_t_distribution(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        numerator = (1 + dist / self.df) ** (- (self.df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        return numerator / denominator


# KL Divergence
def kl_loss(q, target_q):
    return (target_q * torch.log(target_q / (q + 1e-8))).sum(dim=1).mean()


# Train
def train_dec(embeddings, n_clusters, max_iters=1000):
    # normalization
    data = normalize(embeddings)
    data = torch.Tensor(data).to(device)


    model = DeepEmbeddedClustering(input_dim=data.shape[1], n_clusters=n_clusters)
    model = model.to(device)

    # initial cluster centers
    with torch.no_grad():

        encoded = model.encoder(data).cpu().detach().numpy()

        # K-means initialization
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(encoded)

        # set device to cuda
        model.cluster_centers.data.copy_(
            torch.Tensor(kmeans.cluster_centers_).to(device)
        )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # begin to train
    for epoch in range(max_iters):
        # forward
        q = model(data)

        # T-distribution caculation
        p = (q ** 2 / torch.sum(q, dim=0))
        p = (p.T / torch.sum(p, dim=1)).T

        loss = kl_loss(q, p.detach())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        q = model(data)
        labels = torch.argmax(q, dim=1).cpu().numpy()

    return labels, model.cluster_centers.cpu().detach().numpy()


# extract topic words
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


def main():
    tsv_path = "./drugsCom.tsv"
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