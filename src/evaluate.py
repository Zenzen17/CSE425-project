import os
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATENT_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/latent_vectors")


latent_vectors = np.load(os.path.join(LATENT_DIR, "latent_vectors.npy"))
kmeans_labels = np.load(os.path.join(LATENT_DIR, "kmeans_labels.npy"))

sil_score = silhouette_score(latent_vectors, kmeans_labels)
ch_score = calinski_harabasz_score(latent_vectors, kmeans_labels)
db_score = davies_bouldin_score(latent_vectors, kmeans_labels)

print("=== Clustering Evaluation Metrics ===")
print(f"Silhouette Score       : {sil_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
print(f"Davies-Bouldin Index   : {db_score:.4f}")
