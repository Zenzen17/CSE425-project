import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATENT_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/latent_vectors")
PLOT_DIR = os.path.join(PROJECT_ROOT, "images")  
os.makedirs(PLOT_DIR, exist_ok=True)


latent_vectors = np.load(os.path.join(LATENT_DIR, "latent_vectors.npy"))
filenames = np.load(os.path.join(LATENT_DIR, "filenames.npy"))
genres = np.load(os.path.join(LATENT_DIR, "genres.npy"))

print(f"Loaded {latent_vectors.shape[0]} latent vectors, dimension {latent_vectors.shape[1]}")


n_clusters = 10  # GTZAN has 10 genres

#K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(latent_vectors)

#t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

# Plot t-SNE colored by K-Means clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=latent_2d[:,0], y=latent_2d[:,1], hue=kmeans_labels, palette="tab10", s=60)
plt.title("t-SNE of VAE Latent Vectors (K-Means)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "tsne_kmeans.png"))
plt.show()

#Plot t-SNE colored by true genres
plt.figure(figsize=(10, 8))
sns.scatterplot(x=latent_2d[:,0], y=latent_2d[:,1], hue=genres, palette="tab10", s=60)
plt.title("t-SNE of VAE Latent Vectors (True Genres)")
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "tsne_true_genres.png"))
plt.show()

#Save K-Means labels
np.save(os.path.join(LATENT_DIR, "kmeans_labels.npy"), kmeans_labels)
print("Saved K-Means cluster labels.")
