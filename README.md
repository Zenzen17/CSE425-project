# Music Genre Clustering using Variational Autoencoder (VAE)

This project explores unsupervised music clustering using a Variational Autoencoder (VAE).
Audio features are extracted from songs and compressed into a latent space, where clustering
is performed to discover underlying structure in music genres.

## Dataset
- **GTZAN Music Genre Dataset**
- 10 genres, 100 tracks per genre
- Audio format: `.wav`
- Used only audio (no lyrics)

## Features
- MFCC (Mel-Frequency Cepstral Coefficients)
- 20 MFCCs per track
- Fixed-length padding/truncation

## Project Structure
music_vae_clustering/
├── src/
│ ├── extract_features.py
│ ├── vae.py
│ ├── train_vae.py
│ ├── clustering.py
│ └── evaluate.py
├── data/
│ └── gtzan/
│ └── features_mfcc/
├── .gitignore
└── README.md

## Pipeline
1. Extract MFCC features from audio files
2. Train a VAE to learn latent representations
3. Apply K-Means clustering in latent space
4. Visualize clusters using dimensionality reduction
5. Compare with PCA + K-Means baseline
6. Evaluate clustering quality using standard metrics

## Models
- Fully connected Variational Autoencoder (VAE)
- Latent dimension: 16
- Optimizer: Adam
- Loss: Reconstruction (MSE) + KL divergence

## Clustering
- Algorithm: K-Means
- Number of clusters: 10
- Baseline: PCA + K-Means

## Evaluation Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

## Results (VAE Latent Space)
| Metric | Value |
|------|------|
| Silhouette Score | 0.0505 |
| Calinski-Harabasz Index | 45.06 |
| Davies-Bouldin Index | 2.51 |

## How to Run
```bash
# Feature extraction
python src/extract_features.py

# Train VAE
python src/train_vae.py

# Clustering + visualization
python src/clustering.py

# Evaluation
python src/evaluate.py

Author 
Mehezebin