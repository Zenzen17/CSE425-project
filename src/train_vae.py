import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from vae import VAE
import torch.nn.functional as F

#PARAMETERS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/features_mfcc")
LATENT_SAVE_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/latent_vectors")
os.makedirs(LATENT_SAVE_DIR, exist_ok=True)

INPUT_DIM = 20 * 130
LATENT_DIM = 16
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#DATASET
class MFCCDataset(Dataset):
    def __init__(self, features_dir):
        self.files = []
        for genre in os.listdir(features_dir):
            genre_path = os.path.join(features_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            for f in os.listdir(genre_path):
                if f.endswith(".npy"):
                    self.files.append(os.path.join(genre_path, f))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)
        
        x = (x - np.mean(x)) / (np.std(x) + 1e-9)
        x = x.flatten() 
        filename = os.path.basename(self.files[idx])
        genre = os.path.basename(os.path.dirname(self.files[idx]))
        return x, filename, genre

#LOSS FUNCTION
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")  # sum for VAE
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

#TRAINING
def main():
    dataset = MFCCDataset(FEATURES_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x_batch, _, _ = batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            loss = vae_loss(recon, x_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(dataset)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    #LATENT VECTORS
    model.eval()
    latent_vectors = []
    filenames = []
    genres = []
    dataloader_eval = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for x, fname, genre in tqdm(dataloader_eval, desc="Encoding latent vectors"):
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy().squeeze())
            filenames.append(fname[0])
            genres.append(genre[0])
    
    np.save(os.path.join(LATENT_SAVE_DIR, "latent_vectors.npy"), np.array(latent_vectors))
    np.save(os.path.join(LATENT_SAVE_DIR, "filenames.npy"), np.array(filenames))
    np.save(os.path.join(LATENT_SAVE_DIR, "genres.npy"), np.array(genres))

    print(f"Saved latent vectors, filenames, and genres to {LATENT_SAVE_DIR}")

if __name__ == "__main__":
    main()
