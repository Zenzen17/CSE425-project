import os
import numpy as np
import librosa
from tqdm import tqdm

#PARAMETERS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/genres_original")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/gtzan/features_mfcc")

SAMPLE_RATE = 22050
N_MFCC = 20
MAX_LEN = 130

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """Load audio file and extract MFCC features."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Skipping {file_path}, error: {e}")
        return None
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def main():
    print(f"Starting MFCC extraction...\nData folder: {DATA_DIR}\nOutput folder: {OUTPUT_DIR}")

    for genre in os.listdir(DATA_DIR):
        genre_path = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        
        genre_out_path = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(genre_out_path, exist_ok=True)
        
        for file_name in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if not file_name.endswith(".wav"):
                continue

            out_file = os.path.join(genre_out_path, file_name.replace(".wav", ".npy"))
            
            
            if os.path.exists(out_file):
                continue

            file_path = os.path.join(genre_path, file_name)
            mfcc = extract_mfcc(file_path)
            if mfcc is None:
                continue
            
            np.save(out_file, mfcc)
    
    print("MFCC extraction completed!")

if __name__ == "__main__":
    main()
