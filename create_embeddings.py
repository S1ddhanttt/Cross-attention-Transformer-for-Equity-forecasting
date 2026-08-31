import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_SEC_PATH = "data/processed_text/sec_8k_corpus.parquet"
OUTPUT_EMBED_PATH = "data/processed_text/sec_embeddings.parquet"
MODEL_NAME = "all-MiniLM-L6-v2" 

def generate_sec_embeddings():
    print(f"Loading SEC text corpus from '{INPUT_SEC_PATH}'...")
    df_sec = pd.read_parquet(INPUT_SEC_PATH)
    
    if len(df_sec) == 0:
        print("Error: SEC corpus is empty. Please check your data dump.")
        return

    print(f"Loading embedding model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"Generating embeddings for {len(df_sec)} filings on device: {device}...")
    texts = df_sec["text"].tolist()
    
  
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    df_sec["embedding"] = [emb.tolist() for emb in embeddings]
    
    df_sec.to_parquet(OUTPUT_EMBED_PATH)
    print(f"\n✅ Embeddings successfully generated and saved to '{OUTPUT_EMBED_PATH}'")
    print(f"Embedding Dimension: {len(df_sec['embedding'].iloc[0])}")

if __name__ == "__main__":
    generate_sec_embeddings()