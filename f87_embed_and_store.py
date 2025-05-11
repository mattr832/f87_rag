import json
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import os
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)  # Replace with your secure method
EMBED_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536  # Dimensions for this model

# === LOAD CHUNKS ===
with open("f87_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === HELPER: Get Embedding ===
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

# === Initialize Cosine Similarity Index ===
index = faiss.IndexFlatIP(1536)
metadata_store = []

# === EMBED & STORE ===
for chunk in tqdm(chunks, desc="Embedding chunks"):
    try:
        embedding = get_embedding(chunk["text"])

        # Normalize to unit length (required for cosine similarity)
        embedding = np.array(embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(embedding)

        index.add(embedding)

        metadata_store.append({
            "title": chunk.get("title", "Untitled"),
            "url": chunk["url"],
            "text": chunk["text"]
        })

    except Exception as e:
        print(f"[ERROR] Failed to embed chunk from '{chunk.get('title', 'Untitled')}': {e}")
        continue

# === SAVE INDEX & METADATA ===
faiss.write_index(index, "f87_faiss.index")

with open("f87_metadata.json", "w", encoding="utf-8", errors="ignore") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print("✅ Embedding complete. Index and metadata saved.")