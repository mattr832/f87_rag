import json
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# === CONFIGURATION ===
client = OpenAI(api_key="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")
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

# === INIT FAISS INDEX ===
index = faiss.IndexFlatL2(VECTOR_DIM)
metadata_store = []

# === EMBED & STORE ===
for chunk in tqdm(chunks, desc="Embedding chunks"):
    try:
        embedding = get_embedding(chunk["text"])
        index.add(np.array([embedding], dtype="float32"))

        metadata_store.append({
            "title": chunk.get("title", "Untitled"),
            "url": chunk["url"],
            "text": chunk["text"]
        })
    except Exception as e:
        print(f"[ERROR] Failed to embed chunk from '{chunk.get('title', 'Untitled')}': {e}")

# === SAVE INDEX & METADATA ===
faiss.write_index(index, "f87_faiss.index")

with open("f87_metadata.json", "w", encoding="utf-8", errors="ignore") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print("✅ Embedding complete. Index and metadata saved.")