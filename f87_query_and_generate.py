import json
import faiss
import numpy as np
from openai import OpenAI

# === CONFIG ===
client = OpenAI(api_key="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")  # Replace with your key
EMBED_MODEL = "text-embedding-3-small"
# CHAT_MODEL = "gpt-3.5-turbo"
CHAT_MODEL = "gpt-4"
TOP_K = 10

# === Load FAISS Index and Metadata ===
index = faiss.read_index("f87_faiss.index")
with open("f87_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Helper: Embed Query ===
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)  # normalize for cosine similarity
    return vec

# === Helper: Retrieve Top K Chunks ===
def retrieve_context(query_embedding, top_k=TOP_K):
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[i] for i in indices[0]]

# === Helper: Build Prompt ===
def build_augmented_prompt(context_chunks, question):
    context = "\n\n".join(f"- {chunk['text']}" for chunk in context_chunks)
    return f"""You are a helpful assistant with access to the Bimmerpost F87 M2 forums.

Answer the user's question using ONLY the information provided in the following posts:

{context}

Question: {question}
Answer:"""

# === Helper: Generate Answer ===
def generate_answer(prompt):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === MAIN INTERFACE ===
def ask_question():
    while True:
        question = input("\nAsk a question about the F87 M2 (or type 'exit'): ")
        if question.lower() in {"exit", "quit"}:
            break

        query_embedding = embed_query(question)
        context_chunks = retrieve_context(query_embedding)
        prompt = build_augmented_prompt(context_chunks, question)
        answer = generate_answer(prompt)

        print("\n🔎 {CHAT_MODEL} Answer:")
        print(answer)
        print("\n📚 Sources:")
        for chunk in context_chunks:
            print(f"- {chunk['title']} → {chunk['url']}")
        print("-" * 50)

if __name__ == "__main__":
    ask_question()
