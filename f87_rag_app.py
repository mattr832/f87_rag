import json
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

# === CONFIG ===
client = OpenAI(api_key="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")  # Replace securely in production
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
TOP_K = 10

# === Load Data ===
index = faiss.read_index("f87_faiss.index")
with open("f87_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Embed Query ===
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# === Retrieve Relevant Chunks ===
def retrieve_context(query_embedding, top_k=TOP_K):
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[i] for i in indices[0]]

# === Build Prompt ===
def build_prompt(context_chunks, question):
    context = "\n\n".join(f"- {chunk['text']}" for chunk in context_chunks)
    return f"""You are a helpful assistant with access to the Bimmerpost F87 M2 forums.

Answer the user's question using ONLY the information provided in the following posts:

{context}

Question: {question}
Answer:"""

# === Generate GPT Answer ===
def generate_answer(prompt):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === Streamlit UI ===
st.set_page_config(page_title="F87 M2 RAG Assistant", layout="wide")
st.title("🔧 F87 M2 Forum Assistant")
st.write("Ask a question and get grounded answers from Bimmerpost threads.")

question = st.text_input("Enter your question about the F87 M2")

if st.button("Search") and question:
    with st.spinner("Retrieving relevant forum posts..."):
        query_embedding = embed_query(question)
        context_chunks = retrieve_context(query_embedding)
        prompt = build_prompt(context_chunks, question)
        answer = generate_answer(prompt)

    st.subheader("🔎 GPT Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Sources")
    for chunk in context_chunks:
        st.markdown(f"**{chunk['title']}**  \n{chunk['text'][:300]}...  \n🔗 [{chunk['url']}]({chunk['url']})")