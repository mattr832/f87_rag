import json
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

# === CONFIG ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Runtime check for key
if not openai_api_key or openai_api_key.startswith("sk-old"):
    st.error("⚠️ Invalid or outdated OpenAI API key loaded. Please check your .env file.")
    st.stop()
client = OpenAI(api_key=openai_api_key)  # Replace with your secure method
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
TOP_K = 10

# === Load Data ===
index = faiss.read_index("f87_faiss.index")
with open("f87_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Utilities ===
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def retrieve_context(query_embedding, top_k=TOP_K):
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[i] for i in indices[0]]

def build_prompt(history, current_question, context_chunks):
    history_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    context = "\n\n".join(f"- {chunk['text']}" for chunk in context_chunks)
    return f"""You are a helpful assistant with access to the Bimmerpost F87 M2 forums.

Answer the user's question using ONLY the information provided in the following posts:

{context}

Previous conversation:
{history_text}

Current question:
{current_question}

Answer:"""

def generate_answer(prompt):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === Streamlit UI ===
st.set_page_config(page_title="F87 M2 Chat Assistant", layout="wide")
st.title("💬 F87 M2 Multi-Turn Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

# === Clear Chat Button ===
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# === Display chat history ===
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**Q{i+1}: {q}**")
    st.markdown(f"{a}")
    st.markdown("---")

# === Input area ===
new_question = st.text_input("Ask a question about the F87 M2")
if st.button("Ask") and new_question:
    with st.spinner("Retrieving and generating..."):
        # Embed the current question (with context if needed)
        full_context_query = new_question
        query_embedding = embed_query(full_context_query)
        context_chunks = retrieve_context(query_embedding)

        # Build prompt including history
        prompt = build_prompt(st.session_state.chat_history, new_question, context_chunks)
        answer = generate_answer(prompt)

        # Update chat history
        st.session_state.chat_history.append((new_question, answer))

        # Display answer immediately
        st.markdown(f"**You:** {new_question}")
        st.markdown(f"**Assistant:** {answer}")
        st.markdown("---")

        # Show sources
        st.subheader("📚 Retrieved Sources")
        for i, chunk in enumerate(context_chunks):
            with st.expander(f"Source {i+1}: {chunk['title']}"):
                st.markdown(f"**URL:** [{chunk['url']}]({chunk['url']})")
                st.markdown(f"**Preview:** {chunk['text'][:500]}...")
