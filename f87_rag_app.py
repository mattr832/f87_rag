import json
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

# === MUST BE FIRST ===
st.set_page_config(page_title="F87 M2 Chat Assistant", layout="wide")

# === CONFIG ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Runtime check for key
if not openai_api_key or openai_api_key.startswith("sk-old"):
    st.error("⚠️ Invalid or outdated OpenAI API key loaded. Please check your .env file.")
    st.stop()
client = OpenAI(api_key=openai_api_key)  # Replace with your secure method
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo"
TOP_K = 5

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

st.sidebar.title("Options")
if st.sidebar.button("💾 Export Chat"):
    if st.session_state.chat_history:
        chat_export = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        st.sidebar.download_button(
            label="Download Chat as .txt",
            data=chat_export,
            file_name="f87_chat_history.txt",
            mime="text/plain"
        )
    else:
        st.sidebar.info("No chat history to export.")

st.title("💬 F87 M2 AI Assistant")

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

        # Compute confidence score
        distances, _ = index.search(query_embedding, TOP_K)
        weights = np.array([1 / (i + 1) for i in range(TOP_K)], dtype="float32")
        weights /= weights.sum()
        avg_similarity = float(np.dot(distances[0], weights))
        confidence_pct = round(avg_similarity * 100, 1)

        # Build prompt including history
        prompt = build_prompt(st.session_state.chat_history, new_question, context_chunks)
        answer = generate_answer(prompt)

        # Update chat history
        st.session_state.chat_history.append((new_question, answer))

        # Display answer immediately
        st.markdown(f"**You:** {new_question}")
        st.markdown(f"**Assistant:** {answer}")

        # Confidence level display
        if confidence_pct >= 80:
            label, color = "High", "green"
        elif confidence_pct >= 60:
            label, color = "Medium", "orange"
        else:
            label, color = "Low", "red"
        st.markdown(f"**Confidence Level:** {label}")

        st.markdown("---")

        # Show most influential chunk
        most_influential_chunk = context_chunks[0]
        st.subheader("🌟 Most Influential Source")
        st.markdown(f"**Title:** {most_influential_chunk['title']}")
        st.markdown(f"**URL:** [{most_influential_chunk['url']}]({most_influential_chunk['url']})")
        st.markdown(f"**Excerpt:** {most_influential_chunk['text'][:500]}...")

        # Show sources
        st.subheader("📚 Retrieved Sources")
        for i, chunk in enumerate(context_chunks):
            with st.expander(f"Source {i+1}: {chunk['title']}"):
                st.markdown(f"**URL:** [{chunk['url']}]({chunk['url']})")
                st.markdown(f"**Preview:** {chunk['text'][:500]}...")
