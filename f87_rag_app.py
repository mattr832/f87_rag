import json
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
import requests
import urllib.request
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# === MUST BE FIRST ===
st.set_page_config(page_title="F87 M2 AI Assistant", layout="wide")

# === CONFIG ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Runtime check for key
if not openai_api_key or openai_api_key.startswith("sk-old"):
    st.error("⚠️ Invalid or outdated OpenAI API key loaded. Please check your .env file.")
    st.stop()

# Manage connections and configs
client = OpenAI(api_key=openai_api_key)  # Replace with your secure method
GITHUB_RELEASE = 'v1.1'
CHUNKING = 'max-token400-overlay50'
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo"
TOP_K = 6
RAG_VERSION = GITHUB_RELEASE + CHUNKING + EMBED_MODEL + CHAT_MODEL


# Download index and store locally
INDEX_URL = "https://github.com/mattr832/f87_rag/releases/download/v1.1/f87_faiss.index"
INDEX_PATH = "f87_faiss.index"

if not os.path.exists(INDEX_PATH):
    with st.spinner("Setting things up..."):
        urllib.request.urlretrieve(INDEX_URL, INDEX_PATH)

# === Load Data ===
index = faiss.read_index(INDEX_PATH)
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
    history_text = "\n\n".join(f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history)
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

def send_to_slack(question, answer, confidence_label, top_chunk_url, user_comment=None):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ SLACK_WEBHOOK_URL not set.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
    f"*🚩 Flagged Response*\n"
    f"*Time:* {timestamp}\n"
    f"*Confidence:* {confidence_label}\n"
    f"*Top Source:* {top_chunk_url}\n"
    f"*Q:* {question}\n"
    f"*A:* {answer}"
)
    if user_comment:
        message += f"\n*User Comment:* {user_comment}"

    try:
        response = requests.post(webhook_url, json={"text": message})
        print(f"[Slack] Status: {response.status_code}")
        print(f"[Slack] Response: {response.text}")
    except Exception as e:
        print(f"[Slack] Failed to send alert: {e}")

def log_to_google_sheets(rag_version, question, answer, context_chunks):
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # Load from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)

    sheet = client.open("f87_rag_logs").sheet1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, rag_version, question, answer] + [chunk["text"][:500] for chunk in context_chunks]
    sheet.append_row(row)

# ====================
# === Streamlit UI ===
st.sidebar.title("Options")
if st.sidebar.button("💾 Export Chat"):
    if st.session_state.chat_history:
        chat_export = "\n\n".join(
            [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in st.session_state.chat_history]
        )
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
    st.session_state.chat_history = []

# Convert old (q, a) tuples to dicts with new structure
for i in range(len(st.session_state.chat_history)):
    item = st.session_state.chat_history[i]
    if isinstance(item, tuple):
        q, a = item
        st.session_state.chat_history[i] = {
            "question": q,
            "answer": a,
            "confidence": "Unknown",
            "top_url": "N/A"
        }

# === Clear Chat Button ===
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# === Display chat history ===
for i, entry in enumerate(st.session_state.chat_history):
    st.markdown(f"**Q{i+1}: {entry['question']}**")
    st.markdown(f"{entry['answer']}")
    # st.markdown(f"**Confidence Level:** {entry.get("confidence", "Unknown")}")

    if st.button("🚩 Report This Response", key=f"report_{i}"):
        st.session_state.report_requested = True

    if st.session_state.report_requested:
        send_to_slack(
            entry['question'],
            entry['answer'],
            entry.get("confidence", "Unknown"),
            entry.get("top_url", "N/A")
        )

        # Clear input and set transient feedback flag
        st.session_state.report_feedback_shown = True
        st.session_state.report_requested = False
        if st.session_state.get("report_feedback_shown"):
            st.success("This response has been flagged and sent for review. Thank you!")
            st.session_state.report_feedback_shown = False

    st.markdown("---")

if "report_requested" not in st.session_state:
    st.session_state.report_requested = False

# === Input area ===
new_question = st.text_input("Ask a question about the F87 M2")
if st.button("Ask") and new_question:
    with st.spinner("Retrieving and generating..."):
        # Embed the current question (with context if needed)
        full_context_query = new_question
        query_embedding = embed_query(full_context_query)
        context_chunks = retrieve_context(query_embedding)
        most_influential_chunk = context_chunks[0]

        # Compute confidence score
        distances, _ = index.search(query_embedding, TOP_K)
        weights = np.array([1 / (i + 1) for i in range(TOP_K)], dtype="float32")
        weights /= weights.sum()
        avg_similarity = float(np.dot(distances[0], weights))
        confidence_pct = round(avg_similarity * 100, 1)

        # Calculate confidence level
        if confidence_pct >= 80:
            label, color = "High", "green"
        elif confidence_pct >= 60:
            label, color = "Medium", "orange"
        else:
            label, color = "Low", "red"

        # Build prompt including history
        prompt = build_prompt(st.session_state.chat_history, new_question, context_chunks)
        answer = generate_answer(prompt)

        # Log to Google Sheets
        log_to_google_sheets(RAG_VERSION, new_question, answer, context_chunks)

        # Display new response immediately
        st.markdown(f"**You:** {new_question}")
        st.markdown(f"**Assistant:** {answer}")
        
        # Display confidence level
        st.markdown(f"**Confidence Level:** {label}")

        # Create report button
        if st.button("🚩 Report This Response", key="report_latest"):
            st.session_state.report_requested = True

        if st.session_state.report_requested:
            send_to_slack(
                new_question,
                answer,
                label,
                most_influential_chunk["url"]
            )

            # Clear input and set transient feedback flag
            st.session_state.report_feedback_shown = True
            st.session_state.report_requested = False
            if st.session_state.get("report_feedback_shown"):
                st.success("This response has been flagged and sent for review. Thank you!")
                st.session_state.report_feedback_shown = False

        # THEN append it to chat history for persistence
        st.session_state.chat_history.append({
            "question": new_question,
            "answer": answer,
            "confidence": label,
            "top_url": most_influential_chunk["url"]
        })

        st.markdown("---")

        # Show most influential chunk
        st.subheader("🌟 Most Influential Source")
        st.markdown(f"**Title:** {most_influential_chunk['title']}")
        st.markdown(f"**URL:** [{most_influential_chunk['url']}]({most_influential_chunk['url']})")
        st.markdown(f"**Excerpt:** {most_influential_chunk['text'][:500]}...")

        st.markdown("---")

        # Show sources
        st.subheader("📚 Retrieved Sources")
        for i, chunk in enumerate(context_chunks):
            with st.expander(f"Source {i+1}: {chunk['title']}"):
                st.markdown(f"**URL:** [{chunk['url']}]({chunk['url']})")
                st.markdown(f"**Preview:** {chunk['text'][:500]}...")
