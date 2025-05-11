# 🏎️ F87 M2 Forum Assistant — RAG-Powered Chat App

This Streamlit app is a Retrieval-Augmented Generation (RAG) assistant trained on scraped threads from the Bimmerpost F87 M2 forum. It allows users to ask technical or community questions and receive grounded, forum-based answers.

---

## 🚀 Features

- ✅ Multi-turn conversation memory (chat-style interface)
- 🔍 Retrieval using FAISS vector search
- 🧠 Answer generation via OpenAI's GPT-3.5
- 📚 Transparent source citation for each answer
- 🔐 Secure `.env`-based local API key loading
- ☁️ Secrets-compatible with Streamlit Cloud

---

## 🧱 Tech Stack

- Streamlit
- OpenAI API
- FAISS
- Transformers (HuggingFace)
- Python 3.8+

---

## 🧪 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/f87-m2-chat
cd f87-m2-chat
```

### 2. Create .env
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run f87_rag_app.py
```

📁 Project Structure
```graphql
f87-m2-chat/
├── f87_rag_app.py          # Main Streamlit app
├── f87_faiss.index         # FAISS vector index of forum content
├── f87_metadata.json       # Metadata for each chunk
├── requirements.txt        # Python dependencies
└── .streamlit/
    ├── config.toml         # Streamlit server config
```

📖 How It Works (RAG Pipeline)
Embeds the user question using text-embedding-3-small

Retrieves the top-k similar forum chunks from FAISS

Builds a prompt using:

Retrieved chunks

Previous Q&A turns (chat history)

Sends the prompt to GPT-3.5

Displays the answer with cited sources

