# 🧠 CounselChat RAG Assistant

A Retrieval-Augmented Generation (RAG) web application designed to assist mental health counselors by retrieving real-world therapeutic responses and generating LLM-powered suggestions.

## 🚀 Features

- 🔍 **Semantic Search** over ~2,700 real counselor Q&A pairs (CounselChat dataset)
- 🤖 **LLM-Generated Guidance** based on retrieved examples
- 🧑‍⚕️ Tailored for clinicians looking for advice when treating complex patient situations
- 🖥️ Built with [Streamlit](https://streamlit.io) for an interactive web UI

## 🧰 Tech Stack

- Python 3.10
- FAISS for fast vector search
- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings
- OpenAI GPT-3.5 for response generation (via `.env`-stored API key)
- Streamlit for frontend

## 📁 Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
