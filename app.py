import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import os

st.set_page_config(page_title="CounselChat", layout="wide")
st.title("🧠 Mental Health Counselor Assistant")

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Loading SentenceTransformer...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index():
    return faiss.read_index("counselchat_index.faiss")


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata():
    with open("counselchat_metadata.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
index = load_faiss_index()
metadata = load_metadata()

# Retrieve top-k examples
def retrieve_examples(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [metadata[i] for i in indices[0]]

# Generate LLM suggestion
def generate_llm_response(user_query, examples):
    examples_text = "\n\n".join(
        [f"Q: {r['question']}\nA: {r['answer']}" for r in examples]
    )
    prompt = f"""You are a helpful mental health counseling assistant.
Here are past examples of therapist advice for similar situations:

{examples_text}

Now, based on the examples above, respond to this new situation from a clinician:

Q: {user_query}
A:"""

    try:
        response = openai.chat.completions.create(            
            model="gpt-3.5-turbo",  # Change to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant for mental health counselors."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


query = st.text_area("Describe your patient's situation:", height=150)

if st.button("Get Guidance") and query.strip():
    with st.spinner("Retrieving similar cases..."):
        results = retrieve_examples(query)
        for r in results:
            st.markdown("---")
            st.markdown(f"**🗂 Topic**: `{r['topic']}`")
            st.markdown(f"**🧾 Original Question**: {r['question']}")
            st.markdown(f"**📣 Patient Details**:\n> {r['question_text']}")
            st.markdown(f"**🧑‍⚕️ Therapist Advice**:\n> {r['answer']}")

    with st.spinner("Generating LLM-based suggestion..."):
        suggestion = generate_llm_response(query, results)
        st.markdown("## 🤖 LLM Suggestion")
        st.markdown(f"> {suggestion}")
