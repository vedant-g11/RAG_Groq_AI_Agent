import streamlit as st

# --- Patch for sqlite on Streamlit Cloud ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# 🔑 Load Groq API Key from Streamlit Cloud secrets
GROQ_API_KEY = st.secrets["groq_api_key"]

# Initialize Chroma client (local DB)
chroma_client = chromadb.Client()

# Create / get a collection
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Function to add documents (only run once to populate DB)
def add_documents(docs: list[str], ids: list[str]):
    collection.add(documents=docs, ids=ids)

# Retrieval-Augmented Generation with Groq
def rag_with_groq(user_query: str) -> str:
    # 1. Retrieve top-k documents from Chroma
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )
    matched_docs = results["documents"][0]
    doc_ids = results["ids"][0]

    # 2. Prepare context
    context = "\n".join(matched_docs)
    system_prompt = f"""
    Instructions:
    - Be concise and accurate.
    - If unsure, say "I don't know."
    Context:
    {context}
    Sources: {doc_ids}
    """

    # 3. Call Groq LLM
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-70b-8192",   # or any Groq-supported model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
    )

    return response.choices[0].message.content


# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="RAG with Groq + Chroma", page_icon="🤖", layout="centered")

st.title("🔎 RAG with Groq + ChromaDB")
st.write("Ask me anything! I'll search the knowledge base and answer using **Groq LLM**.")

# Populate DB only once
if "docs_added" not in st.session_state:
    add_documents(
        docs=[
            "Argentina won the FIFA World Cup in 2022.",
            "The capital of France is Paris.",
            "Groq provides ultra-fast inference for LLMs.",
            "Max Verstappen won the Formula 1 Drivers Championship in 2024.",
            "Red Bull Racing won the Formula 1 Constructors Championship in 2024.",
            "Cristiano Ronaldo has scored more than 850 career goals across club and country.",
            "The Great Wall of China is over 21,000 kilometers long.",
            "Python is a popular programming language created by Guido van Rossum in 1991.",
            "Mount Everest is the tallest mountain in the world, standing at 8,849 meters.",
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
            "The Amazon Rainforest produces 20% of the world's oxygen supply.",
            "India became independent from British rule on August 15, 1947.",
            "The speed of light is approximately 299,792 kilometers per second.",
            "The human brain has around 86 billion neurons.",
            "The Taj Mahal, located in Agra, India, was built by Mughal Emperor Shah Jahan.",
        ],
        ids=[
            "doc1", "doc2", "doc3", "doc4", "doc5",
            "doc6", "doc7", "doc8", "doc9", "doc10",
            "doc11", "doc12", "doc13", "doc14", "doc15"
        ]
    )
    st.session_state.docs_added = True

# Input box for user query
user_query = st.text_input("💬 Enter your query:")

if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question first!")
    else:
        with st.spinner("Thinking... 🤔"):
            answer = rag_with_groq(user_query)
        st.success("Answer:")
        st.write(answer)
