import streamlit as st
import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb


# Setup 
load_dotenv("key.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Amazon Feedback Assistant", page_icon="💬")

st.title("Amazon Review Insights Chatbot")
st.write("Ask questions based on analyzed customer feedback (complaints, suggestions, and positives).")

# Load RAG database 
client = chromadb.PersistentClient(path="./rag_storage")
embedding_func = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="amazon_feedback",
    embedding_function=embedding_func
)


# Define the Q&A function 
def ask_question(query):
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_docs = [doc for doc in results["documents"][0]]
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are an assistant analyzing Amazon product feedback.
    Use the context below to answer the user's question accurately.

    Context:
    {context}

    Question: {query}

    Answer in a short, clear paragraph.
    """

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

# User input 
user_input = st.text_input("Ask me anything about the product feedback:")

if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("Analyzing with RAG..."):
            answer = ask_question(user_input)
            st.success("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question first.")
