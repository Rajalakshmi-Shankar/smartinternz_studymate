import streamlit as st
import json
import os
import tempfile

# 🔹 config.json la irundhu credentials load panrathu
with open("config.json", "r") as f:
    config = json.load(f)

ibm_api_key = config["WATSONX_API_KEY"]
ibm_project = config["WATSONX_PROJECT_ID"]
ibm_url = config["WATSONX_URL"]

# 🔹 backend functions import
from backend.ingest_index import ingest_pdf_and_index
from backend.query_llm import answer_question

st.title("📘 StudyMate - AI Powered PDF Q&A")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Temporary file save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmpf.write(uploaded_file.getbuffer())
        tmp_pdf_path = tmpf.name

    # Indexing button
    if st.button("Index PDF"):
        with st.spinner("Processing and indexing PDF..."):
            ingest_pdf_and_index(tmp_pdf_path, "indexes/default")
        st.success("✅ PDF indexed successfully!")

# Question input
question = st.text_input("Ask a question from the PDF:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            res = answer_question(
                question,
                index_dir="indexes/default",
                ibm_api_key=ibm_api_key,
                ibm_url=ibm_url,
                ibm_project=ibm_project
            )
        st.success("Answer:")
        st.write(res)
