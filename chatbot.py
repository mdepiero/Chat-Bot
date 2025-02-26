import streamlit as st
from openai import OpenAI
import PyPDF2
import tiktoken
import pandas as pd
import docx
import openpyxl
import os
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ask the user for an OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

client = OpenAI(api_key=api_key)

# Streamlit UI
st.title("Team Document Chatbot")
st.markdown("Ask questions about the stored documents!")

# Define network folder path
NETWORK_FOLDER = r"\\ad\dfs\Shared Data\MCI CI Performance Analyst\OI Automation Committee\PoW\chat_bot\documents"
MAPPED_DRIVE = "Z:"

# Function to unmap and remap Z: drive
def remap_drive(drive_letter, network_path):
    try:
        subprocess.run(["net", "use", drive_letter, "/delete"], check=False, shell=True)  # Unmap if already exists
        subprocess.run(["net", "use", drive_letter, network_path], check=True, shell=True)  # Remap
        st.success(f"✅ Network drive remapped to {drive_letter}")
    except subprocess.CalledProcessError as e:
        st.error(f"🚨 Failed to map network drive: {e}")
        st.stop()

# Always remap Z: to ensure it points to the correct location
remap_drive(MAPPED_DRIVE, NETWORK_FOLDER)

# Use the mapped drive instead of the UNC path
DOCUMENTS_FOLDER = os.path.join(MAPPED_DRIVE, '')

# Check if folder exists
st.write(f"Checking folder path: {DOCUMENTS_FOLDER}")
if not os.path.exists(DOCUMENTS_FOLDER):
    st.error(f"🚨 Error: The folder '{DOCUMENTS_FOLDER}' does not exist or is not accessible.")
    st.stop()
else:
    st.success(f"✅ Folder is accessible: {DOCUMENTS_FOLDER}")

# PDF Text Extraction
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Word Document Text Extraction
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Excel Extraction (Extract text from all sheets)
def extract_text_from_xlsx(file_path):
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    text = ""
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        text += df.to_string(index=False) + "\n"
    return text

# Text Chunking
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " "]
    )
    return text_splitter.split_text(text)

# Load all documents when app starts
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
    document_list = []

    for filename in os.listdir(DOCUMENTS_FOLDER):
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        file_type = filename.split(".")[-1]

        if file_type == "pdf":
            raw_text = extract_text_from_pdf(file_path)
        elif file_type == "docx":
            raw_text = extract_text_from_docx(file_path)
        elif file_type == "xlsx":
            raw_text = extract_text_from_xlsx(file_path)
        else:
            continue  # Skip unsupported formats

        chunks = chunk_text(raw_text)
        st.session_state.doc_chunks.extend(chunks)
        document_list.append(filename)

    st.success(f"Loaded {len(document_list)} documents and split into {len(st.session_state.doc_chunks)} chunks.")

# User Query Input
user_query = st.text_input("Ask a question about the stored documents:")

if user_query:
    # Retrieve the most relevant chunk (for now, send all for simplicity)
    context = "\n".join(st.session_state.doc_chunks[:5])  # Limit to first 5 chunks
    
    prompt = f"""
    You are an AI assistant helping answer questions based on stored documents. Here is the document context:
    
    {context}
    
    Question: {user_query}
    Answer:
    """

    # OpenAI API Call
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4 for better responses
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )

    st.write("**Answer:**", response.choices[0].message.content.strip())
