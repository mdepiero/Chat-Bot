import streamlit as st
from openai import OpenAI
import PyPDF2
import tiktoken
import pandas as pd
import docx
import openpyxl
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ask the user for an OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

client = OpenAI(api_key=api_key)

# Streamlit UI
st.title("Team Document Chatbot")
st.markdown("Upload a document and ask questions about it!")

# File Uploader
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "xlsx"])

# Tokenizer for counting tokens
def num_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Adjust based on model
    return len(enc.encode(text))

# PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Word Document Text Extraction
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Excel Extraction (Extract text from all sheets)
def extract_text_from_xlsx(uploaded_file):
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
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

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    
    if file_type == "pdf":
        raw_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        raw_text = extract_text_from_docx(uploaded_file)
    elif file_type == "xlsx":
        raw_text = extract_text_from_xlsx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
    
    chunks = chunk_text(raw_text)
    st.success(f"Extracted text and split into {len(chunks)} chunks.")

    # Store chunks in session state to persist across interactions
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = chunks

    # User Query Input
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        # Retrieve the most relevant chunk (for now, send all for simplicity)
        context = "\n".join(st.session_state.doc_chunks[:5])  # Limit to first 5 chunks to fit model limit
        
        prompt = f"""
        You are an AI assistant helping answer questions based on a document. Here is the document context:
        
        {context}
        
        Question: {user_query}
        Answer:
        """
        
        # OpenAI API Call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change to gpt-4-turbo if needed
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        
        st.write("**Answer:**", response.choices[0].message.content.strip())