import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import docx
import os


INDEX_PATH = "vectorstore_index"

# --- Helper functions ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        st.error("Unsupported file format")
        return ""

def build_qa_system(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)
    # Save index to disk
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

def load_qa_system():
    if os.path.exists(INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
        return vectorstore
    return None

def answer_question(vectorstore, query):
    """Generate answer from vectorstore using LLM"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt_text = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""
    
    answer = llm.invoke(prompt_text)
    
    return answer

# --- Streamlit UI ---
st.title("📄 LLM-Powered Document Q&A (RAG)")
st.write("Upload a PDF or DOCX, then ask questions about it.")

# Load existing index if available
qa_system = load_qa_system()
if qa_system:
    st.success("Existing FAISS index loaded. You can ask questions right away!")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    qa_system = build_qa_system(text)
    st.success("Document processed and FAISS index saved. You can now ask questions!")

query = st.text_input("Ask a question:")
if query and qa_system:
    with st.spinner("Generating answer..."):
        answer = answer_question(qa_system, query)
    st.markdown("### Answer")
    st.write(answer)