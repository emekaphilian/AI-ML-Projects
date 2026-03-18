from langchain_community.text_splitter import RecursiveCharacterTextSplitter   # ✅ text splitter
from langchain_openai import OpenAIEmbeddings, OpenAI                          # ✅ OpenAI models + embeddings
from langchain_community.vectorstores import FAISS                             # ✅ FAISS vector store
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import docx
import os

INDEX_PATH = "vectorstore_index"

def extract_text(file_path: str) -> str:
    """Extract text from PDF or DOCX file."""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

def build_qa_system(text: str):
    """Build FAISS index and QA system from document text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4"),
        retriever=vectorstore.as_retriever()
    )
    return qa

def load_qa_system():
    """Load FAISS index if available."""
    if os.path.exists(INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model="gpt-4"),
            retriever=vectorstore.as_retriever()
        )
        return qa
    return None