from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Create embedding model (small but effective for code & text)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma DB in-memory for now
vectorstore = Chroma(collection_name="style_guides", embedding_function=embeddings)

def add_documents():
    """Load initial style documents (PEP8 rules, Zen of Python, etc.)"""
    docs = [
        Document(page_content="Use 4 spaces per indentation level."),
        Document(page_content="Limit all lines to a maximum of 79 characters."),
        Document(page_content="Use type hints for function signatures."),
        Document(page_content="Write clear docstrings for all public modules, functions, classes, and methods."),
        Document(page_content="Imports should usually be on separate lines."),
        Document(page_content="Use 'is' or 'is not' when comparing with None."),
    ]
    vectorstore.add_documents(docs)

def retrieve_style(query: str, k: int = 2):
    """Retrieve top-k relevant style guidelines for a given query."""
    return vectorstore.similarity_search(query, k=k)
