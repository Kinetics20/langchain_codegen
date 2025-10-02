import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


# Load environment variables (e.g., OPENAI_API_KEY, TEST_MODEL)
load_dotenv()


# ======================================================================
# MODEL CONFIGURATION
# ======================================================================

# Default models
CODE_MODEL = "gpt-5-mini"       # Deterministic & enterprise-grade code
REPAIR_MODEL = "gpt-5-mini"     # Repair should be strict & consistent
DEFAULT_TEST_MODEL = "gpt-4o-mini"  # Verbose & exploratory for tests

# Allow overriding test model via environment variable
TEST_MODEL = os.getenv("TEST_MODEL", DEFAULT_TEST_MODEL)


def get_llm_code() -> ChatOpenAI:
    """Return LLM for code generation."""
    return ChatOpenAI(model=CODE_MODEL, temperature=0)


def get_llm_tests() -> ChatOpenAI:
    """Return LLM for test generation (switchable)."""
    return ChatOpenAI(model=TEST_MODEL, temperature=0)


def get_llm_repair() -> ChatOpenAI:
    """Return LLM for repairing code based on failing tests."""
    return ChatOpenAI(model=REPAIR_MODEL, temperature=0)


# ======================================================================
# STYLE KNOWLEDGE BASE (RAG)
# ======================================================================

_vectorstore: FAISS | None = None


def add_documents() -> None:
    """Initialize vectorstore with style guidelines (idempotent)."""
    global _vectorstore
    if _vectorstore is not None:
        return  # Already initialized

    style_docs: List[Document] = [
        Document(page_content="Always follow PEP8 guidelines."),
        Document(page_content="Use type hints for all function signatures."),
        Document(page_content="Every function and class must have a docstring."),
        Document(page_content="Avoid inline print statements in libraries."),
        Document(page_content="Follow Zen of Python principles."),
        Document(page_content="Keep functions short and focused."),
        Document(page_content="Prefer List, Dict, Optional imports from typing explicitly."),
        Document(page_content="Use descriptive variable names."),
        Document(page_content="Raise specific exceptions (ValueError, TypeError)."),
        Document(page_content="Write modular, testable, enterprise-grade code."),
    ]

    embeddings = OpenAIEmbeddings()
    _vectorstore = FAISS.from_documents(style_docs, embeddings)


def retrieve_style(query: str) -> List[Document]:
    """Retrieve top style guidelines relevant to the query."""
    if _vectorstore is None:
        add_documents()
    assert _vectorstore is not None
    return _vectorstore.similarity_search(query, k=3)


# ======================================================================
# DEBUG / MANUAL TEST
# ======================================================================
if __name__ == "__main__":
    print(">>> Testing style_knowledge_base_advanced.py")

    add_documents()
    results = retrieve_style("matrix multiplication")
    print("\nTop style guidelines retrieved:")
    for r in results:
        print("-", r.page_content)

    print("\nLLM setup:")
    print("Code model:", CODE_MODEL)
    print("Repair model:", REPAIR_MODEL)
    print("Test model (active):", TEST_MODEL)
