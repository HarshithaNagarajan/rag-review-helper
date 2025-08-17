import os
from typing import List, Union

# Document reading
from langchain_community.document_loaders import PyPDFLoader
# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# Vector Store
from langchain_core.vectorstores import InMemoryVectorStore

import set_api_key


def load_pdf(file_path: str) -> List:
    """
    Load a PDF file and return its pages as a list of Document objects.

    @param file_path: Path to the PDF file.
    @type file_path: str
    @return: List of pages loaded from the PDF.
    @rtype: list
    """    
    loader = PyPDFLoader(file_path)
    pages = list(loader.lazy_load())

    # print("Document loaded.")
    # print("Metadata:\n", pages[0].metadata)
    # print("First 500 characters:\n", pages[0].page_content[:500])

    return pages


def select_embeddings(embedding_type: str = "hf") -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings]:
    """
    Select and return the embedding model based on the embedding_type.

    @param embedding_type: Type of embeddings to use, either 'hf' for HuggingFace or 'openai' for OpenAI.
    @type embedding_type: str
    @return: Initialized embeddings object.
    @rtype: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
    """    
        
    if embedding_type == "openai":

        # Set API Key 
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            os.environ["OPENAI_API_KEY"] = set_api_key.get_key()

        # Declare openai embeddings
        embeddings = OpenAIEmbeddings()
        print("Using OpenAI embeddings...")

    elif embedding_type == "hf":
        # Use local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Using local HuggingFace embeddings...")

    else:
        print("Unsupported embedding type, exiting...")
        exit()

    return embeddings


def create_vector_store(pages: list, embeddings) -> InMemoryVectorStore:
    """
    Create an in-memory vector store from document pages using the specified embeddings.

    @param pages: List of document pages.
    @type pages: list
    @param embeddings: Embeddings model to convert documents to vectors.
    @type embeddings: object
    @return: Vector store containing the embedded documents.
    @rtype: InMemoryVectorStore
    """
    vector_store = InMemoryVectorStore.from_documents(pages[:3], embeddings)  
    return vector_store


def retrieve_info(query: str, vector_store: InMemoryVectorStore, k: int) -> List:
    """
    Retrieve the top-k most similar documents from the vector store for the given query.

    @param query: Search query string.
    @type query: str
    @param vector_store: Vector store containing embedded documents.
    @type vector_store: InMemoryVectorStore
    @param k: Number of top documents to retrieve.
    @type k: int
    @return: List of top-k most relevant documents.
    @rtype: list
    """
    docs = vector_store.similarity_search(query, k=k)
    return docs


def perform_search(input_file: str, query: str, embedding_type:str="hf") -> None:
    """
    Execute the full search pipeline: load PDF, select embeddings, create vector store, and retrieve information.

    @param input_file: Path to the PDF file.
    @type input_file: str
    @param query: Search query string.
    @type query: str
    @param embedding_type: Type of embeddings to use, either 'hf' for HuggingFace or 'openai' for OpenAI.
    @type embedding_type: str
    @return: None
    @rtype: None
    """    # Load document
    pages = load_pdf(input_file)
    # Choose embedding model
    embeddings = select_embeddings(embedding_type="hf")
    # Create a vector storage using the document and embedding model
    vector_store = create_vector_store(pages, embeddings)
    # Retrive information from vector store using query
    docs = retrieve_info(query, vector_store, 3)
    # Print results
    print("=== Retrieved Docs ===")
    for doc in docs:
        print(f"Page {doc.metadata['page']}: {doc.page_content[:300]}\n")
        print("===================================================")



if __name__ == "__main__":
    file_path = r"C:\Users\harsh\Downloads\2025, 06 - HN.pdf"
    query = "Data collection"
    embedding_type = "hf"
    perform_search(file_path, query, embedding_type)
    
    
