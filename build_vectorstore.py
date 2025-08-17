from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from pdf_loader import load_pdfs
from explore.set_api_key import set_key

def build_vectorstore(pdf_folder="data", persist_directory="vectorstore"):
    # Load PDFs and split into chunks
    docs = load_pdfs(pdf_folder)
    print(f"Loaded {len(docs)} chunks from PDFs.")

    # Prepare texts and metadata
    texts = [doc["content"] for doc in docs]
    metadatas = [{"source": doc["source"], "chunk": doc["chunk"]} for doc in docs]

    # Create embeddings
    set_key()
    embeddings = OpenAIEmbeddings()  

    # Create Chroma vectorstore
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

    #TODO: The newest version persists automatically, check this again
    vectorstore.persist()
    print(f"Vectorstore saved to {persist_directory}.")

if __name__ == "__main__":
    build_vectorstore()


