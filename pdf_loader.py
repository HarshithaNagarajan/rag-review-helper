import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_folder="data", chunk_size=1000, chunk_overlap=100):
    #TODO: Experiment with different chunking strategies

    all_docs = []

    for file_name in os.listdir(pdf_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    all_docs.append({
                        "content": chunk,
                        "source": file_name,
                        "chunk": i
                    })
    return all_docs

if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} chunks from PDFs!")
