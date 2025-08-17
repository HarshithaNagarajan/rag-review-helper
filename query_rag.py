# query_rag.py
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

from explore.set_api_key import set_key


def query_rag(query, vectorstore_dir="vectorstore"):
    # Load persisted vectorstore
    set_key()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

    # Create retrieval object
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create RAG chain using an OpenAI LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Get answer
    answer = qa_chain.run(query)
    return answer

if __name__ == "__main__":
    question = input("Enter your question: ")
    response = query_rag(question)
    print("\nAnswer:\n", response)
