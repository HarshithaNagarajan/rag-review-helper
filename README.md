# rag-review-helper
A personal RAG pipeline for organizing, summarizing, and reusing my monthly review meeting notes. Building this to explore LLM workflows, and improve my own review process.

This is a wip.

# Steps to run for now
Assuming you've created/have an environment that supports the libraries in `requirements.txt`:
1. Create a folder called "`data`" in this repository. Upload your pdfs there.
2. Run `build_vectorstore.py`. This will create a folder called `vectorstore`, where embeddings are stored.
3. Then run either `query_rag.py` or `app.py` to ask questions about the pdfs.

# To-do:
1. Experiment with various data storing strategies (chunking, other embeddings etc.).
2. Automate database updation when a new file is added to the `data` folder.
3. Include memory to enhance the quality of this assistant.
