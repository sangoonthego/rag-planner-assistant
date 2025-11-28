import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.document_loader import load_documents
from src.utils.file_chunker import chunk_documents
from src.core.vector_store import initialize_vector_store

def run_indexing():
    print("--- Starting Indexing Process for Planner Assistant ---")
    
    raw_documents = load_documents()
    
    if not raw_documents:
        print("File not Found!!!")
        return
        
    chunks = chunk_documents(raw_documents)
    
    # embedding and store in VD
    # auto call embedding model and save in ChromaDB
    vectordb = initialize_vector_store(documents=chunks)
    
    print("\n--- Indexing Process finished successfully! ---")
    print(f"Vector Database is ready with {vectordb._collection.count()} chunks.")

if __name__ == "__main__":
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
        print("Created data/raw. Add a docs and rerun!!!")
    else:
        run_indexing()