from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .embedding_model import get_embedding_model
import os

PERSIST_DIRECTORY = "./vector_db"

def initialize_vector_store(documents: list[Document] = None):
    embedding_model = get_embedding_model()
    
    if documents:
        print(f"Starting to embed and store {len(documents)} para in ChromaDB...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        return vectordb
    else:
        print(f"Load Vector Store that existed from: {PERSIST_DIRECTORY}")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model 
        )