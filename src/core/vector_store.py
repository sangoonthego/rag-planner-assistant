from langchain_community.vectorstores import Chroma
# from langchain.schema import Document
from langchain_core.documents import Document
from .embedding_model import get_embedding_model

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
        vectordb.persist()
        print(f"Vector Store Saved in folder: {PERSIST_DIRECTORY}")
        return vectordb
    else:
        print(f"Download existed Vector Store from: {PERSIST_DIRECTORY}")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding=embedding_model
        )