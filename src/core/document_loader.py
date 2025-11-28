from langchain_community.document_loaders import Docx2txtLoader, TextLoader
# from langchain.schema import Document
from langchain_core.documents import Document
from src.utils.data_converter import convert_excel_to_text
import os

RAW_DATA_PATH = "data/raw"

def load_documents() -> list[Document]:
    all_documents = []
    
    for filename in os.listdir(RAW_DATA_PATH):
        file_path = os.path.join(RAW_DATA_PATH, filename)
        
        if filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['type'] = 'plan_document'
            all_documents.extend(docs)
            print(f"Đã tải file DOCX: {filename}")
            
        elif filename.endswith(".xlsx"):
            excel_docs = convert_excel_to_text(file_path)
            all_documents.extend(excel_docs)
            print(f"Đã xử lý file XLSX: {filename}")
            
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            all_documents.extend(loader.load())
            print(f"Đã tải file TXT: {filename}")
            
        
    print(f"\nDownloaded all {len(all_documents)} raw Document.")
    return all_documents