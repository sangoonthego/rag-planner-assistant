# from langchain.schema import Document
from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[Document]) -> list[Document]:
    print(f"Starting to chunk {len(documents)} docs...")
    
    # chunk_size: Kích thước tối đa của một đoạn (token/ký tự)
    # chunk_overlap: Độ chồng lấn giữa các đoạn để đảm bảo ngữ cảnh không bị mất
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400,
        length_function=len
    )
    
    # Thực hiện chia nhỏ
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} paragraphs (chunks).")
    return chunks