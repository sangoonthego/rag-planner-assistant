from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from src.core.vector_store import initialize_vector_store
import os
import sys
# Thiết lập khóa API cho Gemini (Nên được thực hiện trong môi trường)
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"

# Định nghĩa Prompt hệ thống để hướng dẫn LLM
# Prompt này RẤT QUAN TRỌNG, nó định hình vai trò và đầu ra của LLM
SYSTEM_PROMPT = (
    "Bạn là Trợ lý Duyệt Kế hoạch chuyên nghiệp. "
    "Nhiệm vụ của bạn là phân tích các đoạn văn bản (context) được cung cấp từ các tài liệu kế hoạch. "
    "Hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và dựa trên sự thật, sử dụng hoàn toàn ngữ cảnh đã cho. "
    "Không được sử dụng kiến thức bên ngoài. Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rõ ràng 'Tôi không tìm thấy thông tin này trong các tài liệu hiện có.'"
    "\n\nContext:\n{context}"
)

def get_rag_chain():
    print("Đang khởi tạo chuỗi RAG...")
    
    # init LLM (Generator)
    # use gemini-2.5-flash 'cause fast speed and good thinking
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # define Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    # load Vector Store and init Retriever
    # load Vector Store that created at indexing step
    vector_store = initialize_vector_store()
    # init Retriever: allow to find top 3 most related para
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Build (Combine Documents Chain)
    # get context and question to create prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # (Retrieval Chain)
    # combine Retriever and Document Chain for creating RAG Pipeline
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG Chain Ready!!!.")
    return retrieval_chain

def run_rag_query(query: str):
    try:
        rag_chain = get_rag_chain()
        
        response = rag_chain.invoke({"input": query})
        
        return response
    except Exception as e:
        return {"answer": f"Error in RAG Process: {e}", "context": []}