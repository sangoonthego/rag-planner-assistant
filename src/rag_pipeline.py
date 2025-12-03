import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.core.vector_store import initialize_vector_store

SYSTEM_PROMPT = (
    "Bạn là Trợ lý Duyệt Kế hoạch chuyên nghiệp. "
    "Nhiệm vụ của bạn là phân tích các đoạn văn bản (context) được cung cấp từ các tài liệu kế hoạch. "
    "Hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và dựa trên sự thật, sử dụng hoàn toàn ngữ cảnh đã cho."
    "\n\nContext:\n{context}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain_and_retriever():
    print("Initializing RAG (LCEL)...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # load Vector Store and init Retriever
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        retriever 
        | RunnableLambda(format_docs) 
        | ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT.replace("{context}", "{context}")), 
                                            ("human", "{input}")]) # 3c. Tạo Prompt
        | llm 
        | str 
    )

    rag_chain_final = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(retriever.invoke(x['input'])))) 
        | ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
        | llm
        | str
    )
    
    print("RAG Chain Ready!!!")
    return rag_chain_final, retriever

def run_rag_query(query: str):
    try:
        rag_chain, retriever = get_rag_chain_and_retriever()
        context_docs = retriever.invoke(query)
        answer = rag_chain.invoke({"input": query})
        return {"answer": answer, "context": context_docs}
        
    except Exception as e:
        return {"answer": f"Error occurs when run RAG Process: {e}", "context": []}