import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.tools.financial_tool import financial_analyzer, check_schedule_conflict

ALL_AGENT_TOOLS = [financial_analyzer, check_schedule_conflict]
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là Trợ lý phân tích Agent của Ban Tổ chức. Nhiệm vụ của bạn là phân tích và trả lời các câu hỏi tài chính hoặc lịch trình bằng cách sử dụng các công cụ (Tools) được cung cấp. Nếu câu hỏi yêu cầu tính toán hoặc kiểm tra lịch, BẮT BUỘC phải gọi Tools trước khi trả lời."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # Vùng nhớ để Agent ghi lại quá trình suy luận
    ]
)

def get_agent_executor():
    print("Initializing Agent Executor...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    agent = create_tool_calling_agent(llm, ALL_AGENT_TOOLS, AGENT_PROMPT)

    agent_executor = AgentExecutor(agent=agent, tools=ALL_AGENT_TOOLS, verbose=True)
    
    print("Agent Executor Ready!!!")
    return agent_executor


def run_agent_query(query: str):
    try:
        agent_executor = get_agent_executor()
        response = agent_executor.invoke({"input": query})
        
        return {"answer": response.get("output", "Agent cannot create a response."), "context": []}
        
    except Exception as e:
        return {"answer": f"Lỗi Agent: {e}", "context": []}