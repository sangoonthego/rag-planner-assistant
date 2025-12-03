import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import run_rag_query

def main():
    print("-----------------------------------------------------")
    print("Planner Assistant Started!!!")
    print("-----------------------------------------------------")
    print("Warning: Need to Run scripts/run_indexing.py first.")
    print("Enter 'exit' or 'quit' if wanna end.")
    
    while True:
        # get question from user
        user_query = input("\nQuestion (Query): ")
        
        if user_query.lower() in ["exit", "quit"]:
            print("PP.")
            break
            
        if not user_query.strip():
            continue
            
        print("\nProcessing...")
        
        # run logic RAG
        result = run_rag_query(user_query)
        
        print("\n--- Assistant's Response ---")
        print(result.get("answer", "Cannot create Answer!!!"))
        
        print("\n--- Query Source (Context) ---")
        # Context là danh sách các Document (chunks)
        context_docs = result.get("context", []) 
        if context_docs:
            for i, doc in enumerate(context_docs):
                source_file = doc.metadata.get("source", "Undefined")
                print(f"[{i+1}] Source: {source_file} | Cited Content: {doc.page_content[:150]}...")
        else:
            print("No paragraph (chunk) exported.")
        print("----------------------------------")

if __name__ == "__main__":
    main()