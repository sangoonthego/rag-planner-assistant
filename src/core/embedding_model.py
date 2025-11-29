from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embedding_model():
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY was not set up in .env .")

    try:
        # Sử dụng GoogleGenerativeAIEmbeddings và truyền khóa API rõ ràng
        return GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key 
        )
    except Exception as e:
        print(f"Error init embedding model: {e}")
        print("Check your API Key if set first.")
        raise