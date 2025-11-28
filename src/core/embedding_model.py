from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embedding_model():
    try:
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    except Exception as e:
        print(f"Error initialize Embedding Model: {e}")
        print("Check your Gemini_API_Key!!!")
        raise