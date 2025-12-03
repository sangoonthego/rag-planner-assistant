import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import run_rag_query

# --- Interface Configuration ---
st.set_page_config(page_title="Planner RAG Assistant", layout="centered")
st.title("Planner RAG Assistant")
st.caption("Planning Review Assistant using RAG and Gemini API.")

# --- Function to display results ---
def display_rag_response(query):
    with st.spinner(f"Processing query: '{query}'..."):
        result = run_rag_query(query)
        
        st.subheader("RAG Assistant's Response:")
        st.info(result.get("answer", "Cannot generate response. Please check logs."))
        
        st.subheader("Retrieved Sources (Context):")
        context_docs = result.get("context", []) 
        
        if context_docs:
            for i, doc in enumerate(context_docs):
                source_file = doc.metadata.get("source", "Undefined")
                content = doc.page_content
                
                with st.expander(f"[{i+1}] Source: {source_file}"):
                    st.code(content[:500] + "..." if len(content) > 500 else content, language='text')
        else:
            st.warning("No context documents retrieved or an error occurred.")


# --- Main Interface Loop ---

user_query = st.text_input("Enter your question:", placeholder="Enter your question")

if st.button("Query", type="primary"):
    if user_query:
        display_rag_response(user_query)
    else:
        st.error("Please enter a question.")

# Warning/Guidance Message
# st.markdown("""
# <hr style='border: 1px solid #ccc;'>
# <p style='font-size: 12px; color: gray;'>
#     **Warning:** The command <code>python -m scripts.run_indexing.py</code> must be run first.
#     <br>Ensure the <code>GEMINI_API_KEY</code> environment variable is set.
# </p>
# """, unsafe_allow_html=True)