import pandas as pd
# from langchain.schema import Document
from langchain_core.documents import Document
import os

def convert_excel_to_text(file_path: str) -> list[Document]:
    print(f"Converting file Excel: {file_path}")
    
    xls = pd.ExcelFile(file_path)
    documents = []
    file_name = os.path.basename(file_path)
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        
        table_markdown = df.to_markdown(index=False)
        
        content = (
            f"--- Starting Sheet file '{sheet_name}' in file '{file_name}' ---\n"
            f"{table_markdown}\n"
            f"--- End ---"
        )
        
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_name, "sheet": sheet_name, "type": "financial_projection"}
            )
        )
    
    print(f"Converted {len(documents)} sheets from file Excel.")
    return documents