import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf_files(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"✅ Created folder: {data_path}")
        print("⚠️ Please put your PDF files inside this folder and re-run the script.")
        return []

    documents = []
    if os.path.isfile(data_path) and data_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(data_path)
        documents = loader.load()
    elif os.path.isdir(data_path):
        pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"⚠️ No PDF files found in directory: {data_path}")
            return []
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
    else:
        print(f"❌ Path is neither a PDF nor a directory: {data_path}")
        return []

    print(f"✅ Total documents loaded: {len(documents)}")
    return documents
