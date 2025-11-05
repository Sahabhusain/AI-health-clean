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
import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor for multiple file types"""
    
    def __init__(self, data_path: str = "data", vectorstore_path: str = "vectorstore/db_faiss"):
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        self.supported_extensions = {'.pdf', '.txt', '.md'}
    
    def create_data_directory(self) -> bool:
        """Create data directory if it doesn't exist"""
        try:
            os.makedirs(self.data_path, exist_ok=True)
            logger.info(f"✅ Data directory created/verified: {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating data directory: {e}")
            return False
    
    def get_available_files(self) -> List[str]:
        """Get list of available files in data directory"""
        if not os.path.exists(self.data_path):
            return []
        
        files = []
        for ext in self.supported_extensions:
            files.extend(list(Path(self.data_path).glob(f"*{ext}")))
        
        return [str(f) for f in files]
    
    def load_documents(self) -> List:
        """
        Load documents from data directory with enhanced error handling
        
        Returns:
            List of loaded documents
        """
        if not self.create_data_directory():
            return []
        
        # Check if data path is a single file
        if os.path.isfile(self.data_path) and Path(self.data_path).suffix.lower() in self.supported_extensions:
            return self._load_single_file(self.data_path)
        
        # Load from directory
        elif os.path.isdir(self.data_path):
            return self._load_from_directory()
        
        else:
            logger.error(f"❌ Invalid path: {self.data_path}")
            return []
    
    def _load_single_file(self, file_path: str) -> List:
        """Load a single file"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.error(f"❌ Unsupported file type: {file_ext}")
                return []
            
            documents = loader.load()
            logger.info(f"✅ Loaded {len(documents)} pages from: {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading file {file_path}: {e}")
            return []
    
    def _load_from_directory(self) -> List:
        """Load all supported files from directory"""
        all_documents = []
        
        # Load PDF files
        pdf_files = list(Path(self.data_path).glob("*.pdf"))
        if pdf_files:
            try:
                pdf_loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
                pdf_documents = pdf_loader.load()
                all_documents.extend(pdf_documents)
                logger.info(f"✅ Loaded {len(pdf_documents)} pages from {len(pdf_files)} PDF files")
            except Exception as e:
                logger.error(f"❌ Error loading PDF files: {e}")
        
        # Load text files
        text_files = list(Path(self.data_path).glob("*.txt")) + list(Path(self.data_path).glob("*.md"))
        if text_files:
            try:
                text_loader = DirectoryLoader(self.data_path, glob="*.txt", loader_cls=TextLoader)
                text_documents = text_loader.load()
                all_documents.extend(text_documents)
                logger.info(f"✅ Loaded {len(text_documents)} chunks from {len(text_files)} text files")
            except Exception as e:
                logger.error(f"❌ Error loading text files: {e}")
        
        if not all_documents:
            available_files = self.get_available_files()
            if available_files:
                logger.info(f"📁 Available files: {', '.join([os.path.basename(f) for f in available_files])}")
            else:
                logger.warning(f"⚠️ No supported files found in: {self.data_path}")
                logger.info(f"💡 Supported formats: {', '.join(self.supported_extensions)}")
        
        return all_documents
    
    def chunk_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """Split documents into chunks for processing"""
        if not documents:
            logger.warning("⚠️ No documents to chunk")
            return []
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error chunking documents: {e}")
            return []
    
    def create_vectorstore(self, documents: List) -> Optional[FAISS]:
        """Create FAISS vectorstore from documents"""
        if not documents:
            logger.error("❌ No documents provided for vectorstore creation")
            return None
        
        try:
            # Create vectorstore directory
            os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
            
            # Initialize embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            
            # Create vectorstore
            vectorstore = FAISS.from_documents(documents, embedding_model)
            
            # Save vectorstore
            vectorstore.save_local(self.vectorstore_path)
            
            logger.info(f"✅ Vectorstore created and saved: {self.vectorstore_path}")
            logger.info(f"📊 Total documents in vectorstore: {vectorstore.index.ntotal}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            return None
    
    def load_existing_vectorstore(self) -> Optional[FAISS]:
        """Load existing vectorstore if available"""
        if not os.path.exists(self.vectorstore_path):
            logger.warning(f"⚠️ Vectorstore not found: {self.vectorstore_path}")
            return None
        
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            
            vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"✅ Loaded existing vectorstore: {self.vectorstore_path}")
            logger.info(f"📊 Documents in vectorstore: {vectorstore.index.ntotal}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {e}")
            return None

# ========== LEGACY FUNCTION FOR BACKWARD COMPATIBILITY ==========
def load_pdf_files(data_path: str = "data") -> List:
    """
    Legacy function - Load PDF files from directory
    
    Args:
        data_path: Path to PDF file or directory containing PDFs
    
    Returns:
        List of loaded documents
    """
    processor = DocumentProcessor(data_path)
    return processor.load_documents()

# ========== MAIN EXECUTION ==========
def main():
    """Test the document processor"""
    processor = DocumentProcessor()
    
    print("🚀 HealthBot Document Processor")
    print("=" * 40)
    
    # Check available files
    files = processor.get_available_files()
    if files:
        print(f"📁 Found {len(files)} files:")
        for file in files:
            print(f"  - {os.path.basename(file)}")
    else:
        print("📁 No files found in data directory")
    
    # Load documents
    print("\n📥 Loading documents...")
    documents = processor.load_documents()
    
    if documents:
        print(f"✅ Successfully loaded {len(documents)} document chunks")
        
        # Create vectorstore
        print("\n🔄 Creating vectorstore...")
        vectorstore = processor.create_vectorstore(documents)
        
        if vectorstore:
            print("🎉 Vectorstore creation completed!")
        else:
            print("❌ Failed to create vectorstore")
    else:
        print("❌ No documents loaded")

if __name__ == "__main__":
    main()
