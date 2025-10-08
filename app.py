import os
import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------- CONFIGURATION --------
DATA_PATH = r"data" # Use a relative path for better portability
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom prompt template for the chatbot
CUSTOM_PROMPT_TEMPLATE = """
You are HealthBot, a professional and empathetic AI health assistant. Your goal is to provide accurate, detailed, and easy-to-understand health information based on the context provided.

Context: {context}
Question: {question}

Based on the context, please provide a comprehensive and helpful answer. Structure your response clearly, using bullet points or numbered lists where appropriate to improve readability. Always maintain a supportive and caring tone.
"""

# -------- CORE FUNCTIONS --------

def set_custom_prompt():
    """Creates and returns a PromptTemplate for the QA chain."""
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

@st.cache_resource
def build_vectorstore():
    """
    Builds or loads the FAISS vector store.
    - Checks if a pre-built vector store exists.
    - If not, it loads PDFs from the DATA_PATH, splits them into chunks,
      creates embeddings, and saves the new vector store.
    - Returns the vector store object or None if no PDFs are found.
    """
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found! Please set it in your .env file.")
        return None

    os.makedirs("vectorstore", exist_ok=True)
    
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'} # Use CPU for broader compatibility
        )
        
        if os.path.exists(DB_FAISS_PATH):
            st.sidebar.success("✅ Knowledge Base loaded successfully!")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )

        if os.path.exists(DATA_PATH) and any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
            with st.sidebar.status("Building Knowledge Base...", expand=True) as status:
                st.write("Loading PDF documents...")
                loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
                
                st.write("Splitting documents into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                
                st.write("Creating vector embeddings...")
                db = FAISS.from_documents(docs, embedding_model)
                db.save_local(DB_FAISS_PATH)
                status.update(label="✅ Knowledge Base built!", state="complete", expanded=False)
            return db
        
        st.sidebar.warning("⚠️ No PDFs found in the 'data' folder. HealthBot will use its general knowledge.")
        return None
        
    except Exception as e:
        st.sidebar.error(f"❌ Error building knowledge base: {e}")
        return None

def get_retrieval_qa_chain(vectorstore):
    """Creates the RetrievalQA chain with a custom prompt."""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=2048,
        groq_api_key=GROQ_API_KEY
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )

def get_ai_response(question, vectorstore):
    """
    Gets a response from the AI.
    - Tries to use the PDF knowledge base (RAG) first.
    - If RAG fails or isn't available, it falls back to a direct LLM call.
    """
    try:
        if vectorstore:
            qa_chain = get_retrieval_qa_chain(vectorstore)
            result = qa_chain.invoke({"query": question})
            return result["result"], result["source_documents"]
        else:
            # Fallback to direct AI response without RAG
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=2048,
                groq_api_key=GROQ_API_KEY
            )
            response = llm.invoke(f"You are a helpful AI health assistant. Answer the following question: {question}")
            return response.content, None
    except Exception as e:
        error_msg = f"I apologize, but I'm facing a technical difficulty. Please try again. Error: {str(e)}"
        return error_msg, None

# -------- STREAMLIT UI --------

def display_message(msg):
    """Displays a single message in the chat with custom styling."""
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        # If the message is from the assistant and has sources, show them in an expander
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for source in msg["sources"]:
                    # Check if 'source' metadata exists before accessing it
                    source_name = source.metadata.get('source', 'Unknown Source')
                    st.info(f"Source: {os.path.basename(source_name)}")
                    st.markdown(f"> {source.page_content[:250]}...")

def main():
    st.set_page_config(
        page_title="HealthBot - Your AI Health Assistant",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a modern look
    st.markdown("""
        <style>
            /* Main chat container */
            .st-emotion-cache-1jicfl2 {
                padding-bottom: 5rem; /* Space for the sticky input */
            }

            /* Sticky input bar at the bottom */
            .st-emotion-cache-1629p8f {
                position: fixed;
                bottom: 0;
                width: 100%;
                background-color: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
                padding: 1rem 1.5rem;
                border-top: 1px solid #e0e0e0;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
                z-index: 999;
            }
            
            /* Chat message styling */
            [data-testid="stChatMessage"] {
                background-color: #f8f9fa;
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid #e9ecef;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            /* User message specific styling */
            [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][class*="user"]) {
                background-color: #e7f5ff;
                border-color: #b3d7ff;
            }

            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #f8f9fa;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🏥 HealthBot")
        st.markdown("Your trusted AI Health Assistant, powered by Llama 3.1.")
        st.divider()
        st.markdown("### 🛠️ Settings")
        
        # Build vectorstore and store in session state
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = build_vectorstore()

        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io) & [LangChain](https://langchain.com)")

    # --- Main Chat Interface ---
    st.header("HealthBot AI Assistant")

    # Medical Disclaimer
    st.warning(
        "**Disclaimer:** I am an AI assistant and not a medical professional. "
        "The information I provide is for educational purposes only. "
        "Please consult with a qualified healthcare provider for any medical advice or treatment."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        display_message(message)

    # Chat input field
    if prompt := st.chat_input("Ask me a health-related question..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})

        # Generate and display bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("HealthBot is thinking..."):
                response, sources = get_ai_response(prompt, st.session_state.vectorstore)
                
                # Use a streaming-like effect for better UX
                placeholder = st.empty()
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

        # Add the complete bot message to history (with sources)
        bot_message = {"role": "assistant", "content": full_response, "sources": sources}
        st.session_state.messages.append(bot_message)
        
        # Display sources if available
        if sources:
            with st.expander("📚 View Sources"):
                for source in sources:
                    source_name = source.metadata.get('source', 'Unknown Source')
                    st.info(f"Source: {os.path.basename(source_name)}")
                    st.markdown(f"> {source.page_content[:250]}...")


if __name__ == "__main__":
    main()
