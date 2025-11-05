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

# Load environment variables
load_dotenv()

# -------- CONFIG --------
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Debug: Check API key
print(f"GROQ_API_KEY available: {bool(GROQ_API_KEY)}")

CUSTOM_PROMPT_TEMPLATE = """
You are HealthBot, an AI health assistant. Use the provided context to give detailed and accurate health information.

Context: {context}
Question: {question}

Provide a comprehensive, helpful health-related answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# -------- Vectorstore Load/Build --------
@st.cache_resource
def build_vectorstore():
    """Build or load vectorstore with caching"""
    os.makedirs("vectorstore", exist_ok=True)
    
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Check if vectorstore already exists
        if os.path.exists(DB_FAISS_PATH):
            print("Loading existing vectorstore...")
            return FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )

        # Create new vectorstore if PDFs exist
        if os.path.exists(DATA_PATH):
            print("Loading PDF documents...")
            loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if documents:
                print(f"Loaded {len(documents)} documents")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)
                db = FAISS.from_documents(docs, embedding_model)
                db.save_local(DB_FAISS_PATH)
                print("Vectorstore created successfully")
                return db
            else:
                print("No documents found in data folder")
        
        # If no PDFs found, return None (will use direct AI responses)
        print("No vectorstore available, using direct AI responses")
        return None
        
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return None

# -------- Direct AI Response (Fallback) --------
def get_direct_ai_response(question):
    """Get response directly from AI when no PDFs are available"""
    try:
        if not GROQ_API_KEY:
            return "❌ Error: GROQ_API_KEY not found. Please check your .env file"
        
        print(f"Getting direct AI response for: {question}")
        
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        
        health_prompt = f"""
        You are HealthBot, a professional AI health assistant. Provide accurate, helpful health information.
        
        User Question: {question}
        
        Please provide clear, factual health information with practical advice:
        """
        
        response = llm.invoke(health_prompt)
        print("Direct AI response received")
        return response.content
        
    except Exception as e:
        error_msg = f"❌ I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
        print(f"Error in direct AI response: {e}")
        return error_msg

# -------- Get AI Response --------
def get_ai_response(question):
    """Get response from AI - tries PDF knowledge base first, falls back to direct AI"""
    try:
        print(f"Processing question: {question}")
        
        # Try to use PDF knowledge base
        vectorstore = build_vectorstore()
        
        if vectorstore:
            print("Using vectorstore for response...")
            # Create QA chain with PDF knowledge
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1024,
                    groq_api_key=GROQ_API_KEY
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )
            
            result = qa_chain.invoke({"query": question})
            print("Vectorstore response generated")
            return result["result"]
        else:
            # Fallback to direct AI response
            print("Using direct AI response (fallback)")
            return get_direct_ai_response(question)
            
    except Exception as e:
        print(f"Error in get_ai_response: {e}")
        # Final fallback if everything fails
        return get_direct_ai_response(question)

# -------- Simple Message Display --------
def display_message(msg):
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:16px;'>
                <div style='color:white;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding:16px 20px;border-radius:20px;max-width:75%;line-height:1.6;font-size:15px;
                            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);position:relative;'>
                    <div style='font-weight:600;color:rgba(255,255,255,0.9);font-size:13px;margin-bottom:4px;'>You</div>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:16px;'>
                <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            width:42px;height:42px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                    <span style='color:white;font-size:20px;'>🤖</span>
                </div>
                <div style='color:#2c3e50;background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                            padding:16px 20px;border-radius:20px;max-width:75%;line-height:1.6;font-size:15px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);border:1px solid #e0e0e0;
                            position:relative;'>
                    <div style='font-weight:600;color:#667eea;font-size:13px;margin-bottom:4px;'>HealthBot</div>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- Quick reply buttons --------
def create_quick_replies():
    quick_questions = [
        "What are common cold symptoms?",
        "How to improve sleep quality?",
        "What foods boost immunity?",
        "Exercise recommendations for beginners?",
        "Stress management techniques?",
        "When should I see a doctor for fever?"
    ]
    
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_question = question
                st.rerun()

# -------- Clear chat function --------
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm HealthBot, your AI health assistant. I can help you with:\n\n• Understanding symptoms and conditions\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health questions\n\nWhat would you like to know about your health today? 😊"}
    ]

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="HealthBot - AI Health Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for enhanced styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 700;
        }
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        .stButton>button {
            border-radius: 25px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .success-box {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 10px 0;
            font-size: 14px;
        }
        .error-box {
            background: linear-gradient(135deg, #f8d7da 0%, #f1aeb5 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
            margin: 10px 0;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm HealthBot, your AI health assistant. I can help you with:\n\n• Understanding symptoms and conditions\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health questions\n\nWhat would you like to know about your health today? 😊"}
        ]

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #667eea; font-size: 1.8rem;'>🏥 HealthBot</h1>
                <p style='color: #666;'>Your AI Health Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # API Status
        if not GROQ_API_KEY:
            st.markdown("""
                <div class="error-box">
                    <strong>❌ API Error:</strong> GROQ_API_KEY not found
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="success-box">
                    <strong>✅ API Status:</strong> Connected
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Chat Info")
        st.info(f"💬 Messages: {len(st.session_state.messages)}")
        
        st.markdown("### ⚡ Features")
        st.markdown("""
        - 🤖 AI-powered health advice
        - 📚 Medical knowledge base
        - 💬 Natural conversations
        - 🔒 Private and secure
        """)
        
        st.markdown("### 🏥 Common Topics")
        st.markdown("""
        - Cold & Flu Symptoms
        - Sleep Improvement  
        - Nutrition & Diet
        - Exercise Guidance
        - Stress Management
        - First Aid Advice
        """)
        
        if st.button("🔄 Clear Chat", use_container_width=True, on_click=clear_chat):
            st.rerun()

        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"API Key: {'✅ Found' if GROQ_API_KEY else '❌ Missing'}")
            st.write(f"Data Path: {DATA_PATH}")
            st.write(f"Vectorstore: {DB_FAISS_PATH}")

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">🏥 HealthBot AI Assistant</h1>', unsafe_allow_html=True)
        
        # System status
        vectorstore = build_vectorstore()
        if vectorstore:
            st.markdown("""
                <div class="success-box">
                    <strong>✅ System Status:</strong> AI Health Assistant with knowledge base is ready!
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="success-box">
                    <strong>⚠️ System Status:</strong> Using general AI knowledge (upload PDFs for better responses)
                </div>
            """, unsafe_allow_html=True)
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question

        # Chat container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                display_message(msg)

        # Quick replies for new chats
        if len(st.session_state.messages) <= 1:
            create_quick_replies()

        # Input area with enhanced design
        st.markdown("---")
        
        # Use a form to handle input properly
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Ask me anything about health...",
                    value=current_input_value,
                    placeholder="Type your health question here...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                submitted = st.form_submit_button("Send 🚀", use_container_width=True)

        # Process input when form is submitted
        if submitted and user_input:
            # Add user message immediately
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show user message immediately
            st.rerun()
            
            # Generate AI response
            try:
                with st.spinner("🤖 HealthBot is thinking..."):
                    answer = get_ai_response(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ I apologize, but I encountered a technical issue. Please try again. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()

if __name__ == "__main__":
    main()
