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
DATA_PATH = r"C:\Users\sahah\Downloads\HealthChatbot\data"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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
            return FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )

        # Create new vectorstore if PDFs exist
        if os.path.exists(DATA_PATH):
            loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)
                db = FAISS.from_documents(docs, embedding_model)
                db.save_local(DB_FAISS_PATH)
                return db
        
        # If no PDFs found, return None (will use direct AI responses)
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None

# -------- Direct AI Response (Fallback) --------
def get_direct_ai_response(question):
    """Get response directly from AI when no PDFs are available"""
    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        
        health_prompt = f"""
        You are HealthBot, a professional AI health assistant. Provide accurate, helpful health information.
        
        User Question: {question}
        
        Please provide:
        1. Clear, factual health information
        2. Practical advice and tips
        3. Helpful recommendations
        
        Provide a detailed, informative response:
        """
        
        response = llm.invoke(health_prompt)
        return response.content
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

# -------- Typing effect --------
def bot_typing(container, text, delay=0.02):
    """Typing effect for bot responses"""
    thinking_time = min(1.0, len(text) * 0.005)
    time.sleep(thinking_time)
    
    message_container = container.empty()
    typed = ""
    for char in text:
        typed += char
        message_container.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:16px;'>
                <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            width:48px;height:48px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                    <span style='color:white;font-size:22px;'>🏥</span>
                </div>
                <div style='color:#2c3e50;background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                            padding:18px 22px;border-radius:20px;max-width:70%;line-height:1.6;font-size:15px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.08);border:1px solid #e3f2fd;
                            border-left: 4px solid #2196f3;'>
                    <div style='font-weight:700;color:#1976d2;font-size:14px;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>
                        <span>HealthBot Assistant</span>
                        <span style='background:#4caf50;color:white;padding:2px 8px;border-radius:12px;font-size:11px;'>Verified</span>
                    </div>
                    <div style='color:#37474f;line-height:1.7;'>{typed}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay)

# -------- Display messages --------
def display_message(msg):
    if msg["role"] == "user":
        # User message - Right side with medical theme
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:16px;'>
                <div style='color:white;background:linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
                            padding:18px 22px;border-radius:20px;max-width:70%;line-height:1.6;font-size:15px;
                            box-shadow: 0 4px 15px rgba(67, 160, 71, 0.3);position:relative;
                            border-right: 4px solid #1b5e20;'>
                    <div style='font-weight:700;color:rgba(255,255,255,0.95);font-size:14px;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>
                        <span>You</span>
                        <span style='background:rgba(255,255,255,0.2);color:white;padding:2px 8px;border-radius:12px;font-size:11px;'>Patient</span>
                    </div>
                    <div style='color:white;line-height:1.7;'>{msg['content']}</div>
                </div>
                <div style='background:linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
                            width:48px;height:48px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-left:12px;
                            box-shadow: 0 4px 12px rgba(67, 160, 71, 0.3);'>
                    <span style='color:white;font-size:22px;'>👤</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Bot message - Left side with professional medical theme
        st.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:16px;'>
                <div style='background:linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
                            width:48px;height:48px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);'>
                    <span style='color:white;font-size:22px;'>🤖</span>
                </div>
                <div style='color:#2c3e50;background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                            padding:18px 22px;border-radius:20px;max-width:70%;line-height:1.6;font-size:15px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.08);border:1px solid #e3f2fd;
                            border-left: 4px solid #2196f3;'>
                    <div style='font-weight:700;color:#1976d2;font-size:14px;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>
                        <span>HealthBot Assistant</span>
                        <span style='background:#4caf50;color:white;padding:2px 8px;border-radius:12px;font-size:11px;'>Verified</span>
                    </div>
                    <div style='color:#37474f;line-height:1.7;'>{msg['content']}</div>
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
    st.markdown("""
        <div style='text-align:center; margin-bottom:20px;'>
            <h3 style='color:#1976d2; margin-bottom:15px;'>💡 Common Health Questions</h3>
            <p style='color:#666; font-size:14px;'>Click any question to get instant medical advice</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(
                question, 
                key=f"quick_{i}", 
                use_container_width=True,
                help="Click to ask this question"
            ):
                st.session_state.quick_question = question
                st.rerun()

# -------- Get AI Response --------
def get_ai_response(question):
    """Get response from AI - tries PDF knowledge base first, falls back to direct AI"""
    try:
        # Try to use PDF knowledge base
        vectorstore = build_vectorstore()
        
        if vectorstore:
            # Create QA chain with PDF knowledge
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1024,
                    groq_api_key=GROQ_API_KEY
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )
            
            result = qa_chain.invoke({"query": question})
            return result["result"]
        else:
            # Fallback to direct AI response
            return get_direct_ai_response(question)
            
    except Exception as e:
        # Final fallback if everything fails
        return get_direct_ai_response(question)

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="HealthBot Pro - AI Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Professional Medical Theme CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        }
        
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
            font-family: 'Arial', sans-serif;
        }
        
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
            min-height: 500px;
            max-height: 60vh;
            overflow-y: auto;
        }
        
        .status-box {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            padding: 15px 20px;
            border-radius: 15px;
            border-left: 5px solid #4caf50;
            margin: 15px 0;
            font-size: 14px;
            color: #2e7d32;
            font-weight: 500;
        }
        
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 16px 24px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            background: white;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #1976d2;
            box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
        }
        
        .stButton>button {
            border-radius: 25px;
            padding: 14px 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
            font-size: 15px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            border-right: 1px solid #e0e0e0;
        }
        
        .feature-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #1976d2;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #1976d2;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #0d47a1;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Welcome to **HealthBot Pro**! I'm your AI medical assistant, here to provide reliable health information and guidance.\n\n🔬 **I can help you with:**\n• Symptom analysis and understanding\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health education\n\n💡 **Please remember:** I provide informational support only. For medical emergencies, consult a healthcare professional immediately.\n\nWhat health concern would you like to discuss today?"}
        ]

    # Sidebar with Medical Features
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem; padding: 20px 0;'>
                <div style='background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%); 
                            width: 80px; height: 80px; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; 
                            margin: 0 auto 15px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);'>
                    <span style='color: white; font-size: 35px;'>🏥</span>
                </div>
                <h2 style='color: #1976d2; font-size: 1.5rem; margin: 0;'>HealthBot Pro</h2>
                <p style='color: #666; font-size: 0.9rem; margin: 5px 0 0 0;'>AI Medical Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Session Info")
        st.info(f"💬 **Messages:** {len(st.session_state.messages)}")
        
        st.markdown("### 🎯 Core Features")
        features = [
            "🤖 AI-Powered Diagnosis Support",
            "💊 Medication Information", 
            "🩺 Symptom Checker",
            "🥗 Health & Nutrition",
            "💪 Exercise Guidance",
            "😴 Sleep & Wellness"
        ]
        
        for feature in features:
            st.markdown(f"""
                <div class="feature-card">
                    <div style='font-weight: 600; color: #1976d2; font-size: 14px;'>{feature}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ⚕️ Medical Topics")
        st.markdown("""
        - Cold & Flu Symptoms
        - Chronic Conditions  
        - Mental Health
        - Women's Health
        - Pediatric Care
        - Emergency Signs
        """)
        
        if st.button("🔄 Start New Conversation", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Welcome to **HealthBot Pro**! I'm your AI medical assistant, here to provide reliable health information and guidance.\n\n🔬 **I can help you with:**\n• Symptom analysis and understanding\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health education\n\n💡 **Please remember:** I provide informational support only. For medical emergencies, consult a healthcare professional immediately.\n\nWhat health concern would you like to discuss today?"}
            ]
            st.rerun()

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Professional Header
        st.markdown('<h1 class="main-header">HealthBot Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your AI-Powered Medical Assistant • 24/7 Health Support</p>', unsafe_allow_html=True)
        
        # System Status
        st.markdown("""
            <div class="status-box">
                <strong>✅ SYSTEM READY</strong> | 🤖 AI Assistant Online | 🏥 Medical Database Active | 🔒 Secure & Private
            </div>
        """, unsafe_allow_html=True)
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question

        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            display_message(msg)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick replies for new chats
        if len(st.session_state.messages) <= 1:
            create_quick_replies()

        # Professional Input Area
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin-bottom: 15px;'>
                <h4 style='color: #1976d2; margin: 0;'>Ask Your Health Question</h4>
                <p style='color: #666; font-size: 14px; margin: 5px 0 0 0;'>Describe your symptoms, ask about medications, or seek health advice</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Input form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Type your health question...",
                    value=current_input_value,
                    placeholder="Example: What are the symptoms of flu? How to manage stress?",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)

        # Process input when form is submitted
        if submitted and user_input:
            # Add user message immediately
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate AI response
            with st.spinner("🔍 Analyzing your query with medical database..."):
                try:
                    answer = get_ai_response(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"⚠️ I apologize, but I'm experiencing technical difficulties. Please try again in a moment.\n\n**Error Details:** {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            # Auto-scroll to bottom
            st.markdown(
                """
                <script>
                    window.scrollTo(0, document.body.scrollHeight);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.rerun()

if __name__ == "__main__":
    main()
