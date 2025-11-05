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
        
        if os.path.exists(DB_FAISS_PATH):
            return FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )

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
        
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None

# -------- Direct AI Response --------
def get_direct_ai_response(question):
    """Get response directly from AI"""
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
        
        Provide a detailed, informative response:
        """
        
        response = llm.invoke(health_prompt)
        return response.content
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

# -------- Get AI Response --------
def get_ai_response(question):
    """Get response from AI"""
    try:
        vectorstore = build_vectorstore()
        
        if vectorstore:
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
            return get_direct_ai_response(question)
            
    except Exception as e:
        return get_direct_ai_response(question)

# -------- MODERN UI COMPONENTS --------
def display_message(msg):
    """Modern message display"""
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='
                display: flex;
                justify-content: flex-end;
                margin: 15px 0;
                animation: fadeIn 0.5s ease-in;
            '>
                <div style='
                    background: linear-gradient(45deg, #FF6B6B, #FF8E53);
                    color: white;
                    padding: 12px 18px;
                    border-radius: 18px 18px 4px 18px;
                    max-width: 70%;
                    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
                    border: 1px solid rgba(255,255,255,0.2);
                '>
                    <div style='font-size: 12px; opacity: 0.9; margin-bottom: 5px;'>You</div>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='
                display: flex;
                align-items: start;
                margin: 15px 0;
                animation: slideIn 0.5s ease-out;
            '>
                <div style='
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    padding: 12px 18px;
                    border-radius: 18px 18px 18px 4px;
                    max-width: 70%;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    border: 1px solid rgba(255,255,255,0.2);
                '>
                    <div style='font-size: 12px; opacity: 0.9; margin-bottom: 5px;'>
                        <span style='font-weight: bold;'>🤖 HealthBot</span>
                    </div>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_health_topics():
    """Health topics with icons"""
    topics = [
        {"icon": "🤒", "title": "Symptoms Check", "desc": "Understand your symptoms"},
        {"icon": "💊", "title": "Medications", "desc": "Drug information & side effects"},
        {"icon": "🏃", "title": "Fitness", "desc": "Exercise & workout plans"},
        {"icon": "🥗", "title": "Nutrition", "desc": "Diet & healthy eating"},
        {"icon": "😴", "title": "Sleep", "desc": "Sleep quality improvement"},
        {"icon": "🧘", "title": "Mental Health", "desc": "Stress & anxiety management"}
    ]
    
    st.markdown("### 🌟 Health Topics")
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            if st.button(
                f"{topic['icon']} {topic['title']}",
                key=f"topic_{i}",
                help=topic['desc'],
                use_container_width=True
            ):
                st.session_state.quick_question = f"Tell me about {topic['title'].lower()}"
                st.rerun()

def show_typing_animation():
    """Modern typing animation"""
    with st.empty():
        st.markdown(
            """
            <div style='
                display: flex;
                align-items: center;
                padding: 15px;
                background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                border-radius: 15px;
                margin: 10px 0;
                animation: pulse 2s infinite;
            '>
                <div style='
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 15px;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                '>
                    <span style='color: white; font-size: 18px;'>⚕️</span>
                </div>
                <div style='flex: 1;'>
                    <div style='font-weight: 600; color: #667eea; margin-bottom: 5px;'>HealthBot is thinking</div>
                    <div style='display: flex; gap: 5px;'>
                        <div style='width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite;'></div>
                        <div style='width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite 0.2s;'></div>
                        <div style='width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite 0.4s;'></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(2)

# -------- MAIN APP --------
def main():
    st.set_page_config(
        page_title="HealthBot Pro - AI Health Assistant",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern CSS with animations
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .main-header {
            font-size: 3rem;
            background: linear-gradient(45deg, #FF6B6B, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
            text-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .feature-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin: 8px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            background: #f8f9fa;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .stButton>button {
            border-radius: 25px;
            padding: 12px 24px;
            font-weight: 600;
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm **HealthBot Pro**, your advanced AI health assistant. I'm here to provide you with accurate health information, symptom analysis, medication guidance, and wellness tips. How can I assist you with your health today?"}
        ]

    # Sidebar - Modern Design
    with st.sidebar:
        # Header with gradient
        st.markdown("""
            <div style='
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 25px 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 25px;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            '>
                <h1 style='margin: 0; font-size: 1.8rem;'>⚕️ HealthBot Pro</h1>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>AI Health Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Status Card
        vectorstore = build_vectorstore()
        status_text = "✅ **Enhanced Mode** with Medical Database" if vectorstore else "⚡ **Basic Mode** - General AI Knowledge"
        st.markdown(f"""
            <div class='status-card'>
                <div style='font-size: 14px; opacity: 0.9;'>System Status</div>
                <div style='font-size: 16px; font-weight: 600;'>{status_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Chat Statistics
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Status", "Active")
        
        # Features
        st.markdown("### 🚀 Features")
        features = [
            "🤖 AI-Powered Health Advice",
            "📚 Medical Knowledge Base", 
            "💊 Medication Information",
            "🏃 Fitness Guidance",
            "🥗 Nutrition Plans",
            "😴 Sleep Analysis"
        ]
        
        for feature in features:
            st.markdown(f"<div class='feature-card'>{feature}</div>", unsafe_allow_html=True)
        
        # Clear Chat
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Conversation cleared! How can I help you with your health today?"}
            ]
            st.rerun()

    # Main Content Area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Modern Header
        st.markdown('<h1 class="main-header">HealthBot Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your Advanced AI Health Assistant</p>', unsafe_allow_html=True)
        
        # Health Topics Grid
        create_health_topics()
        
        st.markdown("---")
        
        # Chat Container with modern design
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                display_message(msg)
        
        # Input Area - Modern Design
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                padding: 20px;
                border-radius: 20px;
                margin-top: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            '>
        """, unsafe_allow_html=True)
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question
        
        # Modern Input Form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "💬 Ask your health question...",
                    value=current_input_value,
                    placeholder="Describe symptoms, ask about medications, or seek health advice...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                submitted = st.form_submit_button(
                    "Send 🚀", 
                    use_container_width=True,
                    type="primary"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process input
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show user message immediately
            st.rerun()
            
            # Show typing animation
            show_typing_animation()
            
            # Generate AI response
            try:
                answer = get_ai_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"❌ I apologize, but I encountered a technical issue. Please try again. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

if __name__ == "__main__":
    main()
