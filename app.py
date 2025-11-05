import os
import time
import streamlit as st
import sqlite3
import hashlib
import secrets
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# -------- CONFIGURATION --------
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USER_DB_PATH = "users.db"

# Available models
MODELS = {
    "🚀 Fast (8B)": "llama-3.1-8b-instant",
    "⚡ Balanced (70B)": "llama-3.1-70b-versatile", 
    "🎯 Detailed (Maverick)": "meta-llama/llama-4-maverick-17b-128e-instruct"
}

# -------- DATABASE FUNCTIONS --------
def init_user_database():
    """Initialize user database"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_name TEXT DEFAULT 'llama-3.1-8b-instant',
            language TEXT DEFAULT 'English',
            response_style TEXT DEFAULT 'default',
            theme TEXT DEFAULT 'light',
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    return password_hash, salt

def verify_password(password, password_hash, salt):
    """Verify password against hash"""
    new_hash, _ = hash_password(password, salt)
    return new_hash == password_hash

def create_user(username, password, email=None, full_name=None):
    """Create new user"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists"
        
        password_hash, salt = hash_password(password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, salt, full_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, salt, full_name))
        
        user_id = cursor.lastrowid
        
        cursor.execute('''
            INSERT INTO user_preferences (user_id)
            VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        return True, "User created successfully"
        
    except Exception as e:
        return False, f"Error creating user: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, password_hash, salt, full_name 
            FROM users 
            WHERE username = ? AND is_active = TRUE
        ''', (username,))
        
        user = cursor.fetchone()
        if not user:
            return False, "User not found"
        
        user_id, username, password_hash, salt, full_name = user
        
        if verify_password(password, password_hash, salt):
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP 
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            conn.close()
            
            return True, {
                "user_id": user_id,
                "username": username,
                "full_name": full_name
            }
        else:
            conn.close()
            return False, "Invalid password"
            
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

def get_user_preferences(user_id):
    """Get user preferences"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_name, language, response_style, theme
            FROM user_preferences
            WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "model_name": result[0],
                "language": result[1],
                "response_style": result[2],
                "theme": result[3]
            }
        return None
        
    except Exception as e:
        return None

def update_user_preferences(user_id, preferences):
    """Update user preferences"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_preferences 
            SET model_name = ?, language = ?, response_style = ?, theme = ?
            WHERE user_id = ?
        ''', (
            preferences.get('model_name'),
            preferences.get('language'),
            preferences.get('response_style'),
            preferences.get('theme'),
            user_id
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        return False

# -------- AUTHENTICATION COMPONENTS --------
def show_login_sidebar():
    """Show login/register options in sidebar"""
    with st.sidebar:
        st.markdown("### 🔐 Account")
        
        if st.session_state.get('logged_in'):
            user = st.session_state.user
            st.success(f"✅ Logged in as **{user['username']}**")
            
            if st.button("🚪 Logout", use_container_width=True):
                logout()
        else:
            # Login/Register in expander
            with st.expander("Login / Register", expanded=False):
                tab1, tab2 = st.tabs(["Login", "Register"])
                
                with tab1:
                    login_username = st.text_input("👤 Username", key="sidebar_login_user")
                    login_password = st.text_input("🔒 Password", type="password", key="sidebar_login_pass")
                    
                    if st.button("🚀 Login", key="sidebar_login_btn", use_container_width=True):
                        if login_username and login_password:
                            success, result = authenticate_user(login_username, login_password)
                            if success:
                                st.session_state.user = result
                                st.session_state.logged_in = True
                                st.session_state.user_preferences = get_user_preferences(result["user_id"])
                                st.success(f"✅ Welcome back, {result['username']}!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {result}")
                        else:
                            st.error("❌ Please fill in all fields")
                
                with tab2:
                    reg_full_name = st.text_input("👤 Full Name", key="sidebar_reg_name")
                    reg_username = st.text_input("👤 Username", key="sidebar_reg_user")
                    reg_email = st.text_input("📧 Email", key="sidebar_reg_email")
                    reg_password = st.text_input("🔒 Password", type="password", key="sidebar_reg_pass")
                    reg_confirm = st.text_input("🔒 Confirm Password", type="password", key="sidebar_reg_confirm")
                    
                    if st.button("📝 Create Account", key="sidebar_reg_btn", use_container_width=True):
                        if all([reg_username, reg_password, reg_confirm]):
                            if reg_password == reg_confirm:
                                if len(reg_password) >= 6:
                                    success, message = create_user(reg_username, reg_password, reg_email, reg_full_name)
                                    if success:
                                        st.success("✅ Account created! Please login.")
                                    else:
                                        st.error(f"❌ {message}")
                                else:
                                    st.error("❌ Password must be at least 6 characters")
                            else:
                                st.error("❌ Passwords do not match")
                        else:
                            st.error("❌ Please fill in required fields")

def logout():
    """Logout user but preserve chat history"""
    user_backup = st.session_state.get('user')
    messages_backup = st.session_state.get('messages', [])
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Restore chat history for guest mode
    st.session_state.messages = messages_backup
    if user_backup:
        st.session_state.previous_user = user_backup
    
    st.success("✅ Logged out successfully!")
    time.sleep(1)
    st.rerun()

# -------- AI FUNCTIONS --------
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
            documents = []
            
            # Load PDFs
            if any(Path(DATA_PATH).glob("*.pdf")):
                pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
                documents.extend(pdf_loader.load())
            
            # Load text files
            if any(Path(DATA_PATH).glob("*.txt")):
                text_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
                documents.extend(text_loader.load())
            
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

def get_ai_response(question, model_name="llama-3.1-8b-instant"):
    """Get response from AI - tries PDF knowledge base first, falls back to direct AI"""
    try:
        vectorstore = build_vectorstore()
        
        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name=model_name,
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

# -------- CHAT FUNCTIONS --------
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
                <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            width:42px;height:42px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-left:12px;
                            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                    <span style='color:white;font-size:20px;'>👤</span>
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

def clear_chat():
    username = st.session_state.user['username'] if st.session_state.get('logged_in') else "Guest"
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello {username}! I'm HealthBot, your AI health assistant. How can I help you today? 😊"}
    ]

# -------- MAIN APP --------
def main_app():
    st.set_page_config(
        page_title="HealthBot - AI Health Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
        }
        .stButton>button {
            border-radius: 25px;
            padding: 10px 24px;
            font-weight: 600;
        }
        .success-box {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 12px 16px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 10px 0 20px 0;
            font-size: 14px;
        }
        .user-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .guest-info {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 12px 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        username = st.session_state.user['username'] if st.session_state.get('logged_in') else "Guest"
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello {username}! I'm HealthBot, your AI health assistant. How can I help you today? 😊"}
        ]

    # Sidebar
    with st.sidebar:
        # User/Guest info
        if st.session_state.get('logged_in'):
            user = st.session_state.user
            st.markdown(f"""
                <div class="user-info">
                    <div style='font-size: 14px;'>👤 Logged in as</div>
                    <div style='font-weight: bold; font-size: 16px;'>{user['username']}</div>
                    <div style='font-size: 12px; opacity: 0.8;'>{user.get('full_name', '')}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="guest-info">
                    <div style='font-size: 14px;'>👤 Currently browsing as</div>
                    <div style='font-weight: bold; font-size: 16px;'>Guest User</div>
                    <div style='font-size: 12px; opacity: 0.8;'>Login to save preferences</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #667eea; font-size: 1.8rem;'>🏥 HealthBot</h1>
                <p style='color: #666;'>Your AI Health Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Login/Register Section
        show_login_sidebar()
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        if st.session_state.get('logged_in') and st.session_state.get('user_preferences'):
            prefs = st.session_state.user_preferences
            
            # Model selection
            selected_model_name = st.selectbox(
                "🤖 AI Model",
                options=list(MODELS.keys()),
                index=0
            )
            selected_model = MODELS[selected_model_name]
            
            # Update preferences if changed
            if selected_model != prefs.get('model_name'):
                update_user_preferences(
                    st.session_state.user['user_id'],
                    {"model_name": selected_model}
                )
                st.session_state.user_preferences['model_name'] = selected_model
                st.success("Preferences updated!")
        else:
            # Guest model selection
            selected_model_name = st.selectbox(
                "🤖 AI Model",
                options=list(MODELS.keys()),
                index=0
            )
            selected_model = MODELS[selected_model_name]
            st.session_state.guest_model = selected_model
        
        st.markdown("### 📊 Chat Info")
        st.info(f"💬 Messages: {len(st.session_state.messages)}")
        
        st.markdown("### ⚡ Features")
        st.markdown("""
        - 🤖 AI-powered health advice
        - 📚 Medical knowledge base
        - 💬 Natural conversations
        - 🔒 Private and secure
        """)
        
        if st.button("🔄 Clear Chat", use_container_width=True, on_click=clear_chat):
            st.rerun()

    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">🏥 HealthBot AI Assistant</h1>', unsafe_allow_html=True)
        
        # System status
        vectorstore = build_vectorstore()
        if vectorstore:
            st.success("✅ **System Ready** - AI with knowledge base")
        else:
            st.warning("⚠️ **Basic Mode** - Using general AI knowledge")
        
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

        # Input area
        st.markdown("---")
        
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

        # Process input
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show user message immediately
            st.rerun()
            
            # Generate AI response
            if st.session_state.get('logged_in'):
                model_name = st.session_state.user_preferences.get('model_name', 'llama-3.1-8b-instant')
            else:
                model_name = st.session_state.get('guest_model', 'llama-3.1-8b-instant')
            
            try:
                answer = get_ai_response(user_input, model_name)
                
                # Add bot response
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered a technical issue. Please try again. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()

# -------- MAIN EXECUTION --------
def main():
    # Initialize database
    init_user_database()
    
    # Start the app directly without login requirement
    main_app()

if __name__ == "__main__":
    main()
