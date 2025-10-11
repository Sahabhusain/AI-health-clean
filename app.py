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

# -------- LOGIN SYSTEM --------
def login_system():
    """Handle login and registration"""
    st.set_page_config(
        page_title="HealthBot - Login",
        page_icon="🔐",
        layout="centered"
    )
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .login-header {
            text-align: center;
            margin-bottom: 40px;
        }
        .login-card {
            background: white;
            padding: 40px 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .stButton>button {
            border-radius: 25px;
            padding: 12px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            margin-top: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 12px 15px;
            margin-bottom: 15px;
        }
        .register-link {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1 style='font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>🏥 HealthBot</h1>
                <p style='color: #666; font-size: 1.1rem;'>Your AI Health Assistant</p>
            </div>
            
            <div class="login-card">
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'users' not in st.session_state:
        st.session_state.users = {}
    if 'gmail_users' not in st.session_state:
        st.session_state.gmail_users = {}
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # If user is already logged in, redirect to HealthBot
    if st.session_state.get('logged_in', False):
        st.success("✅ Already logged in! Redirecting...")
        time.sleep(1)
        # Switch to HealthBot page
        st.switch_page("pages/healthbot.py")
        return

    # Show registration form if toggle is True
    if st.session_state.show_register:
        show_registration_form()
    else:
        show_login_form()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Demo credentials info
    st.markdown("""
        <div style='text-align: center; margin-top: 20px; color: #666;'>
            <p><strong>Demo Credentials:</strong></p>
            <p>Username: <code>user</code> | Password: <code>password123</code></p>
            <p>Username: <code>admin</code> | Password: <code>admin123</code></p>
            <p>Username: <code>test</code> | Password: <code>test123</code></p>
            <p>Any Gmail | Password: <code>gmail123</code></p>
        </div>
    """, unsafe_allow_html=True)

def show_registration_form():
    """Show registration form"""
    st.subheader("📝 Create Account")
    
    with st.form("register_form"):
        register_method = st.radio(
            "Choose account type:",
            ["Username/Password", "Gmail"],
            horizontal=True,
            key="register_method"
        )
        
        if register_method == "Username/Password":
            new_username = st.text_input("👤 Choose Username", placeholder="Enter your username", key="reg_username")
            new_password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="reg_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            
        else:  # Gmail registration
            new_gmail = st.text_input("📧 Gmail Address", placeholder="Enter your Gmail address", key="reg_gmail")
            new_password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="reg_gmail_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password", key="reg_gmail_confirm")
        
        col1, col2 = st.columns(2)
        with col1:
            register_submitted = st.form_submit_button("📝 Register", use_container_width=True)
        with col2:
            back_to_login = st.form_submit_button("🔙 Back to Login", use_container_width=True)
        
        if back_to_login:
            st.session_state.show_register = False
            st.rerun()
        
        if register_submitted:
            if register_method == "Username/Password":
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        # Check if username already exists
                        if new_username in st.session_state.users:
                            st.error("❌ Username already exists! Please choose a different username.")
                        else:
                            # Store user data
                            st.session_state.users[new_username] = new_password
                            st.success("✅ Account created successfully! Please login with your new credentials.")
                            time.sleep(2)
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("❌ Passwords do not match! Please try again.")
                else:
                    st.error("❌ Please fill in all fields!")
            
            else:  # Gmail registration
                if new_gmail and new_password and confirm_password:
                    if "@gmail.com" not in new_gmail:
                        st.error("❌ Please enter a valid Gmail address!")
                    elif new_password == confirm_password:
                        # Check if Gmail already exists
                        if new_gmail in st.session_state.gmail_users:
                            st.error("❌ Gmail already registered! Please use a different Gmail.")
                        else:
                            # Store Gmail user data
                            st.session_state.gmail_users[new_gmail] = new_password
                            st.success("✅ Account created successfully! Please login with your Gmail.")
                            time.sleep(2)
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("❌ Passwords do not match! Please try again.")
                else:
                    st.error("❌ Please fill in all fields!")

def show_login_form():
    """Show login form"""
    st.subheader("🔐 Login to HealthBot")
    
    with st.form("login_form"):
        login_method = st.radio(
            "Choose login method:",
            ["Username/Password", "Gmail"],
            horizontal=True,
            key="login_method"
        )
        
        if login_method == "Username/Password":
            username = st.text_input("👤 Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_password")
            
        else:  # Gmail login
            gmail = st.text_input("📧 Gmail", placeholder="Enter your Gmail address", key="login_gmail")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_gmail_password")
        
        login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if login_submitted:
            if login_method == "Username/Password":
                if username and password:
                    # Check demo credentials first
                    demo_credentials = {
                        "user": "password123",
                        "admin": "admin123",
                        "test": "test123"
                    }
                    
                    # Check registered users
                    registered_users = st.session_state.users
                    
                    if username in demo_credentials and password == demo_credentials[username]:
                        handle_successful_login(username, "username")
                    elif username in registered_users and registered_users[username] == password:
                        handle_successful_login(username, "username")
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.error("❌ Please fill in all fields!")
            
            else:  # Gmail login
                if gmail and password:
                    # Check demo Gmail
                    demo_gmail_password = "gmail123"
                    
                    # Check registered Gmail users
                    registered_gmail_users = st.session_state.gmail_users
                    
                    if "@gmail.com" in gmail and password == demo_gmail_password:
                        handle_successful_login(gmail, "gmail")
                    elif gmail in registered_gmail_users and registered_gmail_users[gmail] == password:
                        handle_successful_login(gmail, "gmail")
                    else:
                        st.error("❌ Invalid Gmail or password!")
                else:
                    st.error("❌ Please fill in all fields!")
    
    # Register button below login form
    st.markdown("---")
    st.markdown('<div class="register-link">', unsafe_allow_html=True)
    st.write("Don't have an account?")
    if st.button("📝 Create New Account", use_container_width=True, key="register_btn"):
        st.session_state.show_register = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def handle_successful_login(username, user_type):
    """Handle successful login"""
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.user_type = user_type
    st.success(f"✅ Welcome {'back' if user_type == 'username' else ''}, {username}!")
    
    # Show redirect message
    st.info("🔄 Redirecting to HealthBot...")
    time.sleep(2)
    
    # Switch to HealthBot page
    st.switch_page("pages/healthbot.py")

# -------- HEALTHBOT CONFIG --------
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

# -------- Realistic Typing Effect --------
def bot_typing(container, text, delay=0.03):
    """Realistic typing effect that feels like a real bot"""
    thinking_time = min(1.5, len(text) * 0.01)
    time.sleep(thinking_time)
    
    # Show typing indicator
    with container:
        typing_indicator = st.empty()
        typing_indicator.markdown(
            """
            <div style='display:flex; align-items:flex-start; margin-bottom:12px;'>
                <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            width:42px;height:42px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                    <span style='color:white;font-size:20px;'>🤖</span>
                </div>
                <div style='color:#666;background:#f8f9fa;padding:12px 16px;border-radius:18px;
                            border:1px solid #e9ecef;font-style:italic;'>
                    HealthBot is typing...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1)
    
    typing_indicator.empty()
    
    # Type out the actual message with realistic pacing
    message_container = container.empty()
    typed = ""
    
    # Split into words for more natural typing
    words = text.split()
    
    for i, word in enumerate(words):
        typed += word + " "
        
        message_container.markdown(
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
                    {typed.strip()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Realistic typing speed variations
        if i < len(words) - 1:
            if word.endswith(('.', '!', '?')):
                time.sleep(0.3)  # Longer pause after sentences
            elif len(word) > 6:
                time.sleep(0.15)  # Slightly longer for long words
            else:
                time.sleep(0.08 + random.random() * 0.05)  # Natural variation

# -------- Display messages in sequence --------
def display_message(msg):
    if msg["role"] == "user":
        # User message - Right side
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
        # Bot message - Left side
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
        {"role": "assistant", "content": f"Hello {st.session_state.username}! I'm HealthBot, your AI health assistant. I can help you with:\n\n• Understanding symptoms and conditions\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health questions\n\nWhat would you like to know about your health today? 😊"}
    ]

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

# -------- HealthBot Main App --------
def healthbot_app():
    st.set_page_config(
        page_title="HealthBot - AI Health Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        st.error("🔐 Please login first!")
        st.info("Redirecting to login page...")
        time.sleep(2)
        st.switch_page("app.py")
        return

    # Custom CSS for enhanced styling
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
            padding: 12px 15px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 5px 0 15px 0;
            font-size: 14px;
        }
        
        /* Remove white gaps and extra spacing */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .main .block-container {
            padding-top: 1rem;
        }
        
        .user-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello {st.session_state.username}! I'm HealthBot, your AI health assistant. I can help you with:\n\n• Understanding symptoms and conditions\n• Medication information and side effects\n• Healthy lifestyle recommendations\n• Preventive care advice\n• General health questions\n\nWhat would you like to know about your health today? 😊"}
        ]

    # Sidebar with user info
    with st.sidebar:
        # User info card
        st.markdown(f"""
            <div class="user-info">
                <div style='font-size: 14px;'>👤 Logged in as</div>
                <div style='font-weight: bold; font-size: 16px;'>{st.session_state.username}</div>
                <div style='font-size: 12px; opacity: 0.8;'>{st.session_state.user_type.title()} User</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #667eea; font-size: 1.8rem;'>🏥 HealthBot</h1>
                <p style='color: #666;'>Your AI Health Assistant</p>
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
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True, on_click=clear_chat):
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                # Clear session state and redirect to login
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Logged out successfully!")
                time.sleep(1)
                st.switch_page("app.py")

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">🏥 HealthBot AI Assistant</h1>', unsafe_allow_html=True)
        
        # System status - No white gap
        st.markdown("""
        <div class="success-box">
            <strong>✅ System Status:</strong> AI Health Assistant is ready to help! Start typing your health questions below.
        </div>
        """, unsafe_allow_html=True)
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question

        # Display all messages
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
            
            # Generate AI response with typing effect
            bot_container = st.empty()
            try:
                answer = get_ai_response(user_input)
                bot_typing(bot_container, answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered a technical issue. Please try again. Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            # Auto-scroll to bottom and refresh
            st.markdown(
                """
                <script>
                    window.scrollTo(0, document.body.scrollHeight);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.rerun()

# -------- MAIN EXECUTION --------
def main():
    """Main function that decides which page to show"""
    
    # Check if user is logged in and trying to access HealthBot
    if st.session_state.get('logged_in', False):
        # User is logged in, show HealthBot
        healthbot_app()
    else:
        # User is not logged in, show login page
        login_system()

if __name__ == "__main__":
    main()
