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
import hashlib
import json
import re

# Load environment variables
load_dotenv()

# -------- CONFIG --------
DATA_PATH = r"C:\Users\sahah\Downloads\HealthChatbot\data"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USER_DATA_FILE = "users.json"

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

# -------- User Authentication Functions --------
def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file."""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_users(users):
    """Save users to JSON file."""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        return True
    except:
        return False

def register_user(username, email, password):
    """Register a new user."""
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists"
    
    # Check if email already exists
    for user in users.values():
        if user['email'] == email:
            return False, "Email already registered"
    
    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False, "Invalid email format"
    
    # Validate password strength
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Create new user
    users[username] = {
        'email': email,
        'password': hash_password(password),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'chat_history': []
    }
    
    if save_users(users):
        return True, "Registration successful!"
    else:
        return False, "Registration failed. Please try again."

def login_user(username, password):
    """Login user."""
    users = load_users()
    
    if username in users:
        if users[username]['password'] == hash_password(password):
            return True, "Login successful!"
    
    return False, "Invalid username or password"

def save_chat_history(username, messages):
    """Save chat history for user."""
    users = load_users()
    if username in users:
        users[username]['chat_history'] = messages
        save_users(users)

def load_chat_history(username):
    """Load chat history for user."""
    users = load_users()
    if username in users:
        return users[username].get('chat_history', [])
    return []

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

# -------- Enhanced Realistic Typing Effect --------
def realistic_typing_effect(container, text, message_index):
    """Realistic typing effect with human-like variations"""
    
    # Calculate thinking time based on text complexity
    thinking_time = min(2.0, max(1.0, len(text) * 0.01))
    
    # Show realistic typing indicator
    with container:
        typing_container = st.empty()
        
        # Animated typing indicator with dots
        typing_stages = [
            "HealthBot is thinking ●○○",
            "HealthBot is thinking ○●○", 
            "HealthBot is thinking ○○●",
            "HealthBot is thinking ●●●"
        ]
        
        # Show thinking animation
        for i in range(3):
            for stage in typing_stages:
                typing_container.markdown(
                    f"""
                    <div style='display:flex; align-items:flex-start; margin-bottom:20px;'>
                        <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                                    width:40px;height:40px;border-radius:12px;
                                    display:flex;align-items:center;justify-content:center;margin-right:12px;
                                    box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                            <span style='color:white;font-size:18px;'>💠</span>
                        </div>
                        <div style='color:#666;background:#f8fafc;padding:12px 16px;border-radius:16px;
                                    border:1px solid #e2e8f0;font-style:italic;font-size:14px;'>
                            {stage}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(0.3)
        
        time.sleep(thinking_time)
        
        # Clear typing indicator
        typing_container.empty()
        
        # Start typing the actual message
        message_container = st.empty()
        typed_text = ""
        
        # Human-like typing patterns
        typing_speed_variations = [
            0.02, 0.03, 0.015, 0.025, 0.02, 0.035, 0.018, 0.028
        ]
        
        # Simulate realistic typing with occasional pauses
        words = text.split(' ')
        current_sentence = ""
        
        for i, word in enumerate(words):
            # Add word with space
            current_sentence += word + " "
            typed_text = current_sentence
            
            # Update message display
            message_container.markdown(
                f"""
                <div style='display:flex; align-items:flex-start; margin-bottom:20px;'>
                    <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                                width:40px;height:40px;border-radius:12px;
                                display:flex;align-items:center;justify-content:center;margin-right:12px;
                                box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                        <span style='color:white;font-size:18px;'>💠</span>
                    </div>
                    <div style='color:#2d3748;background:#ffffff;padding:16px 20px;border-radius:18px;max-width:80%;line-height:1.5;font-size:14px;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.08);border:1px solid #e2e8f0;
                                position:relative;'>
                        <div style='font-weight:600;color:#3a7bd5;font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;'>Health Assistant</div>
                        {typed_text}
                        <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:#a0aec0;'>{time.strftime('%H:%M')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Variable typing speed
            base_speed = random.choice(typing_speed_variations)
            
            # Occasionally pause at sentence ends or commas
            if word.endswith(('.', '!', '?', ',')) or random.random() < 0.1:
                pause_time = random.uniform(0.1, 0.3)
                time.sleep(pause_time)
            
            # Speed variations for different word lengths
            if len(word) > 8:
                time.sleep(base_speed * 1.5)
            elif len(word) < 3:
                time.sleep(base_speed * 0.7)
            else:
                time.sleep(base_speed)
        
        # Final message with cursor blink effect
        for _ in range(2):
            # Show with cursor
            message_container.markdown(
                f"""
                <div style='display:flex; align-items:flex-start; margin-bottom:20px;'>
                    <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                                width:40px;height:40px;border-radius:12px;
                                display:flex;align-items:center;justify-content:center;margin-right:12px;
                                box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                        <span style='color:white;font-size:18px;'>💠</span>
                    </div>
                    <div style='color:#2d3748;background:#ffffff;padding:16px 20px;border-radius:18px;max-width:80%;line-height:1.5;font-size:14px;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.08);border:1px solid #e2e8f0;
                                position:relative;'>
                        <div style='font-weight:600;color:#3a7bd5;font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;'>Health Assistant</div>
                        {typed_text}<span style='animation: blink 1s infinite;'>|</span>
                        <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:#a0aec0;'>{time.strftime('%H:%M')}</div>
                    </div>
                </div>
                <style>
                @keyframes blink {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0; }}
                    100% {{ opacity: 1; }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.3)
            
            # Show without cursor
            message_container.markdown(
                f"""
                <div style='display:flex; align-items:flex-start; margin-bottom:20px;'>
                    <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                                width:40px;height:40px;border-radius:12px;
                                display:flex;align-items:center;justify-content:center;margin-right:12px;
                                box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                        <span style='color:white;font-size:18px;'>💠</span>
                    </div>
                    <div style='color:#2d3748;background:#ffffff;padding:16px 20px;border-radius:18px;max-width:80%;line-height:1.5;font-size:14px;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.08);border:1px solid #e2e8f0;
                                position:relative;'>
                        <div style='font-weight:600;color:#3a7bd5;font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;'>Health Assistant</div>
                        {typed_text}
                        <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:#a0aec0;'>{time.strftime('%H:%M')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.2)
        
        return typed_text

# -------- Display messages with modern design --------
def display_message(msg):
    current_time = time.strftime('%H:%M')
    
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:20px;'>
                <div style='color:white;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding:16px 20px;border-radius:18px;max-width:80%;line-height:1.5;font-size:14px;
                            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);position:relative;'>
                    <div style='font-weight:600;color:rgba(255,255,255,0.9);font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;'>You</div>
                    {msg['content']}
                    <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:rgba(255,255,255,0.7);'>{current_time}</div>
                </div>
                <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            width:40px;height:40px;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;margin-left:12px;
                            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
                    <span style='color:white;font-size:18px;'>👤</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:20px;'>
                <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                            width:40px;height:40px;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                    <span style='color:white;font-size:18px;'>💠</span>
                </div>
                <div style='color:#2d3748;background:#ffffff;padding:16px 20px;border-radius:18px;max-width:80%;line-height:1.5;font-size:14px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.08);border:1px solid #e2e8f0;
                            position:relative;'>
                    <div style='font-weight:600;color:#3a7bd5;font-size:12px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;'>Health Assistant</div>
                    {msg['content']}
                    <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:#a0aec0;'>{current_time}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- Quick reply chips --------
def create_quick_replies():
    quick_questions = [
        "What are common cold symptoms?",
        "How to improve sleep quality?",
        "What foods boost immunity?",
        "Exercise recommendations?",
        "Stress management tips?",
        "When to see a doctor for fever?"
    ]
    
    st.markdown("""
        <div style='margin:20px 0;'>
            <div style='font-size:12px;color:#718096;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:12px;'>Quick Questions</div>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(
                question, 
                key=f"quick_{i}", 
                use_container_width=True
            ):
                st.session_state.quick_question = question
                st.rerun()

# -------- Clear chat function --------
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
    ]
    # Save chat history if user is logged in
    if st.session_state.get('logged_in'):
        save_chat_history(st.session_state.username, st.session_state.messages)

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

# -------- Authentication Pages --------
def show_login_page():
    """Display login page"""
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #2d3748; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
                🏥 Health Assistant
            </h1>
            <p style='color: #718096; font-size: 1.1rem; margin: 0;'>
                Sign in to your account
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.subheader("🔐 Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Sign In", use_container_width=True)
            
            if login_button:
                if username and password:
                    success, message = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.show_login = False
                        st.session_state.show_register = False
                        # Load user's chat history
                        st.session_state.messages = load_chat_history(username)
                        if not st.session_state.messages:
                            st.session_state.messages = [
                                {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
                            ]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
            
            st.markdown("---")
            st.markdown("Don't have an account?")
            if st.button("Create New Account", use_container_width=True):
                st.session_state.show_login = False
                st.session_state.show_register = True
                st.rerun()
            
            st.markdown("---")
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.show_login = False
                st.session_state.show_register = False
                st.rerun()

def show_register_page():
    """Display registration page"""
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #2d3748; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
                🏥 Health Assistant
            </h1>
            <p style='color: #718096; font-size: 1.1rem; margin: 0;'>
                Create a new account
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("register_form"):
            st.subheader("📝 Create Account")
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Create a password (min. 6 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            register_button = st.form_submit_button("Create Account", use_container_width=True)
            
            if register_button:
                if username and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        success, message = register_user(username, email, password)
                        if success:
                            st.success(message)
                            st.session_state.show_register = False
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")
            
            st.markdown("---")
            st.markdown("Already have an account?")
            if st.button("Sign In", use_container_width=True):
                st.session_state.show_register = False
                st.session_state.show_login = True
                st.rerun()

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="Health Assistant",
        page_icon="💠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern CSS styling
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 16px;
            padding: 14px 20px;
            font-size: 14px;
            border: 1.5px solid #e2e8f0;
            background: #ffffff;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #3a7bd5;
            box-shadow: 0 0 0 3px rgba(58, 123, 213, 0.1);
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 16px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 14px;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Chat container */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* Blinking cursor animation */
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
        ]
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Show authentication pages if needed
    if st.session_state.show_login:
        show_login_page()
        return
        
    if st.session_state.show_register:
        show_register_page()
        return

    # Sidebar with modern design
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem; padding: 1rem;'>
                <div style='background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%); 
                            padding: 2rem; border-radius: 20px; margin-bottom: 1rem;'>
                    <h1 style='color: white; font-size: 1.8rem; margin: 0;'>💠</h1>
                    <h2 style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Health Assistant</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # User info card
        if st.session_state.logged_in:
            st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    width: 40px; height: 40px; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; margin-right: 12px;'>
                            <span style='color: white; font-weight: bold;'>{st.session_state.username[0].upper()}</span>
                        </div>
                        <div>
                            <div style='font-weight: 600; color: #2d3748;'>{st.session_state.username}</div>
                            <div style='font-size: 12px; color: #718096;'>Premium User</div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #718096; font-size: 14px;'>Messages</span>
                        <span style='color: #3a7bd5; font-weight: 700; font-size: 18px;'>{len(st.session_state.messages)}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                        <div style='background: linear-gradient(135deg, #a0aec0 0%, #718096 100%); 
                                    width: 40px; height: 40px; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; margin-right: 12px;'>
                            <span style='color: white; font-weight: bold;'>G</span>
                        </div>
                        <div>
                            <div style='font-weight: 600; color: #2d3748;'>Guest User</div>
                            <div style='font-size: 12px; color: #718096;'>Limited Access</div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #718096; font-size: 14px;'>Messages</span>
                        <span style='color: #3a7bd5; font-weight: 700; font-size: 18px;'>{len(st.session_state.messages)}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Authentication buttons
        if not st.session_state.logged_in:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", use_container_width=True):
                    st.session_state.show_login = True
                    st.rerun()
            with col2:
                if st.button("📝 Register", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
        else:
            if st.button("🚪 Logout", use_container_width=True):
                # Save chat history before logout
                save_chat_history(st.session_state.username, st.session_state.messages)
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
                ]
                st.success("Logged out successfully!")
                st.rerun()
        
        st.markdown("### Features")
        features = [
            ("🤖", "AI Health Guidance"),
            ("📚", "Medical Knowledge"),
            ("🔒", "Private & Secure"),
            ("⚡", "Instant Responses")
        ]
        
        for icon, text in features:
            st.markdown(f"""
                <div style='display: flex; align-items: center; padding: 0.5rem 0; color: #4a5568;'>
                    <span style='font-size: 1.2rem; margin-right: 0.75rem;'>{icon}</span>
                    <span style='font-size: 14px;'>{text}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔄 Clear Conversation", use_container_width=True):
            clear_chat()
            st.rerun()

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Header
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2d3748; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem;'>
                    Health Assistant
                </h1>
                <p style='color: #718096; font-size: 1rem; margin: 0;'>
                    Your AI-powered health companion
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        if st.session_state.logged_in:
