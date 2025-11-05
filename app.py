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
# NOTE: Changed to relative path for portability. 
# Make sure your 'data' folder is in the same directory as this app.py
DATA_PATH = "data"  
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
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        if os.path.exists(DB_FAISS_PATH):
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        if os.path.exists(DATA_PATH):
            loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                db = FAISS.from_documents(docs, embedding_model)
                db.save_local(DB_FAISS_PATH)
                return db
        return None
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None

# -------- Direct AI Response (Fallback) --------
def get_direct_ai_response(question):
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

# -------- Get AI Response --------
def get_ai_response(question):
    try:
        vectorstore = build_vectorstore()
        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=1024, groq_api_key=GROQ_API_KEY),
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

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Variables for Medical Theme */
        :root {
            --primary-color: #0284c7; /* Medical Blue */
            --primary-dark: #0369a1;
            --secondary-color: #e0f2fe; /* Light Blue Bg */
            --accent-color: #10b981; /* Success Green */
            --text-dark: #1e293b;
            --text-light: #64748b;
            --bg-light: #f8fafc;
        }
        
        /* Main App Background */
        .stApp {
            background-color: var(--bg-light);
        }

        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            text-align: center;
            padding: 1rem 0;
            letter-spacing: -1px;
        }
        .sub-header {
            text-align: center;
            color: var(--text-light);
            font-size: 1.1rem;
            margin-top: -20px;
            margin-bottom: 2rem;
        }

        /* Chat Bubbles - Modern Look */
        .user-bubble-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 1rem;
        }
        .user-bubble {
            background-color: var(--primary-color);
            color: white;
            padding: 14px 18px;
            border-radius: 20px 20px 4px 20px;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.2);
            font-size: 15px;
            line-height: 1.5;
        }

        .bot-bubble-container {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 1rem;
            align-items: flex-start;
        }
        .bot-avatar {
            background: linear-gradient(135deg, var(--accent-color), #059669);
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            flex-shrink: 0;
        }
        .bot-bubble {
            background-color: white;
            color: var(--text-dark);
            padding: 16px 20px;
            border-radius: 4px 20px 20px 20px;
            max-width: 80%;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            font-size: 15px;
            line-height: 1.6;
        }
        .bot-name {
            font-size: 12px;
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 4px;
        }

        /* Typing Indicator */
        .typing-dots {
            display: inline-block;
        }
        @keyframes dot-blink {
            0% { opacity: 0.2; }
            20% { opacity: 1.0; }
            100% { opacity: 0.2; }
        }
        .typing-dots span {
            animation: dot-blink 1.4s infinite both;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

        /* Input Area Styling */
        .stTextInput > div > div > input {
            border-radius: 30px !important;
            padding: 12px 25px !important;
            border: 2px solid #e2e8f0 !important;
            background-color: white !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px var(--secondary-color) !important;
        }
        .stButton > button {
            border-radius: 30px !important;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
            color: white !important;
            border: none !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.3) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #e2e8f0;
        }
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        .info-box {
            background-color: var(--secondary-color);
            padding: 1rem;
            border-radius: 12px;
            color: var(--primary-dark);
            font-size: 0.9rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-color);
        }
        </style>
    """, unsafe_allow_html=True)

def display_message(msg):
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-bubble-container">
                <div class="user-bubble">
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bot-bubble-container">
                <div class="bot-avatar">🩺</div>
                <div class="bot-bubble">
                    <div class="bot-name">HealthBot AI</div>
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)

def bot_typing(container, text, delay=0.01):
    """Simpler, smoother typing effect"""
    # Initial typing indicator
    container.markdown("""
        <div class="bot-bubble-container">
            <div class="bot-avatar">🩺</div>
            <div class="bot-bubble" style="padding: 12px 24px;">
                <div class="bot-name">HealthBot AI</div>
                <div class="typing-dots" style="color: #64748b; font-size: 24px; line-height: 10px;">
                    <span>•</span><span>•</span><span>•</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(0.8) # Initial thinking time

    # Typing animation
    message_placeholder = container.empty()
    typed_text = ""
    # Speed up typing for longer responses
    dynamic_delay = max(0.005, min(delay, 1.0 / len(text))) if len(text) > 500 else delay

    for char in text:
        typed_text += char
        # Update every few chars for better performance on long text
        if len(typed_text) % 3 == 0 or char == text[-1]: 
            message_placeholder.markdown(f"""
                <div class="bot-bubble-container">
                    <div class="bot-avatar">🩺</div>
                    <div class="bot-bubble">
                        <div class="bot-name">HealthBot AI</div>
                        {typed_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(dynamic_delay)

def main():
    st.set_page_config(page_title="HealthBot AI", page_icon="🩺", layout="centered")
    inject_custom_css()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm **HealthBot AI**. 👋\n\nI can help you with symptoms, healthy lifestyle tips, and general medical information based on my knowledge base.\n\n*How can I assist you today?*"}
        ]

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🩺 HealthBot AI</div>', unsafe_allow_html=True)
        st.markdown("Your personal AI health assistant, powered by advanced LLMs and verified medical documents.")
        
        st.markdown("---")
        st.markdown("#### 📊 Session Info")
        st.markdown(f"""
            <div class="info-box">
                <b>Messages:</b> {len(st.session_state.messages)}<br>
                <b>Status:</b> ✅ Online
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Quick Tips")
        with st.expander("See common topics"):
            st.markdown("""
            - 🤒 Cold & Flu
            - 🥗 Diet & Nutrition
            - 💤 Sleep Hygiene
            - 🧘 Mental Health
            """)
            
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

    # --- MAIN PAGE ---
    st.markdown('<h1 class="main-header">AI Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about your health & wellness</p>', unsafe_allow_html=True)

    # Chat Container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            display_message(msg)

    # Handle quick replies if it's a new chat
    if len(st.session_state.messages) <= 1:
        st.markdown("##### Try asking:")
        cols = st.columns(3)
        quick_prompts = ["How to boost immunity?", "Symptoms of migraine?", "Tips for better sleep"]
        for i, prompt in enumerate(quick_prompts):
            with cols[i]:
                if st.button(prompt, use_container_width=True, key=f"quick_{i}"):
                    st.session_state.quick_ask = prompt
                    st.rerun()

    # Input Area - Fixed at bottom look
    st.markdown("---")
    input_val = st.session_state.get("quick_ask", "")
    if "quick_ask" in st.session_state:
        del st.session_state.quick_ask

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message", 
                value=input_val, 
                placeholder="Type your health question here...", 
                label_visibility="collapsed"
            )
        with col2:
            # Using an emoji in button for a cleaner look
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            display_message({"role": "user", "content": user_input})
            bot_placeholder = st.empty()
            response = get_ai_response(user_input)
            bot_typing(bot_placeholder, response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
