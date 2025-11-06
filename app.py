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

# -------- Typing effect with realistic delays --------
def bot_typing(container, text, delay=0.02):
    """Enhanced typing effect with realistic behavior"""
    thinking_time = min(1.5, len(text) * 0.008)
    time.sleep(thinking_time)
    
    # Show typing indicator
    with container:
        typing_indicator = st.empty()
        typing_indicator.markdown(
            """
            <div style='display:flex; align-items:flex-start; margin-bottom:12px;'>
                <div style='background:linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                            width:40px;height:40px;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;margin-right:12px;
                            box-shadow: 0 4px 15px rgba(58, 123, 213, 0.3);'>
                    <span style='color:white;font-size:18px;'>💠</span>
                </div>
                <div style='color:#666;background:#f8fafc;padding:12px 16px;border-radius:16px;
                            border:1px solid #e2e8f0;font-style:italic;font-size:14px;'>
                    HealthBot is thinking...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        time.sleep(1.2)
    
    # Clear typing indicator and show actual message
    typing_indicator.empty()
    
    # Type out the actual message
    message_container = container.empty()
    typed = ""
    for char in text:
        typed += char
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
                    {typed}
                    <div style='position:absolute;bottom:8px;right:12px;font-size:10px;color:#a0aec0;'>{time.strftime('%H:%M')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay * random.uniform(0.3, 0.8))

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
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.quick_question = question
                st.rerun()

# -------- Clear chat function --------
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
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
        
        /* Secondary buttons */
        .stButton>button[kind="secondary"] {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: #4a5568;
            border: 1.5px solid #e2e8f0;
        }
        .stButton>button[kind="secondary"]:hover {
            border-color: #3a7bd5;
            color: #3a7bd5;
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
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Health Assistant, here to provide you with reliable health information and guidance. I can help you understand symptoms, provide wellness tips, and offer general health advice. Remember, I'm an AI assistant and not a substitute for professional medical care. How can I help you today?"}
        ]

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
        
        # Stats card
        st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #718096; font-size: 14px;'>Messages</span>
                    <span style='color: #3a7bd5; font-weight: 700; font-size: 18px;'>{len(st.session_state.messages)}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
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
        
        if st.button("🔄 Clear Conversation", use_container_width=True, type="secondary"):
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
        st.markdown("""
            <div style='background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); 
                        padding: 1rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;'>
                <div style='color: #2d3748; font-weight: 600; font-size: 14px;'>
                    ✅ System Ready • AI Assistant Online
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Chat container - FIXED: This will display above the input
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                display_message(msg)

        # Quick replies for new chats
        if len(st.session_state.messages) <= 1:
            create_quick_replies()

        # Input area - FIXED: This stays at the bottom
        st.markdown("---")
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question

        # Input form - FIXED: Using a different approach
        input_container = st.container()
        with input_container:
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Type your health question...",
                    value=current_input_value,
                    placeholder="Ask about symptoms, treatments, or health tips...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                send_button = st.button("Send", use_container_width=True)

        # Process input when send button is clicked
        if send_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate AI response
            with st.spinner(""):
                try:
                    answer = get_ai_response(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Rerun to update the chat display
            st.rerun()

        # Auto-scroll to bottom after rerun
        st.markdown(
            """
            <script>
                window.scrollTo(0, document.body.scrollHeight);
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
