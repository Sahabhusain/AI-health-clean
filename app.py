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

# -------- CONFIG --------
DATA_PATH = r"C:\Users\sahah\Downloads\HealthChatbot\data"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CUSTOM_PROMPT_TEMPLATE = """
Use all the context provided to give a detailed and informative answer.
Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# -------- Vectorstore Load/Build --------
def build_vectorstore():
    os.makedirs("vectorstore", exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db

# -------- WhatsApp Style Typing Effect --------
def bot_typing(container, text, delay=0.02):
    thinking_time = min(1.0, len(text) * 0.008)
    time.sleep(thinking_time)
    
    with container:
        typing_indicator = st.empty()
        typing_indicator.markdown(
            """
            <div style='display:flex; align-items:flex-start; margin-bottom:8px;'>
                <div style='background:#25D366;width:32px;height:32px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:8px;'>
                    <span style='color:white;font-size:16px;'>🤖</span>
                </div>
                <div style='background:#ffffff;padding:8px 12px;border-radius:18px;
                            border:1px solid #e0e0e0;color:#666;font-style:italic;'>
                    HealthBot is typing...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.8)
    
    typing_indicator.empty()
    
    message_container = container.empty()
    typed = ""
    for char in text:
        typed += char
        message_container.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:8px;'>
                <div style='background:#25D366;width:32px;height:32px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:8px;'>
                    <span style='color:white;font-size:16px;'>🤖</span>
                </div>
                <div style='background:#ffffff;padding:12px 16px;border-radius:18px;max-width:70%;
                            border:1px solid #e0e0e0;box-shadow:0 1px 2px rgba(0,0,0,0.1);'>
                    <div style='color:#2c3e50;line-height:1.4;font-size:14px;'>{typed}</div>
                    <div style='text-align:right;color:#999;font-size:11px;margin-top:4px;'>
                        {time.strftime('%H:%M')}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay * random.uniform(0.3, 0.8))

# -------- WhatsApp Style Messages --------
def display_message(msg):
    current_time = time.strftime('%H:%M')
    
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:8px;'>
                <div style='background:#DCF8C6;padding:12px 16px;border-radius:18px;max-width:70%;
                            margin-left:40px;box-shadow:0 1px 2px rgba(0,0,0,0.1);'>
                    <div style='color:#2c3e50;line-height:1.4;font-size:14px;'>{msg['content']}</div>
                    <div style='text-align:right;color:#999;font-size:11px;margin-top:4px;'>
                        {current_time} ✓
                    </div>
                </div>
                <div style='background:#34B7F1;width:32px;height:32px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-left:8px;'>
                    <span style='color:white;font-size:14px;'>👤</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:8px;'>
                <div style='background:#25D366;width:32px;height:32px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;margin-right:8px;'>
                    <span style='color:white;font-size:16px;'>🤖</span>
                </div>
                <div style='background:#ffffff;padding:12px 16px;border-radius:18px;max-width:70%;
                            border:1px solid #e0e0e0;box-shadow:0 1px 2px rgba(0,0,0,0.1);'>
                    <div style='color:#2c3e50;line-height:1.4;font-size:14px;'>{msg['content']}</div>
                    <div style='text-align:right;color:#999;font-size:11px;margin-top:4px;'>
                        {current_time}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- Quick Questions --------
def create_quick_replies():
    quick_questions = [
        "What are common cold symptoms?",
        "How to improve sleep quality?",
        "What foods boost immunity?",
        "Exercise recommendations?",
        "Stress management tips?"
    ]
    
    st.markdown("---")
    st.markdown("**💡 Quick Questions:**")
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                return question
    return None

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="HealthBot - WhatsApp Style",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # WhatsApp Style CSS
    st.markdown("""
        <style>
        /* Main background */
        .main {
            background: linear-gradient(135deg, #00a884 0%, #1e2a3a 100%);
        }
        
        /* Chat container */
        .chat-container {
            background-color: #e5ddd5;
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%2300a884' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
            padding: 20px;
            border-radius: 10px;
            height: 70vh;
            overflow-y: auto;
        }
        
        /* Input area */
        .input-area {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        /* Header */
        .header {
            background: #00a884;
            color: white;
            padding: 15px;
            border-radius: 10px 10px 0 0;
            text-align: center;
        }
        
        /* Custom text input */
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 12px 20px;
            font-size: 14px;
            border: 1px solid #ddd;
        }
        
        /* Custom button */
        .stButton>button {
            border-radius: 20px;
            background: #00a884;
            color: white;
            border: none;
            padding: 12px 25px;
            font-weight: 500;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm HealthBot, your AI health assistant. I'm here to help you with any health-related questions. Feel free to ask me anything! 💚"}
        ]

    # Main layout
    st.markdown("""
        <div class="header">
            <h3 style="margin:0; display:flex; align-items:center; justify-content:center; gap:10px;">
                <span>🤖</span> HealthBot Assistant <span>💚</span>
            </h3>
            <p style="margin:5px 0 0 0; font-size:12px; opacity:0.9;">Online • Always here to help</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    for msg in st.session_state.messages:
        display_message(msg)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # Quick questions for new chat
    if len(st.session_state.messages) <= 1:
        quick_question = create_quick_replies()
        if quick_question:
            user_input = quick_question
        else:
            user_input = ""
    else:
        user_input = ""

    # Input form
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Type a message...",
            value=user_input,
            placeholder="Type your health question here...",
            key="user_input",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Process user input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate AI response
        with st.spinner(""):
            try:
                vectorstore = build_vectorstore()
                
                if vectorstore is None:
                    st.error("⚠️ No medical documents found.")
                    return

                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                        temperature=0.3,
                        max_tokens=1024,
                        groq_api_key=GROQ_API_KEY
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                # Generate response with typing effect
                bot_container = st.empty()
                result = qa_chain.invoke({"query": user_input})["result"]
                bot_typing(bot_container, result)
                st.session_state.messages.append({"role": "assistant", "content": result})
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Sorry, I'm having trouble right now. Please try again."
                })
        
        st.rerun()

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm HealthBot, your AI health assistant. I'm here to help you with any health-related questions. Feel free to ask me anything! 💚"}
        ]
        st.rerun()

if __name__ == "__main__":
    main()
