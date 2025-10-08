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
You are HealthBot, a friendly and caring health assistant. Be conversational, warm, and helpful.

Context: {context}
Question: {question}

Respond like a real healthcare assistant having a friendly conversation:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# -------- Vectorstore Load/Build --------
@st.cache_resource
def build_vectorstore():
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
        return None

# -------- Direct AI Response --------
def get_direct_ai_response(question):
    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.8,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        You're HealthBot, a friendly health assistant. The user asked: "{question}"
        
        Respond naturally like you're having a real conversation:
        - Be warm and conversational
        - Use simple, friendly language
        - Show empathy and understanding
        - Keep responses concise but helpful
        - Use occasional emojis to make it friendly
        - Sound like a real person chatting
        
        Response:"""
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return "Hey! I'm having some trouble right now. Could you try again? 😊"

# -------- Realistic Typing Effect --------
def simulate_typing(container, text):
    # Show thinking animation
    with container:
        thinking_container = st.empty()
        
        # Animated thinking dots
        for i in range(3):
            dots = "." * (i + 1)
            thinking_container.markdown(f"""
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px; padding: 10px 15px; background: #f8f9fa; border-radius: 15px; border: 1px solid #e9ecef; width: fit-content;'>
                    <div style='width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2); display: flex; align-items: center; justify-content: center;'>
                        <span style='color: white; font-size: 14px;'>AI</span>
                    </div>
                    <div style='font-size: 14px; color: #6c757d;'>HealthBot is thinking{dots}</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.6)
    
    thinking_container.empty()
    
    # Type out message with realistic pacing
    message_container = container.empty()
    displayed_text = ""
    
    words = text.split()
    
    for i, word in enumerate(words):
        displayed_text += word + " "
        
        # Create professional message bubble
        message_container.markdown(f"""
            <div style='display: flex; align-items: flex-start; gap: 10px; margin-bottom: 15px;'>
                <div style='width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                    <span style='color: white; font-size: 12px; font-weight: bold;'>AI</span>
                </div>
                <div style='background: linear-gradient(135deg, #f8f9fa, #ffffff); padding: 12px 16px; border-radius: 18px; border: 1px solid #e9ecef; max-width: 70%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                    <div style='font-size: 14px; line-height: 1.5; color: #2d3748;'>{displayed_text.strip()}</div>
                    <div style='font-size: 11px; color: #a0aec0; text-align: right; margin-top: 5px;'>{time.strftime('%H:%M')}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Realistic typing speed variations
        if i < len(words) - 1:
            if word.endswith(('.', '!', '?')):
                time.sleep(0.4)  # Longer pause after sentences
            elif len(word) > 6:
                time.sleep(0.15)  # Slightly longer for long words
            else:
                time.sleep(0.08 + random.random() * 0.05)  # Natural variation

# -------- Display Messages --------
def display_message(msg):
    current_time = time.strftime('%H:%M')
    
    if msg["role"] == "user":
        # User message - Right side with gradient
        st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; align-items: flex-start; gap: 10px; margin-bottom: 15px;'>
                <div style='background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 12px 16px; border-radius: 18px; max-width: 70%; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);'>
                    <div style='font-size: 14px; line-height: 1.5;'>{msg['content']}</div>
                    <div style='font-size: 11px; color: rgba(255,255,255,0.7); text-align: right; margin-top: 5px;'>{current_time} ✓</div>
                </div>
                <div style='width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #48bb78, #38a169); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                    <span style='color: white; font-size: 14px;'>You</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Bot message - Left side with professional design
        st.markdown(f"""
            <div style='display: flex; align-items: flex-start; gap: 10px; margin-bottom: 15px;'>
                <div style='width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                    <span style='color: white; font-size: 12px; font-weight: bold;'>AI</span>
                </div>
                <div style='background: linear-gradient(135deg, #f8f9fa, #ffffff); padding: 12px 16px; border-radius: 18px; border: 1px solid #e9ecef; max-width: 70%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                    <div style='font-size: 14px; line-height: 1.5; color: #2d3748;'>{msg['content']}</div>
                    <div style='font-size: 11px; color: #a0aec0; text-align: right; margin-top: 5px;'>{current_time}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------- Get AI Response --------
def get_ai_response(question):
    try:
        vectorstore = build_vectorstore()
        
        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.8,
                    max_tokens=800,
                    groq_api_key=GROQ_API_KEY
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )
            
            result = qa_chain.invoke({"query": question})
            return result["result"]
        else:
            return get_direct_ai_response(question)
            
    except Exception as e:
        return get_direct_ai_response(question)

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="HealthBot AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Modern Professional Chat Interface CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .chat-wrapper {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        
        .chat-container {
            padding: 25px;
            height: 65vh;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .input-section {
            background: white;
            padding: 20px;
            border-top: 1px solid #e9ecef;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #48bb78;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .quick-actions {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 12px;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .quick-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 14px;
            border: 2px solid #e2e8f0;
            background: white;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .stButton>button {
            border-radius: 25px;
            padding: 14px 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize chat
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm HealthBot, your AI health assistant. 🤖 I'm here to help you with any health-related questions, symptom analysis, medication information, or general wellness advice. What would you like to know today?"}
        ]

    # Main Chat Interface
    st.markdown("""
        <div class="chat-wrapper">
            <div class="chat-header">
                <h1 style="margin: 0 0 10px 0; font-size: 28px;">🤖 HealthBot AI</h1>
                <p style="margin: 0; opacity: 0.9; font-size: 16px;">
                    <span class="status-indicator"></span>
                    Online • Ready to assist you
                </p>
            </div>
            
            <div class="chat-container">
    """, unsafe_allow_html=True)

    # Display chat messages
    for msg in st.session_state.messages:
        display_message(msg)
    
    st.markdown("</div>")  # Close chat-container
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <p style="font-size: 14px; color: #666; margin-bottom: 10px;">💡 <strong>Quick questions:</strong></p>
            <div class="quick-actions">
    """, unsafe_allow_html=True)
    
    quick_questions = [
        "🤒 Cold symptoms", 
        "😴 Sleep issues", 
        "🍎 Diet advice", 
        "💪 Exercise tips",
        "😌 Stress relief",
        "🌡️ Fever guidance"
    ]
    
    cols = st.columns(6)
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_question = question.split(" ")[1] + " advice"
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Handle quick questions
    current_input = ""
    if hasattr(st.session_state, 'quick_question'):
        current_input = st.session_state.quick_question
        del st.session_state.quick_question

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Type your health question...",
                value=current_input,
                placeholder="Ask me anything about health, symptoms, medications...",
                key="user_input",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send 💬", use_container_width=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)  # Close input-section and chat-wrapper

    # Clear chat button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start New Conversation", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm HealthBot, your AI health assistant. 🤖 I'm here to help you with any health-related questions, symptom analysis, medication information, or general wellness advice. What would you like to know today?"}
            ]
            st.rerun()

    # Process message
    if submitted and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate and display bot response
        bot_container = st.empty()
        try:
            response = get_ai_response(user_input)
            simulate_typing(bot_container, response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment! 😊"})
        
        st.rerun()

if __name__ == "__main__":
    main()
