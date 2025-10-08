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
You are HealthBot, a friendly and caring AI health assistant. Use the provided context to give detailed, accurate, and easy-to-understand health information.

Context: {context}
Question: {question}

Provide a warm, helpful, and comprehensive health-related answer:
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
            temperature=0.7,
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        
        health_prompt = f"""
        You are HealthBot, a friendly and caring health assistant. The user asked: {question}
        
        Please respond in a warm, conversational tone as if you're a real healthcare assistant:
        - Be empathetic and understanding
        - Use simple, clear language
        - Show genuine care and concern
        - Provide practical, actionable advice
        - Keep it conversational but informative
        
        Response:
        """
        
        response = llm.invoke(health_prompt)
        return response.content
        
    except Exception as e:
        return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment! 😊"

# -------- Realistic Typing Effect --------
def bot_typing(container, text, delay=0.015):
    """Realistic typing effect that feels like a real person"""
    # Show thinking indicator
    with container:
        thinking_indicator = st.empty()
        thinking_indicator.markdown(
            """
            <div style='display:flex; align-items:center; margin-bottom:10px;'>
                <div style='background:#4CAF50;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:10px;'>
                    <span style='color:white;font-size:18px;'>💭</span>
                </div>
                <div style='color:#666;font-size:14px;font-style:italic;'>
                    HealthBot is thinking...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1.5)
    
    thinking_indicator.empty()
    
    # Type out message
    message_container = container.empty()
    typed = ""
    
    # Split into sentences for more natural typing
    sentences = text.split('. ')
    current_sentence = 0
    sentence_content = ""
    
    while current_sentence < len(sentences):
        sentence = sentences[current_sentence]
        if len(sentence_content) < len(sentence):
            sentence_content += sentence[len(sentence_content)]
            typed = '. '.join(sentences[:current_sentence]) + ('. ' if current_sentence > 0 else '') + sentence_content
            
            message_container.markdown(
                f"""
                <div style='display:flex; align-items:flex-start; margin-bottom:15px;'>
                    <div style='background:#4CAF50;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:10px;flex-shrink:0;'>
                        <span style='color:white;font-size:18px;'>🤖</span>
                    </div>
                    <div style='background:#f0f8f0;padding:12px 16px;border-radius:18px;max-width:70%;border:1px solid #c8e6c9;'>
                        <div style='color:#2e7d32;font-weight:600;font-size:14px;margin-bottom:4px;'>HealthBot</div>
                        <div style='color:#333;line-height:1.5;font-size:14px;'>{typed}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(delay * random.uniform(0.3, 0.8))
        else:
            # Move to next sentence
            current_sentence += 1
            sentence_content = ""
            if current_sentence < len(sentences):
                typed += '. '
                # Small pause between sentences
                time.sleep(0.3)

# -------- Display Messages --------
def display_message(msg):
    if msg["role"] == "user":
        # User message - Right side, friendly green
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:15px;'>
                <div style='background:#4CAF50;color:white;padding:12px 16px;border-radius:18px;max-width:70%;margin-left:50px;'>
                    <div style='font-weight:600;font-size:14px;margin-bottom:4px;opacity:0.9;'>You</div>
                    <div style='line-height:1.5;font-size:14px;'>{msg['content']}</div>
                </div>
                <div style='background:#4CAF50;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-left:10px;flex-shrink:0;'>
                    <span style='color:white;font-size:18px;'>😊</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Bot message - Left side, light green background
        st.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:15px;'>
                <div style='background:#4CAF50;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:10px;flex-shrink:0;'>
                    <span style='color:white;font-size:18px;'>🤖</span>
                </div>
                <div style='background:#f0f8f0;padding:12px 16px;border-radius:18px;max-width:70%;border:1px solid #c8e6c9;'>
                    <div style='color:#2e7d32;font-weight:600;font-size:14px;margin-bottom:4px;'>HealthBot</div>
                    <div style='color:#333;line-height:1.5;font-size:14px;'>{msg['content']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- Quick Action Buttons --------
def create_quick_actions():
    """Friendly quick action buttons"""
    actions = [
        {"icon": "🤒", "text": "Cold & Flu", "question": "What are the symptoms of common cold and how can I relieve them?"},
        {"icon": "😴", "text": "Sleep Help", "question": "I'm having trouble sleeping, what can I do?"},
        {"icon": "🍎", "text": "Healthy Diet", "question": "What foods should I eat to boost my immunity?"},
        {"icon": "💪", "text": "Exercise", "question": "What are some good beginner exercises for fitness?"},
        {"icon": "😌", "text": "Stress Relief", "question": "How can I manage stress and anxiety?"},
        {"icon": "🌡️", "text": "Fever Advice", "question": "When should I see a doctor for fever?"}
    ]
    
    st.markdown("""
        <div style='text-align:center; margin:20px 0;'>
            <h4 style='color:#4CAF50; margin-bottom:15px;'>💬 Quick Questions</h4>
            <p style='color:#666; font-size:14px;'>Tap any topic to start chatting!</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, action in enumerate(actions):
        with cols[i % 3]:
            if st.button(
                f"{action['icon']} {action['text']}", 
                key=f"action_{i}",
                use_container_width=True,
                help=f"Ask about {action['text'].lower()}"
            ):
                st.session_state.quick_question = action['question']
                st.rerun()

# -------- Get AI Response --------
def get_ai_response(question):
    """Get friendly AI response"""
    try:
        vectorstore = build_vectorstore()
        
        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.7,
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

# -------- Main App --------
def main():
    st.set_page_config(
        page_title="HealthBot - Your Friendly Health Assistant",
        page_icon="💚",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Friendly & Clean CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f8fff8 0%, #f0f8f0 100%);
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.1);
            border: 1px solid #e8f5e8;
            margin-bottom: 20px;
            min-height: 400px;
            max-height: 55vh;
            overflow-y: auto;
        }
        
        .welcome-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .welcome-title {
            font-size: 2.5rem;
            color: #4CAF50;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .welcome-subtitle {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #c8e6c9;
            background: white;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        
        .stButton>button {
            border-radius: 25px;
            padding: 12px 25px;
            font-weight: 600;
            background: #4CAF50;
            color: white;
            border: none;
            transition: all 0.3s ease;
            font-size: 15px;
        }
        
        .stButton>button:hover {
            background: #45a049;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        
        .quick-action-btn {
            background: white !important;
            color: #4CAF50 !important;
            border: 2px solid #4CAF50 !important;
        }
        
        .quick-action-btn:hover {
            background: #4CAF50 !important;
            color: white !important;
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
            background: #4CAF50;
            border-radius: 3px;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! 👋 I'm HealthBot, your friendly health assistant! \n\nI'm here to help you with any health questions you might have - whether it's about symptoms, medications, healthy living, or just general wellness advice.\n\nWhat's on your mind today? 😊"}
        ]

    # Main Chat Interface
    st.markdown("""
        <div class="welcome-header">
            <div class="welcome-title">💚 HealthBot</div>
            <div class="welcome-subtitle">Your Friendly AI Health Assistant</div>
        </div>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        display_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick actions for new chats
    if len(st.session_state.messages) <= 1:
        create_quick_actions()

    # Handle quick questions
    current_input_value = ""
    if hasattr(st.session_state, 'quick_question'):
        current_input_value = st.session_state.quick_question
        del st.session_state.quick_question

    # Input area - Simple and clean
    st.markdown("""
        <div style='text-align: center; margin: 20px 0 10px 0;'>
            <p style='color: #666; font-size: 14px;'>Type your health question below...</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "Message HealthBot...",
                value=current_input_value,
                placeholder="Example: I have a headache, what should I do?",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col_send:
            submitted = st.form_submit_button("Send 💚", use_container_width=True)

    # Clear chat button
    if st.button("🔄 Start New Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! 👋 I'm HealthBot, your friendly health assistant! \n\nI'm here to help you with any health questions you might have - whether it's about symptoms, medications, healthy living, or just general wellness advice.\n\nWhat's on your mind today? 😊"}
        ]
        st.rerun()

    # Process input when form is submitted
    if submitted and user_input:
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate AI response with realistic typing
        bot_container = st.empty()
        try:
            answer = get_ai_response(user_input)
            bot_typing(bot_container, answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = "I'm sorry, I'm having a bit of trouble right now. Please try asking your question again! 😊"
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
