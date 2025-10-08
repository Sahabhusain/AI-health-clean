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
    # Show typing indicator
    with container:
        typing_container = st.empty()
        
        # Show different typing states
        typing_states = ["✏️ Typing...", "✏️ Still typing...", "✏️ Almost done..."]
        for state in typing_states:
            typing_container.markdown(f"""
                <div style='display: flex; align-items: center; margin-bottom: 10px; padding: 8px 12px; background: #f0f0f0; border-radius: 15px; width: fit-content;'>
                    <div style='font-size: 12px; color: #666;'>{state}</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.8)
    
    typing_container.empty()
    
    # Type out message
    message_container = container.empty()
    displayed_text = ""
    
    # Type word by word for more natural feel
    words = text.split()
    
    for i, word in enumerate(words):
        displayed_text += word + " "
        
        # Create message bubble
        message_container.markdown(f"""
            <div style='display: flex; align-items: flex-start; margin-bottom: 8px;'>
                <div style='background: #0084ff; color: white; padding: 8px 12px; border-radius: 18px; max-width: 70%; margin-right: auto;'>
                    <div style='font-size: 14px; line-height: 1.4;'>{displayed_text.strip()}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Vary typing speed
        if i < len(words) - 1:
            next_word = words[i + 1]
            # Longer pause after sentences
            if word.endswith(('.', '!', '?')):
                time.sleep(0.3)
            # Shorter pause between words
            else:
                time.sleep(0.08 + random.random() * 0.04)

# -------- Display Messages --------
def display_message(msg):
    if msg["role"] == "user":
        # User message - Right side (blue)
        st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 8px;'>
                <div style='background: #0084ff; color: white; padding: 8px 12px; border-radius: 18px; max-width: 70%;'>
                    <div style='font-size: 14px; line-height: 1.4;'>{msg['content']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Bot message - Left side (light gray)
        st.markdown(f"""
            <div style='display: flex; margin-bottom: 8px;'>
                <div style='background: #f0f0f0; color: #333; padding: 8px 12px; border-radius: 18px; max-width: 70%;'>
                    <div style='font-size: 14px; line-height: 1.4;'>{msg['content']}</div>
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
        page_title="HealthBot",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Modern chat app CSS
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            height: 70vh;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: white;
            font-size: 24px;
            margin: 0;
            font-weight: 600;
        }
        
        .header p {
            color: rgba(255,255,255,0.8);
            margin: 5px 0 0 0;
            font-size: 14px;
        }
        
        .input-container {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 12px 16px;
            font-size: 14px;
            border: 1px solid #ddd;
        }
        
        .stButton>button {
            border-radius: 20px;
            padding: 12px 24px;
            font-weight: 500;
            background: #0084ff;
            color: white;
            border: none;
        }
        
        /* Scrollbar */
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
            {"role": "assistant", "content": "Hey there! 👋 I'm HealthBot, your friendly health assistant. How can I help you today?"}
        ]

    # Header
    st.markdown("""
        <div class="header">
            <h1>💬 HealthBot</h1>
            <p>Your friendly health assistant</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        display_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Quick suggestions
    if len(st.session_state.messages) <= 1:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "I have a headache",
            "Can't sleep well", 
            "Healthy diet tips",
            "Exercise advice"
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.quick_question = suggestion
                    st.rerun()

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
                "Type a message...",
                value=current_input,
                placeholder="Ask me anything about health...",
                key="user_input",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I'm having trouble responding. Please try again! 😊"})
        
        st.rerun()

if __name__ == "__main__":
    main()
