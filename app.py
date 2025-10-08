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
                DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
            )

        if os.path.exists(DATA_PATH):
            loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
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
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        health_prompt = f"""
        You are HealthBot, a professional AI health assistant.
        Provide accurate, helpful health information.

        User Question: {question}

        Please provide:
        1. Clear, factual health information
        2. Practical advice and tips
        3. Helpful recommendations
        """
        response = llm.invoke(health_prompt)
        return response.content
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

# -------- Typing effect --------
def bot_typing(container, text, delay=0.03):
    """Typing animation"""
    thinking_time = min(1.5, len(text) * 0.01)
    time.sleep(thinking_time)

    typing_indicator = st.empty()
    typing_indicator.markdown(
        """
        <div style='display:flex; align-items:flex-start; margin-bottom:12px;'>
            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            width:42px;height:42px;border-radius:50%;display:flex;
            align-items:center;justify-content:center;margin-right:12px;
            box-shadow:0 4px 12px rgba(102,126,234,0.3);'>
                <span style='color:white;font-size:20px;'>🤖</span>
            </div>
            <div style='color:#666;background:#f8f9fa;padding:12px 16px;border-radius:18px;
            border:1px solid #e9ecef;font-style:italic;'>HealthBot is typing...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1)
    typing_indicator.empty()

    message_container = container.empty()
    typed = ""
    for char in text:
        typed += char
        message_container.markdown(
            f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:16px;'>
                <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                width:42px;height:42px;border-radius:50%;display:flex;
                align-items:center;justify-content:center;margin-right:12px;
                box-shadow:0 4px 12px rgba(102,126,234,0.3);'>
                    <span style='color:white;font-size:20px;'>🤖</span>
                </div>
                <div style='color:#2c3e50;background:linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);
                padding:16px 20px;border-radius:20px;max-width:75%;line-height:1.6;font-size:15px;
                box-shadow:0 4px 12px rgba(0,0,0,0.1);border:1px solid #e0e0e0;'>
                    <div style='font-weight:600;color:#667eea;font-size:13px;margin-bottom:4px;'>HealthBot</div>
                    {typed}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(delay * random.uniform(0.5, 1.5))

# -------- Display message --------
def display_message(msg):
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex;justify-content:flex-end;align-items:flex-start;margin-bottom:16px;'>
                <div style='color:white;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:16px 20px;border-radius:20px;max-width:75%;line-height:1.6;font-size:15px;
                box-shadow:0 4px 12px rgba(102,126,234,0.3);'>
                    <div style='font-weight:600;color:rgba(255,255,255,0.9);font-size:13px;margin-bottom:4px;'>You</div>
                    {msg['content']}
                </div>
                <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                width:42px;height:42px;border-radius:50%;display:flex;
                align-items:center;justify-content:center;margin-left:12px;
                box-shadow:0 4px 12px rgba(102,126,234,0.3);'>
                    <span style='color:white;font-size:20px;'>👤</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:16px;'>
                <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                width:42px;height:42px;border-radius:50%;display:flex;
                align-items:center;justify-content:center;margin-right:12px;
                box-shadow:0 4px 12px rgba(102,126,234,0.3);'>
                    <span style='color:white;font-size:20px;'>🤖</span>
                </div>
                <div style='color:#2c3e50;background:linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);
                padding:16px 20px;border-radius:20px;max-width:75%;line-height:1.6;font-size:15px;
                box-shadow:0 4px 12px rgba(0,0,0,0.1);border:1px solid #e0e0e0;'>
                    <div style='font-weight:600;color:#667eea;font-size:13px;margin-bottom:4px;'>HealthBot</div>
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------- Clear chat --------
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm HealthBot, your AI health assistant. How can I help you today? 😊"}
    ]

# -------- Get AI Response --------
def get_ai_response(question):
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
    except Exception:
        return get_direct_ai_response(question)

# -------- Main App --------
def main():
    st.set_page_config(page_title="HealthBot - AI Health Assistant", page_icon="🏥", layout="wide")

    if "messages" not in st.session_state:
        clear_chat()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align:center;color:#667eea;'>🏥 HealthBot AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#666;'>Your AI-powered health assistant</p>", unsafe_allow_html=True)
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                display_message(msg)

        st.markdown("---")
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            with col_input:
                user_input = st.text_input("Ask your health question...", key="user_input", label_visibility="collapsed")
            with col_send:
                submitted = st.form_submit_button("Send 🚀", use_container_width=True)

        # --- Handle user message ---
        if submitted and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            chat_container.empty()
            with chat_container:
                for msg in st.session_state.messages:
                    display_message(msg)

            st.markdown("""
                <script>
                const chatEnd = document.createElement('div');
                chatEnd.id = 'chat-end';
                document.body.appendChild(chatEnd);
                chatEnd.scrollIntoView({ behavior: 'smooth', block: 'end' });
                </script>
            """, unsafe_allow_html=True)
            st.rerun()

        # --- Handle bot response ---
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_message = st.session_state.messages[-1]["content"]
            bot_container = st.empty()
            try:
                bot_typing(bot_container, "Thinking...")
                answer = get_ai_response(user_message)
                bot_container.empty()
                bot_typing(bot_container, answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                st.markdown("""
                    <script>
                    const chatEnd = document.createElement('div');
                    chatEnd.id = 'chat-end';
                    document.body.appendChild(chatEnd);
                    chatEnd.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    </script>
                """, unsafe_allow_html=True)
                st.rerun()

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                st.rerun()

if __name__ == "__main__":
    main()
