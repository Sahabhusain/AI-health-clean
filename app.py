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
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None

# -------- Direct AI Response --------
def get_direct_ai_response(question):
    """Get response directly from AI"""
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
        
        Provide a detailed, informative response:
        """
        
        response = llm.invoke(health_prompt)
        return response.content
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

# -------- Get AI Response --------
def get_ai_response(question):
    """Get response from AI"""
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
            
    except Exception as e:
        return get_direct_ai_response(question)

# -------- MODERN UI COMPONENTS --------
def display_modern_message(msg):
    """Modern message display with animations"""
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='
                display: flex;
                justify-content: flex-end;
                margin: 20px 0;
                animation: slideInRight 0.5s ease-out;
            '>
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 16px 22px;
                    border-radius: 22px 22px 6px 22px;
                    max-width: 75%;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
                    border: 1px solid rgba(255,255,255,0.3);
                    backdrop-filter: blur(10px);
                    position: relative;
                '>
                    <div style='font-size: 11px; opacity: 0.8; margin-bottom: 6px; display: flex; align-items: center; gap: 5px;'>
                        <span>👤</span> You
                    </div>
                    <div style='font-size: 15px; line-height: 1.5;'>
                        {msg['content']}
                    </div>
                    <div style='
                        position: absolute;
                        bottom: -8px;
                        right: 10px;
                        width: 0;
                        height: 0;
                        border-left: 10px solid transparent;
                        border-right: 10px solid transparent;
                        border-top: 10px solid #764ba2;
                    '></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='
                display: flex;
                align-items: start;
                margin: 20px 0;
                animation: slideInLeft 0.5s ease-out;
            '>
                <div style='
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 16px 22px;
                    border-radius: 22px 22px 22px 6px;
                    max-width: 75%;
                    box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4);
                    border: 1px solid rgba(255,255,255,0.3);
                    backdrop-filter: blur(10px);
                    position: relative;
                '>
                    <div style='font-size: 11px; opacity: 0.8; margin-bottom: 6px; display: flex; align-items: center; gap: 5px;'>
                        <span>⚕️</span> HealthBot Pro
                    </div>
                    <div style='font-size: 15px; line-height: 1.5;'>
                        {msg['content']}
                    </div>
                    <div style='
                        position: absolute;
                        bottom: -8px;
                        left: 10px;
                        width: 0;
                        height: 0;
                        border-left: 10px solid transparent;
                        border-right: 10px solid transparent;
                        border-top: 10px solid #f5576c;
                    '></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_modern_health_topics():
    """Modern health topics grid"""
    topics = [
        {"icon": "🤒", "title": "Symptoms Check", "desc": "Understand your symptoms", "color": "#FF6B6B"},
        {"icon": "💊", "title": "Medications", "desc": "Drug information & side effects", "color": "#4ECDC4"},
        {"icon": "🏃", "title": "Fitness", "desc": "Exercise & workout plans", "color": "#45B7D1"},
        {"icon": "🥗", "title": "Nutrition", "desc": "Diet & healthy eating", "color": "#96CEB4"},
        {"icon": "😴", "title": "Sleep", "desc": "Sleep quality improvement", "color": "#FECA57"},
        {"icon": "🧘", "title": "Mental Health", "desc": "Stress & anxiety management", "color": "#FF9FF3"}
    ]
    
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 20px;
            margin: 25px 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        '>
            <h2 style='margin: 0; font-size: 1.8rem;'>🌟 Quick Health Topics</h2>
            <p style='margin: 5px 0 0 0; opacity: 0.9;'>Click any topic to get started</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            if st.button(
                f"{topic['icon']} {topic['title']}",
                key=f"topic_{i}",
                help=topic['desc'],
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.quick_question = f"Tell me about {topic['title'].lower()} and best practices"
                st.rerun()

def show_modern_typing_animation():
    """Modern typing animation"""
    with st.empty():
        st.markdown(
            """
            <div style='
                display: flex;
                align-items: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                margin: 15px 0;
                animation: pulse 2s infinite;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                color: white;
            '>
                <div style='
                    width: 50px;
                    height: 50px;
                    background: rgba(255,255,255,0.2);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 20px;
                    backdrop-filter: blur(10px);
                '>
                    <span style='font-size: 20px;'>⚕️</span>
                </div>
                <div style='flex: 1;'>
                    <div style='font-weight: 600; margin-bottom: 8px; font-size: 16px;'>HealthBot Pro is analyzing</div>
                    <div style='display: flex; gap: 6px;'>
                        <div style='width: 10px; height: 10px; background: white; border-radius: 50%; animation: bounce 1.4s infinite;'></div>
                        <div style='width: 10px; height: 10px; background: white; border-radius: 50%; animation: bounce 1.4s infinite 0.2s;'></div>
                        <div style='width: 10px; height: 10px; background: white; border-radius: 50%; animation: bounce 1.4s infinite 0.4s;'></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1.5)

def create_modern_emergency_section():
    """Modern emergency contact section"""
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            color: white;
            padding: 25px;
            border-radius: 20px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        '>
            <h3 style='margin: 0 0 15px 0; font-size: 1.4rem;'>🚨 Emergency Contacts</h3>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px;'>
                <div>
                    <div style='font-weight: 600;'>Medical Emergency</div>
                    <div style='font-size: 1.2rem;'>📞 108</div>
                </div>
                <div>
                    <div style='font-weight: 600;'>Police</div>
                    <div style='font-size: 1.2rem;'>📞 100</div>
                </div>
                <div>
                    <div style='font-weight: 600;'>Fire Service</div>
                    <div style='font-size: 1.2rem;'>📞 101</div>
                </div>
            </div>
            <p style='margin: 15px 0 0 0; opacity: 0.9; font-size: 12px;'>
                In case of emergency, contact these numbers immediately
            </p>
        </div>
    """, unsafe_allow_html=True)

def create_health_tools():
    """Interactive health tools"""
    with st.expander("🛠️ Health Tools", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 BMI Calculator")
            weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0, key="weight")
            height = st.number_input("Height (m)", min_value=0.1, value=1.7, key="height")
            if st.button("Calculate BMI", key="bmi_calc"):
                bmi = weight / (height ** 2)
                st.metric("Your BMI", f"{bmi:.1f}")
                if bmi < 18.5:
                    st.warning("Underweight - Consider consulting a nutritionist")
                elif bmi < 25:
                    st.success("Normal weight - Keep it up!")
                elif bmi < 30:
                    st.warning("Overweight - Consider lifestyle changes")
                else:
                    st.error("Obese - Please consult a healthcare provider")
        
        with col2:
            st.subheader("💧 Water Intake")
            weight_kg = st.number_input("Your Weight (kg)", min_value=1.0, value=65.0, key="water_weight")
            activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Heavy"])
            if st.button("Calculate Water Need", key="water_calc"):
                base_water = weight_kg * 0.033
                activity_multiplier = {"Sedentary": 1.0, "Light": 1.2, "Moderate": 1.5, "Heavy": 2.0}
                total_water = base_water * activity_multiplier[activity]
                st.metric("Daily Water Need", f"{total_water:.1f} Liters")

# -------- MAIN APP --------
def main():
    st.set_page_config(
        page_title="HealthBot Pro - Advanced AI Health Assistant",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern CSS with animations
    st.markdown("""
        <style>
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
            50% { transform: scale(1.02); box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5); }
            100% { transform: scale(1); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .main-header {
            font-size: 3.5rem;
            background: linear-gradient(45deg, #FF6B6B, #667eea, #764ba2, #f093fb);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 3s ease infinite;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
        }
        
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.3rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        .status-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 20px;
            margin: 15px 0;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 18px 25px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
            background: white;
        }
        
        .stButton>button {
            border-radius: 25px;
            padding: 15px 30px;
            font-weight: 600;
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            color: white;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, #764ba2, #667eea);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 **Welcome to HealthBot Pro!** 

I'm your advanced AI health assistant, here to provide you with:

🔬 **Accurate Health Information**
💊 **Medication Guidance** 
🏃 **Fitness & Nutrition Advice**
😴 **Sleep & Mental Health Support**
📊 **Symptom Analysis**

**How can I assist you with your health today?**"""
            }
        ]

    # Modern Sidebar
    with st.sidebar:
        # Premium Header
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #FF6B6B 0%, #f093fb 50%, #667eea 100%);
                color: white;
                padding: 30px 20px;
                border-radius: 20px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
            '>
                <div style='font-size: 3rem; margin-bottom: 10px;'>⚕️</div>
                <h1 style='margin: 0; font-size: 1.8rem; font-weight: 800;'>HealthBot Pro</h1>
                <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9rem;'>Advanced AI Health Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # System Status
        vectorstore = build_vectorstore()
        status_icon = "🔮" if vectorstore else "⚡"
        status_text = "**Enhanced Mode** with Medical Database" if vectorstore else "**Basic Mode** - General AI Knowledge"
        st.markdown(f"""
            <div class='status-card'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                    <div style='font-size: 1.5rem;'>{status_icon}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>System Status</div>
                </div>
                <div style='font-size: 16px; font-weight: 600;'>{status_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Statistics
        st.markdown("### 📊 Chat Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages), delta="+1")
        with col2:
            st.metric("Status", "🟢 Active")
        
        # Health Tools
        create_health_tools()
        
        # Emergency Section
        create_modern_emergency_section()
        
        # Clear Chat
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
            st.session_state.messages = [
                {"role": "assistant", "content": "🗣️ Conversation cleared! I'm ready to help with your health questions."}
            ]
            st.rerun()

    # Main Content Area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Premium Header
        st.markdown('<h1 class="main-header">HealthBot Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your Intelligent AI Health Companion</p>', unsafe_allow_html=True)
        
        # Health Topics Grid
        create_modern_health_topics()
        
        st.markdown("---")
        
        # Enhanced Chat Container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                display_modern_message(msg)
        
        # Premium Input Area
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 25px;
                border-radius: 25px;
                margin-top: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.5);
                backdrop-filter: blur(10px);
            '>
        """, unsafe_allow_html=True)
        
        # Handle quick questions
        current_input_value = ""
        if hasattr(st.session_state, 'quick_question'):
            current_input_value = st.session_state.quick_question
            del st.session_state.quick_question
        
        # Premium Input Form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            
            with col_input:
                user_input = st.text_input(
                    "💭 Ask your health question...",
                    value=current_input_value,
                    placeholder="Describe symptoms, ask about medications, nutrition, exercise, or mental health...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                submitted = st.form_submit_button(
                    "Send 🚀", 
                    use_container_width=True,
                    type="primary"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                color: #856404;
                padding: 15px;
                border-radius: 15px;
                margin-top: 20px;
                text-align: center;
                font-size: 12px;
                border-left: 4px solid #ffc107;
            '>
                ⚠️ <strong>Disclaimer:</strong> This AI assistant provides health information for educational purposes only. 
                Always consult healthcare professionals for medical advice and emergencies.
            </div>
        """, unsafe_allow_html=True)
        
        # Process input
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show user message immediately
            st.rerun()
            
            # Show enhanced typing animation
            show_modern_typing_animation()
            
            # Generate AI response
            try:
                answer = get_ai_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"""❌ **I apologize for the inconvenience**

I'm currently experiencing technical difficulties. Please:

1. **Check your internet connection**
2. **Try again in a moment**
3. **Contact support if the issue persists**

*Error details: {str(e)}*"""
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

if __name__ == "__main__":
    main()
