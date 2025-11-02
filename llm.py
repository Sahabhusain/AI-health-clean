import os
import logging
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
class Config:
    """Configuration settings"""
    MODELS = {
        "fast": "llama-3.1-8b-instant",
        "balanced": "llama-3.1-70b-versatile", 
        "detailed": "meta-llama/llama-4-maverick-17b-128e-instruct"
    }
    DEFAULT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
    MAX_RETRIES = 3
    TIMEOUT = 30

# ========== PROMPT TEMPLATES ==========
PROMPT_TEMPLATES = {
    "default": """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Do not make up an answer.
Only use the given context.

Context: {context}
Question: {question}

Answer:
""",

    "health_expert": """
You are HealthBot, a professional AI health assistant. Use the provided context to give accurate health information.

**Guidelines:**
- Provide clear, factual health information
- Include practical advice when relevant
- Mention if information is from general knowledge vs specific context
- Be empathetic and professional
- If context doesn't contain answer, say so clearly

Context: {context}
Question: {question}

Please provide a helpful, detailed response:
""",

    "detailed_analysis": """
As an expert medical AI, analyze the provided context thoroughly.

CONTEXT: {context}
QUESTION: {question}

Provide a comprehensive analysis including:
1. Key information from context
2. Practical recommendations  
3. Limitations (if any)
4. When to seek professional help

Detailed Response:
"""
}

# ========== CORE FUNCTIONS ==========
def set_custom_prompt(template_type: str = "default") -> PromptTemplate:
    """Set custom prompt based on template type"""
    template = PROMPT_TEMPLATES.get(template_type, PROMPT_TEMPLATES["default"])
    return PromptTemplate(template=template, input_variables=["context", "question"])

def validate_api_key() -> bool:
    """Validate GROQ API key"""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        return False
    return True

@lru_cache(maxsize=100)
def get_cached_response(query: str, model_name: str, template_type: str) -> Optional[Dict]:
    """Cache responses for frequent queries"""
    # Implementation depends on caching strategy
    return None

def get_answer(
    query: str, 
    vectorstore: FAISS, 
    model_name: str = None,
    template_type: str = "default",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    retriever_k: int = 5,
    return_sources: bool = False
) -> Dict:
    """
    Enhanced get_answer function with multiple improvements
    
    Args:
        query: User question
        vectorstore: FAISS vectorstore instance
        model_name: Model to use
        template_type: Prompt template type
        temperature: Creativity level
        max_tokens: Max response length
        retriever_k: Number of context chunks
        return_sources: Whether to return source documents
    
    Returns:
        Dictionary with answer and metadata
    """
    
    # Input validation
    if not query or not query.strip():
        return {
            "answer": "Please provide a valid question.",
            "error": "Empty query",
            "success": False
        }
    
    if not validate_api_key():
        return {
            "answer": "API configuration error. Please check your GROQ API key.",
            "error": "Invalid API key", 
            "success": False
        }
    
    if not vectorstore:
        return {
            "answer": "Knowledge base is not available.",
            "error": "Vectorstore not available",
            "success": False
        }
    
    # Set default model
    if not model_name:
        model_name = Config.DEFAULT_MODEL
    
    try:
        # Check cache first
        cached_response = get_cached_response(query, model_name, template_type)
        if cached_response:
            logger.info("Returning cached response")
            return {**cached_response, "cached": True}
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=os.environ["GROQ_API_KEY"],
                timeout=Config.TIMEOUT
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={'k': retriever_k}
            ),
            return_source_documents=return_sources,
            chain_type_kwargs={
                "prompt": set_custom_prompt(template_type)
            }
        )
        
        # Get response with retry logic
        start_time = time.time()
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = qa_chain.invoke({"query": query})
                break
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
        
        response_time = time.time() - start_time
        
        # Prepare result
        result = {
            "answer": response["result"],
            "success": True,
            "model_used": model_name,
            "response_time": round(response_time, 2),
            "template_used": template_type,
            "cached": False
        }
        
        # Add sources if requested
        if return_sources and "source_documents" in response:
            sources = []
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
            result["sources"] = sources
        
        logger.info(f"Response generated in {response_time:.2f}s using {model_name}")
        return result
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        
        return {
            "answer": "I apologize, but I encountered an error while processing your request.",
            "error": error_msg,
            "success": False,
            "model_used": model_name
        }

# ========== MEDICATION-SPECIFIC FUNCTIONS ==========
def get_medication_info(medication_name: str, vectorstore: FAISS) -> Dict:
    """Get detailed medication information"""
    query = f"""
    Provide comprehensive information about {medication_name} including:
    - Uses and indications
    - Dosage information
    - Side effects
    - Contraindications
    - Drug interactions
    - Important warnings
    """
    
    return get_answer(
        query=query,
        vectorstore=vectorstore,
        template_type="health_expert",
        retriever_k=5
    )

def check_medication_interaction(med1: str, med2: str, vectorstore: FAISS) -> Dict:
    """Check interaction between two medications"""
    query = f"""
    What are the potential interactions between {med1} and {med2}?
    Include:
    - Severity of interaction
    - Mechanism of interaction
    - Recommendations
    - Alternative options if available
    """
    
    return get_answer(
        query=query,
        vectorstore=vectorstore,
        template_type="health_expert",
        retriever_k=5
    )

def get_dosage_guidelines(medication: str, condition: str, vectorstore: FAISS) -> Dict:
    """Get dosage guidelines for specific condition"""
    query = f"""
    What is the recommended dosage of {medication} for {condition}?
    Include:
    - Standard dosage
    - Adjustments for special populations
    - Administration instructions
    - Duration of treatment
    """
    
    return get_answer(
        query=query,
        vectorstore=vectorstore,
        template_type="health_expert"
    )

# ========== HEALTH ANALYSIS FUNCTIONS ==========
def analyze_symptoms(symptoms: str, vectorstore: FAISS) -> Dict:
    """Analyze symptoms and provide possible conditions"""
    query = f"""
    Based on these symptoms: {symptoms}
    Provide:
    - Possible conditions
    - Recommended next steps
    - When to seek emergency care
    - Home care recommendations
    """
    
    return get_answer(
        query=query,
        vectorstore=vectorstore,
        template_type="health_expert",
        retriever_k=5
    )

def provide_first_aid(emergency_type: str, vectorstore: FAISS) -> Dict:
    """Provide first aid guidance for emergencies"""
    query = f"""
    Provide step-by-step first aid instructions for {emergency_type}
    Include:
    - Immediate actions
    - What not to do
    - When to seek professional help
    - Warning signs
    """
    
    return get_answer(
        query=query,
        vectorstore=vectorstore,
        template_type="health_expert"
    )

# ========== UTILITY FUNCTIONS ==========
def get_available_models() -> List[str]:
    """Get list of available models"""
    return list(Config.MODELS.values())

def get_model_info(model_name: str) -> Dict:
    """Get information about a model"""
    for key, value in Config.MODELS.items():
        if value == model_name:
            return {
                "name": value,
                "type": key,
                "description": f"{key.capitalize()} model for {value}"
            }
    return {}

def format_response_for_display(response: Dict) -> str:
    """Format response for better UI display"""
    if not response.get("success", False):
        return f"❌ {response.get('answer', 'Error occurred')}"
    
    answer = response["answer"]
    
    # Add metadata if available
    if response.get("sources"):
        answer += f"\n\n📚 Sources: {len(response['sources'])} documents referenced"
    
    if response.get("response_time"):
        answer += f"\n⏱️ Response time: {response['response_time']}s"
    
    return answer

# ========== MAIN GUARD ==========
if __name__ == "__main__":
    # Test the enhanced functions
    print("🚀 Enhanced LLM Module Loaded!")
    print("📋 Available Models:", get_available_models())
    print("✅ API Key Valid:", validate_api_key())
