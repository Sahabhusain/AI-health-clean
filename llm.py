import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Do not make up an answer.
Only use the given context.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(template):
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_answer(query, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            groq_api_key=os.environ["GROQ_API_KEY"]
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    response = qa_chain.invoke({"query": query})
    return response["result"]
