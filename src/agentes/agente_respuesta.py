import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# --- 1. Prompt ---
PROMPT_TEMPLATE = """
Eres un asistente experto en responder preguntas usando únicamente el contexto proporcionado.
Tu objetivo es ser conciso, preciso y útil.

Si la respuesta no se encuentra en el contexto, di: "La información requerida no se encuentra en los documentos proporcionados."

CONTEXTO:
---
{context}
---
PREGUNTA: {question}
"""

# --- 2. Configuración LLM ---
@st.cache_resource
def get_llm_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.0
    )
    return llm

# --- 3. Generación de Respuesta (SÍNCRONA) ---
def generar_respuesta_rag(pregunta: str, retriever: BaseRetriever) -> str:
    """
    Versión síncrona (invoke) para evitar errores de Event Loop en Streamlit.
    """
    llm = get_llm_model()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Cadena LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Usamos .invoke() en lugar de .ainvoke()
    try:
        respuesta = rag_chain.invoke(pregunta)
        return respuesta
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"