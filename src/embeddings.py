import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

@st.cache_resource
def get_embedding_model():
    """
    Devuelve el modelo de Embeddings de Google Gemini.
    Usamos st.cache_resource para no recargar el modelo en cada interacción.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY no encontrada en variables de entorno")

    # CAMBIO: Usamos 'models/text-embedding-004' que es más estable y actual.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key,
        task_type="retrieval_document" # Optimiza para búsqueda
    )
    return embeddings