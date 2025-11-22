from typing import List
# Importamos la interfaz base de vectorstore
from langchain_core.vectorstores import VectorStore
# Importamos Document de LangChain Core
from langchain_core.documents import Document

class AgenteAnalisis:
    """
    Agente encargado de la Recuperación y Análisis (Inspección) del Contexto.
    Su rol es obtener documentos de la base de datos para:
    1. Mostrar al usuario qué se recuperó (Transparencia).
    2. Entregar el 'Retriever' al Agente de Respuesta.
    """
    
    def __init__(self, vectorstore: VectorStore):
        # vectorstore es ahora el objeto FAISS estándar cargado desde vectores.py
        self.vectorstore = vectorstore

    def recuperar_contexto(self, pregunta: str, k: int = 4) -> List[Document]:
        """
        Consulta la base de datos (FAISS) usando la búsqueda por similitud
        y devuelve los documentos recuperados.
        """
        if not self.vectorstore:
            return []
        
        try:
            # Usamos el método nativo de FAISS/VectorStore para la búsqueda.
            # search_type="similarity" es el método por defecto (distancia del coseno/euclidiana)
            return self.vectorstore.similarity_search(pregunta, k=k)
        except Exception as e:
            print(f"Error en AgenteAnalisis al recuperar contexto: {e}")
            return []
            
    def obtener_retriever(self, k: int = 4):
        """
        Expone el retriever de LangChain necesario para encadenar con el Agente de Respuesta.
        """
        if not self.vectorstore:
            return None
            
        # El objeto FAISS (self.vectorstore) ya tiene el método as_retriever()
        return self.vectorstore.as_retriever(search_kwargs={"k": k})