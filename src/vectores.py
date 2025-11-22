import os
import time
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model

# Ruta donde se guardará el índice localmente
DB_FAISS_PATH = "data/vectorstore/db_faiss"

def crear_vectorstore(chunks):
    """
    Crea un índice FAISS procesando los chunks en LOTES (Batches)
    para evitar el error 429 (Quota Exceeded) de Google.
    """
    embedding_model = get_embedding_model()
    
    # Configuración de lotes
    BATCH_SIZE = 10  # Procesar 10 fragmentos a la vez
    vectorstore = None
    total_chunks = len(chunks)
    
    print(f"--- Iniciando indexación de {total_chunks} fragmentos ---")

    # Bucle para procesar poco a poco
    for i in range(0, total_chunks, BATCH_SIZE):
        # Seleccionar el lote actual
        lote = chunks[i : i + BATCH_SIZE]
        
        print(f"Procesando lote {i} a {i + len(lote)}...")
        
        if vectorstore is None:
            # Crear el vectorstore con el primer lote
            vectorstore = FAISS.from_documents(documents=lote, embedding=embedding_model)
        else:
            # Añadir los siguientes lotes al vectorstore existente
            vectorstore.add_documents(lote)
        
        # PAUSA DE SEGURIDAD: Esperar 1.5 segundos entre lotes para respetar el límite de Google
        time.sleep(1.5)

    # Guardar localmente una vez terminado
    if vectorstore:
        vectorstore.save_local(DB_FAISS_PATH)
        print("--- Indexación completada y guardada ---")
    
    return vectorstore

def cargar_vectorstore():
    """Carga el índice FAISS existente desde el disco."""
    embedding_model = get_embedding_model()
    
    if not os.path.exists(DB_FAISS_PATH):
        return None

    try:
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(f"Error cargando índice: {e}")
        return None