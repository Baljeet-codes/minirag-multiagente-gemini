import streamlit as st
import os
from pathlib import Path

# --- Importación de Módulos Propios ---
# Asegúrate de que estos archivos existan en tu carpeta src/
from src.vectores import cargar_vectorstore, crear_vectorstore
from src.agentes.agente_extraccion import AgenteExtraccion
from src.agentes.agente_analisis import AgenteAnalisis
from src.agentes.agente_respuesta import generar_respuesta_rag  # <--- VERSIÓN SÍNCRONA

# --- Configuración de la Página ---
st.set_page_config(page_title="Mini RAG con Gemini", layout="wide")

# --- Configuración de Rutas ---
# Define la raíz del proyecto subiendo un nivel desde src/
ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = ROOT / "data" / "documentos"
# Crea la carpeta si no existe (reemplaza a utils.ensure_dirs)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Inicialización del Estado (Session State) ---
if "vectorstore" not in st.session_state:
    # Intenta cargar un índice existente al arrancar
    st.session_state.vectorstore = cargar_vectorstore()

# ==========================================
#  BARRA LATERAL: INGESTA (Agente Extracción)
# ==========================================
with st.sidebar:
    st.header("📂 Gestión Documental")
    
    # Widget de subida de archivos
    uploaded_files = st.file_uploader(
        "Sube tus PDFs, TXTs o Imágenes (PNG/JPG).", 
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if st.button("Procesar e Indexar"):
        if not uploaded_files:
            st.warning("⚠️ Por favor selecciona archivos primero.")
        else:
            with st.spinner("🔄 El Agente de Extracción está procesando (OCR + Chunking)..."):
                try:
                    rutas_archivos = []
                    
                    # 1. Guardar archivos temporalmente en disco
                    for archivo in uploaded_files:
                        ruta_destino = DATA_DIR / archivo.name
                        with open(ruta_destino, "wb") as f:
                            f.write(archivo.getbuffer())
                        rutas_archivos.append(str(ruta_destino))

                    # 2. Invocar al Agente de Extracción
                    agente_ext = AgenteExtraccion()
                    chunks = agente_ext.procesar_fuentes(rutas_archivos)

                    if chunks:
                        st.info(f"🧩 Se generaron {len(chunks)} fragmentos. Indexando en FAISS...")
                        
                        # 3. Guardar en Vector Store (FAISS) con Batching
                        vectorstore = crear_vectorstore(chunks)
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ ¡Base de datos actualizada y lista!")
                    else:
                        st.error("❌ No se pudo extraer texto legible de los archivos.")
                
                except Exception as e:
                    st.error(f"Ocurrió un error durante la ingesta: {e}")

    st.markdown("---")
    st.subheader("Configuración de Búsqueda")
    k_retrieval = st.slider("Fragmentos de contexto (k)", min_value=1, max_value=10, value=3)

# ==========================================
#  ÁREA PRINCIPAL: CHAT (Análisis + Respuesta)
# ==========================================
st.title("🤖 Mini RAG con Gemini")
st.markdown("Sistema multiagente para consulta documental.")

# Input del usuario
pregunta = st.chat_input("Haz una pregunta sobre tus documentos...")

if pregunta:
    # Verificación inicial
    if not st.session_state.vectorstore:
        st.warning("⚠️ El índice está vacío. Sube documentos en la barra lateral para comenzar.")
    else:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.write(pregunta)

        # --- FLUJO DE AGENTES ---
        with st.chat_message("assistant"):
            try:
                # 1. AGENTE DE ANÁLISIS: Recuperación
                agente_analisis = AgenteAnalisis(st.session_state.vectorstore)
                
                # Recuperar documentos (para mostrar al usuario)
                docs_rel = agente_analisis.recuperar_contexto(pregunta, k=k_retrieval)
                
                # Mostrar transparencia (qué encontró el agente)
                with st.status("🔍 Agente de Análisis trabajando...", expanded=False) as status:
                    if docs_rel:
                        st.write(f"Encontré {len(docs_rel)} fragmentos relevantes:")
                        for i, doc in enumerate(docs_rel):
                            fuente = doc.metadata.get('source', 'Desconocido')
                            st.text(f"[{i+1}] Fuente: {fuente}\n{doc.page_content[:200]}...")
                        status.update(label="Contexto recuperado ✅", state="complete")
                    else:
                        status.update(label="No se encontró contexto relevante ⚠️", state="error")

                # 2. AGENTE DE RESPUESTA: Generación (Gemini)
                # Obtener el retriever oficial para LangChain
                retriever = agente_analisis.obtener_retriever(k=k_retrieval)
                
                # Espacio para la respuesta
                contenedor_respuesta = st.empty()
                
                with st.spinner("✍️ El Agente de Respuesta está redactando..."):
                    # LLAMADA SÍNCRONA (Soluciona el error 'Event loop is closed')
                    respuesta_final = generar_respuesta_rag(pregunta, retriever)
                    
                    # Mostrar resultado
                    contenedor_respuesta.markdown(respuesta_final)

            except Exception as e:
                st.error(f"❌ Ocurrió un error en el flujo de agentes: {e}")