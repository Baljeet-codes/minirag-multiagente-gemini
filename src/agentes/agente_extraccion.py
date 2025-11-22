import os
from typing import List, Optional
from pathlib import Path

# --- Librerías de LangChain ---
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Librería OCR (Más estable que Tesseract para deploy) ---
# Asegúrate de tener: pip install rapidocr-onnxruntime pillow
from rapidocr_onnxruntime import RapidOCR

class AgenteExtraccion:
    """
    Agente encargado de la ingestión de documentos.
    Capacidades:
    1. Leer PDF (Texto nativo).
    2. Leer TXT/MD.
    3. Leer Imágenes (PNG/JPG) usando OCR.
    4. Segmentar (Chunking) el texto para el índice vectorial.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Inicializar el motor OCR una sola vez
        self.ocr_engine = RapidOCR()
        
        # Configurar el segmentador de texto (Chunker)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _ocr_imagen(self, image_path: str) -> str:
        """Procesa una imagen y devuelve el texto extraído."""
        try:
            result, _ = self.ocr_engine(image_path)
            if result:
                # Unir el texto detectado en cada línea
                return "\n".join([line[1] for line in result])
            return ""
        except Exception as e:
            print(f"[OCR] Error procesando imagen {image_path}: {e}")
            return ""

    def _leer_pdf(self, pdf_path: str) -> List[Document]:
        """Lee un PDF usando el loader optimizado de LangChain."""
        try:
            loader = PyPDFLoader(pdf_path)
            return loader.load()
        except Exception as e:
            print(f"[PDF] Error leyendo {pdf_path}: {e}")
            return []

    def _leer_texto(self, txt_path: str) -> List[Document]:
        """Lee archivos de texto plano."""
        try:
            loader = TextLoader(txt_path, encoding="utf-8")
            return loader.load()
        except Exception as e:
            print(f"[TXT] Error leyendo {txt_path}: {e}")
            return []

    def procesar_fuentes(self, paths: List[str]) -> List[Document]:
        """
        Función Principal: Recibe rutas de archivos, extrae contenido 
        y devuelve documentos segmentados (chunks).
        """
        documentos_crudos = []

        print(f"--- Iniciando Extracción de {len(paths)} archivos ---")

        for p in paths:
            ruta = Path(p)
            ext = ruta.suffix.lower()
            
            # 1. Estrategia para PDFs
            if ext == ".pdf":
                docs = self._leer_pdf(str(ruta))
                documentos_crudos.extend(docs)
            
            # 2. Estrategia para Imágenes (OCR)
            elif ext in [".png", ".jpg", ".jpeg"]:
                print(f"Detectada imagen: {ruta.name} - Aplicando OCR...")
                texto = self._ocr_imagen(str(ruta))
                if texto:
                    # Crear documento manual ya que OCR devuelve string puro
                    doc = Document(
                        page_content=texto,
                        metadata={"source": ruta.name, "type": "image"}
                    )
                    documentos_crudos.append(doc)
            
            # 3. Estrategia para Texto Plano
            elif ext in [".txt", ".md"]:
                docs = self._leer_texto(str(ruta))
                documentos_crudos.extend(docs)
            
            else:
                print(f"Formato no soportado: {ruta.name}")

        if not documentos_crudos:
            print("No se extrajo contenido de ningún archivo.")
            return []

        # 4. Segmentación (Chunking)
        # Este paso es vital: convertimos documentos grandes en piezas pequeñas para FAISS
        chunks = self.splitter.split_documents(documentos_crudos)
        
        print(f"--- Procesamiento completado: {len(chunks)} fragmentos generados ---")
        return chunks