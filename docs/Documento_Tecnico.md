# Documento Técnico: Mini RAG Multiagente con Gemini

## 1. Introducción y Problema
En la gestión documental moderna, el análisis manual de grandes volúmenes de información (PDFs, textos e imágenes) es ineficiente y propenso a errores. Además, muchas soluciones de RAG (Retrieval-Augmented Generation) fallan al intentar procesar información contenida dentro de imágenes (como facturas escaneadas o gráficos).

**El Problema:** Se requiere un sistema unificado que permita la ingesta multimodal (texto e imagen), la indexación semántica y la consulta en lenguaje natural, garantizando que la IA responda estrictamente basada en los documentos proporcionados para evitar alucinaciones.

**La Solución:** Este proyecto implementa un sistema "Mini RAG" basado en una **arquitectura multiagente** que orquesta la extracción (incluyendo OCR), el análisis de relevancia y la generación de respuestas.

---

## 2. Metodología y Herramientas
El sistema fue construido utilizando un enfoque modular en Python, priorizando la estabilidad de las dependencias y la eficiencia de costos.

### Stack Tecnológico:
* **Lenguaje:** Python 3.x.
* **Orquestación de IA:** LangChain (v0.2.x) para la gestión de cadenas y agentes.
* **LLM y Embeddings:** Google Gemini (Modelo `gemini-2.5-flash` para generación y `text-embedding-004` para vectores) vía `langchain-google-genai`.
* **Base de Datos Vectorial:** FAISS (Facebook AI Similarity Search) corriendo en CPU para almacenamiento local y recuperación rápida.
* **Interfaz de Usuario:** Streamlit para una experiencia web interactiva.
* **OCR (Reconocimiento Óptico de Caracteres):** `RapidOCR` (vía ONNX Runtime) para extraer texto de imágenes (PNG/JPG) de forma local y ligera.

### Implementación de Embeddings y Similitud

El requisito de comparación de textos se cumple mediante la integración de la generación de vectores con la base de datos **FAISS**.

1.  **Generación de Embeddings:** Los fragmentos de texto (**chunks**) son transformados en vectores numéricos de alta dimensión utilizando el modelo **Google `text-embedding-004`**. Estos vectores capturan el **significado semántico** del texto.
2.  **Cálculo de Similitud:** La búsqueda de contexto la gestiona FAISS. Cuando el **Agente de Análisis** recibe una pregunta, FAISS calcula la distancia matemática entre el vector de la pregunta y todos los vectores de los documentos indexados. Por defecto, FAISS utiliza la métrica **Distancia Euclidiana (L2)**. Al devolver los fragmentos con la distancia L2 más baja, se garantiza que los resultados sean los semánticamente más cercanos a la consulta del usuario.

### Estrategia de Desarrollo:
1.  **Ingesta Robusta:** Se implementó una lógica de "Chunking" (segmentación) recursiva para manejar documentos extensos.
2.  **Manejo de Rate Limits:** Se aplicó procesamiento por lotes (**Batching**) en la creación de embeddings para respetar los límites del plan gratuito de Google API.
3.  **Transparencia:** La interfaz muestra explícitamente qué fragmentos de texto recuperó el sistema antes de generar la respuesta.

---

## 3. Arquitectura del Sistema
El sistema sigue un patrón de **Arquitectura Multiagente Secuencial**, donde tres componentes especializados colaboran en un flujo lineal.

### Diagrama de Flujo de Información:



```mermaid
graph TD
    User[Usuario] -->|Sube Archivos| App(Streamlit UI)
    App -->|Archivos| Agente1[Agente de Extracción]
    
    subgraph "Capa de Ingesta"
        Agente1 -->|Detecta Tipo| Router{¿Es Imagen?}
        Router -- Sí --> OCR[RapidOCR Engine]
        Router -- No (PDF/TXT) --> Loader[PyPDF / TextLoader]
        OCR --> Texto
        Loader --> Texto
        Texto --> Splitter[Text Splitter] --> Chunks
    end
    
    Chunks -->|Batching| Embeddings[Google Embeddings]
    Embeddings --> FAISS[(Vector Store FAISS)]
    
    User -->|Pregunta| Agente2[Agente de Análisis]
    
    subgraph "Capa de Recuperación"
        Agente2 -->|Consulta Semántica| FAISS
        FAISS -->|Top K Fragmentos| Contexto[Contexto Recuperado]
    end
    
    Contexto --> Agente3[Agente de Respuesta]
    User -->|Pregunta| Agente3
    
    subgraph "Capa de Generación"
        Agente3 -->|Prompt + Contexto| Gemini[Google Gemini LLM]
        Gemini -->|Respuesta Natural| Respuesta
    end
    
    Respuesta --> App